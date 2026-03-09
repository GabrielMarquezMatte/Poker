#ifndef __POKER_RL_FEATURIZER_HPP__
#define __POKER_RL_FEATURIZER_HPP__
#include <dlib/matrix.h>
#include <span>
#include <cmath>
#include <BS_thread_pool.hpp>
#include "../game/game.hpp"
#include "../game.hpp"

// Normalization caps for feature scaling
static constexpr float kMaxToCallBB   = 100.f;  // to-call cap in big blinds
static constexpr float kMaxStackBB    = 200.f;  // stack cap in big blinds
static constexpr float kMaxBetToPot   = 3.f;    // bet-to-pot cap (3× pot)
static constexpr float kMaxSPR        = 20.f;   // stack-to-pot ratio cap
static constexpr float kMaxRaiseToPot = 2.f;    // min-raise-to-pot cap (2× pot)

// 32-feature featurizer. Feature layout:
//   k=0-3   : Street one-hot (PreFlop/Flop/Turn/River)
//   k=4-9   : Betting situation (can_check, to_call, pot, pot_odds, facing_bet, bet_to_pot)
//   k=10-15 : Stack info (hero_stack, spr, committed_ratio, avg_opp_stack, eff_stack, can_raise)
//   k=16-18 : Game state (alive_frac, elig_frac, position)
//   k=19    : Equity (raw Monte Carlo win probability)
//   k=20-24 : Decision features (+EV, EV_margin, short_stack_value, fold_indicator, raise_indicator)
//   k=25-31 : Independent structural features (see comments in code)
inline dlib::matrix<float> featurize(const Game &g, std::size_t heroIdx, const Blinds &blinds, BS::thread_pool<BS::tp::none> &pool)
{
    const auto bb = std::max<std::uint32_t>(1, blinds.bigBlind);
    const float bb_f = static_cast<float>(bb);
    auto const &ps = g.players();
    const auto &hero = ps[heroIdx];
    const auto &bd = g.betData();

    // Clamp-and-normalise helpers: norm(v, cap) → [0,1]; norm_bb divides by bb first.
    auto norm    = [](float v, float cap) { return std::min(v, cap) / cap; };
    auto norm_bb = [&](float v, float cap) { return norm(v / bb_f, cap); };

    // Player counts
    int alive = 0, elig = 0;
    float total_opp_chips = 0.f;
    float max_opp_chips = 0.f;
    float min_opp_chips = 1e9f;
    for (auto const &p : ps)
    {
        if (p.alive())
            ++alive;
        if (p.eligible())
            ++elig;
        if (p.id != heroIdx && p.alive())
        {
            total_opp_chips += static_cast<float>(p.chips);
            max_opp_chips = std::max(max_opp_chips, static_cast<float>(p.chips));
            min_opp_chips = std::min(min_opp_chips, static_cast<float>(p.chips));
        }
    }
    if (min_opp_chips > 1e8f) min_opp_chips = 0.f;
    float avg_opp_stack = (alive > 1) ? total_opp_chips / static_cast<float>(alive - 1) : 0.f;

    // Betting situation
    std::uint32_t to_call = (bd.currentBet > hero.committed) ? (bd.currentBet - hero.committed) : 0;
    float pot_odds = (bd.pot + to_call > 0) ? static_cast<float>(to_call) / static_cast<float>(bd.pot + to_call) : 0.f;
    float spr = (bd.pot > 0) ? static_cast<float>(hero.chips) / static_cast<float>(bd.pot) : kMaxSPR;
    spr = std::min(spr, kMaxSPR);

    // Effective stack (minimum of hero and average opponent)
    float effective_stack = std::min(static_cast<float>(hero.chips), avg_opp_stack);

    // Street encoding
    int street_idx = 0;
    switch (g.state())
    {
    case GameState::PreFlop: street_idx = 0; break;
    case GameState::Flop:    street_idx = 1; break;
    case GameState::Turn:    street_idx = 2; break;
    case GameState::River:   street_idx = 3; break;
    default: break;
    }

    // Equity calculation - multithreaded for better performance
    constexpr int equity_sims = 500;
    float equity = static_cast<float>(probabilityOfWinning(hero.hole, g.board(), equity_sims, ps.size() - 1, pool));

    // Betting indicators
    float facing_bet = (to_call > 0) ? 1.f : 0.f;
    float can_raise = (hero.chips > to_call) ? 1.f : 0.f;
    float bet_to_pot = (bd.pot > 0) ? static_cast<float>(bd.currentBet) / static_cast<float>(bd.pot) : 0.f;
    float committed_ratio = static_cast<float>(hero.committed) / static_cast<float>(std::max(1u, hero.chips + hero.committed));

    // Position indicator (rough approximation)
    float position = static_cast<float>(heroIdx) / static_cast<float>(std::max<size_t>(1, ps.size() - 1));

    // Build feature vector (32 features)
    dlib::matrix<float> x(kInputDims, 1);
    x = 0;
    int k = 0;

    // Street one-hot (4 features)
    x(k++) = (street_idx == 0) ? 1.f : 0.f;  // PreFlop
    x(k++) = (street_idx == 1) ? 1.f : 0.f;  // Flop
    x(k++) = (street_idx == 2) ? 1.f : 0.f;  // Turn
    x(k++) = (street_idx == 3) ? 1.f : 0.f;  // River

    // Betting situation (6 features)
    x(k++) = (to_call == 0) ? 1.f : 0.f;                                // can_check
    x(k++) = norm_bb(static_cast<float>(to_call), kMaxToCallBB);        // to_call_bb
    x(k++) = norm_bb(static_cast<float>(bd.pot),  kMaxStackBB);         // pot_bb
    x(k++) = pot_odds;                                                   // pot odds [0,1]
    x(k++) = facing_bet;                                                 // facing a bet
    x(k++) = norm(bet_to_pot, kMaxBetToPot);                            // bet/pot ratio

    // Stack info (6 features)
    x(k++) = norm_bb(static_cast<float>(hero.chips), kMaxStackBB);      // hero stack
    x(k++) = norm(spr, kMaxSPR);                                        // SPR
    x(k++) = committed_ratio;                                            // street commitment
    x(k++) = norm_bb(avg_opp_stack, kMaxStackBB);                       // avg opp stack
    x(k++) = norm_bb(effective_stack, kMaxStackBB);                     // effective stack
    x(k++) = can_raise;                                                  // can raise

    // Game state (3 features — street_progress removed; one-hot already encodes street)
    x(k++) = static_cast<float>(alive) / static_cast<float>(ps.size()); // fraction alive
    x(k++) = static_cast<float>(elig) / static_cast<float>(ps.size());  // fraction eligible
    x(k++) = position;                                                    // position indicator

    // Hand strength: only raw equity (1 feature)
    // Transforms like equity² or √equity are monotone functions the network can learn itself.
    // Discrete bins are just noisy discretizations of a value the network already has.
    x(k++) = equity; // raw Monte Carlo win probability

    // Decision features (5 features — scaled EV margin removed as it duplicates EV margin × 5)
    x(k++) = (equity > pot_odds) ? 1.f : 0.f;               // +EV to call
    x(k++) = std::max(0.f, equity - pot_odds);               // EV margin
    x(k++) = (equity > 0.5f && spr < 4.f) ? 1.f : 0.f;     // short-stacked value spot
    x(k++) = (equity < 0.3f && facing_bet > 0.5f) ? 1.f : 0.f; // fold indicator
    x(k++) = (equity > 0.7f && can_raise > 0.5f) ? 1.f : 0.f;  // raise indicator

    // Independent structural features (7 features — replace the 7 removed slots above)
    // These capture information NOT derivable from equity alone.
    const float pot_ownership = (bd.pot > 0)
        ? std::min(static_cast<float>(hero.invested) / static_cast<float>(bd.pot), 1.f)
        : 0.f;
    const float min_opp_stack_norm = norm_bb(min_opp_chips, kMaxStackBB);
    const float min_raise_fraction = std::min(
        static_cast<float>(bd.minRaise) / static_cast<float>(std::max(1u, hero.chips)), 1.f);
    const float min_raise_to_pot = std::min(
        static_cast<float>(bd.minRaise) / static_cast<float>(std::max(1u, bd.pot)) / kMaxRaiseToPot, 1.f);
    const float opp_stack_ratio = std::min(
        min_opp_chips / std::max(1.f, static_cast<float>(hero.chips)) / 2.f, 1.f);
    const float hero_stack_invested = static_cast<float>(hero.chips)
        / static_cast<float>(std::max(1u, hero.chips + hero.invested));
    const float facing_allin =
        (static_cast<float>(bd.currentBet) >= static_cast<float>(hero.committed) + static_cast<float>(hero.chips))
        ? 1.f : 0.f;

    x(k++) = pot_ownership;      // hero's share of total pot (commitment signal)
    x(k++) = min_opp_stack_norm; // shortest opponent stack (side-pot / shove pressure)
    x(k++) = min_raise_fraction; // min raise cost as fraction of hero stack
    x(k++) = min_raise_to_pot;   // min raise cost relative to pot
    x(k++) = opp_stack_ratio;    // shortest opponent stack relative to hero
    x(k++) = hero_stack_invested; // remaining stack / total invested (hand commitment)
    x(k++) = facing_allin;       // 1 if a call would put hero all-in

    // Ensure we have exactly kInputDims (32)
    while (k < kInputDims)
        x(k++) = 0.f;

    return x;
}

#endif // __POKER_RL_FEATURIZER_HPP__