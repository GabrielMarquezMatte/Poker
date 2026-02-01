#ifndef __POKER_RL_FEATURIZER_HPP__
#define __POKER_RL_FEATURIZER_HPP__
#include <dlib/matrix.h>
#include <span>
#include <cmath>
#include <BS_thread_pool.hpp>
#include "../game/game.hpp"
#include "../game.hpp"

// Enhanced featurizer with 32 features for better learning
// Uses thread pool for parallel equity calculation
inline dlib::matrix<float> featurize(const Game &g, std::size_t heroIdx, const Blinds &blinds, BS::thread_pool<BS::tp::none> &pool)
{
    const auto bb = std::max<std::uint32_t>(1, blinds.bigBlind);
    const float bb_f = static_cast<float>(bb);
    auto const &ps = g.players();
    const auto &hero = ps[heroIdx];
    const auto &bd = g.betData();

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
    float spr = (bd.pot > 0) ? static_cast<float>(hero.chips) / static_cast<float>(bd.pot) : 20.f;
    spr = std::min(spr, 20.f);

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
    constexpr int equity_sims = 5000;
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
    x(k++) = (to_call == 0) ? 1.f : 0.f;                            // can_check
    x(k++) = std::min(static_cast<float>(to_call) / bb_f, 100.f) / 100.f;  // to_call_bb normalized
    x(k++) = std::min(static_cast<float>(bd.pot) / bb_f, 200.f) / 200.f;   // pot_bb normalized
    x(k++) = pot_odds;                                               // pot odds [0, 1]
    x(k++) = facing_bet;                                             // is facing a bet
    x(k++) = std::min(bet_to_pot, 3.f) / 3.f;                       // bet relative to pot normalized

    // Stack info (6 features)
    x(k++) = std::min(static_cast<float>(hero.chips) / bb_f, 200.f) / 200.f;    // hero_stack_bb normalized
    x(k++) = std::min(spr, 20.f) / 20.f;                                         // normalized SPR
    x(k++) = committed_ratio;                                                     // commitment level
    x(k++) = std::min(avg_opp_stack / bb_f, 200.f) / 200.f;                      // avg opp stack normalized
    x(k++) = std::min(effective_stack / bb_f, 200.f) / 200.f;                    // effective stack normalized
    x(k++) = can_raise;                                                           // can we raise?

    // Game state (4 features)
    x(k++) = static_cast<float>(alive) / static_cast<float>(ps.size());   // fraction alive
    x(k++) = static_cast<float>(elig) / static_cast<float>(ps.size());    // fraction eligible
    x(k++) = position;                                                      // position indicator
    x(k++) = static_cast<float>(street_idx) / 3.f;                         // street progress

    // HAND STRENGTH - Most important features (6 features)
    x(k++) = equity;                                                        // raw equity
    x(k++) = equity * equity;                                               // squared (emphasize strong)
    x(k++) = std::sqrt(equity);                                            // sqrt (emphasize weak differences)
    x(k++) = (equity > 0.65f) ? 1.f : 0.f;                                 // strong hand
    x(k++) = (equity > 0.35f && equity <= 0.65f) ? 1.f : 0.f;             // medium hand
    x(k++) = (equity <= 0.35f) ? 1.f : 0.f;                                // weak hand

    // Decision-relevant derived features (6 features)
    x(k++) = (equity > pot_odds) ? 1.f : 0.f;                              // +EV to call
    x(k++) = std::max(0.f, equity - pot_odds);                             // EV margin
    x(k++) = std::min(1.f, std::max(0.f, (equity - pot_odds) * 5.f));     // scaled EV margin
    x(k++) = (equity > 0.5f && spr < 4.f) ? 1.f : 0.f;                    // short-stacked value
    x(k++) = (equity < 0.3f && facing_bet > 0.5f) ? 1.f : 0.f;            // should fold indicator
    x(k++) = (equity > 0.7f && can_raise > 0.5f) ? 1.f : 0.f;             // should raise indicator

    // Ensure we have exactly kInputDims (32)
    while (k < kInputDims)
        x(k++) = 0.f;

    return x;
}

#endif // __POKER_RL_FEATURIZER_HPP__