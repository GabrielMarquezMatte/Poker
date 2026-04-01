#ifndef __POKER_RL_ACTIONS_HPP__
#define __POKER_RL_ACTIONS_HPP__
#include <array>
#include <span>
#include <cstdint>
#include <algorithm>
#include "../game/game.hpp"

// Discrete action indices used throughout the RL pipeline
enum ActIdx : unsigned
{
    A_Fold       = 0,
    A_CheckCall  = 1,
    A_BetHalfPot = 2,
    A_BetPot     = 3,
    A_AllIn      = 4,
    A_COUNT      = 5
};

// Compute the total-bet target for a sized bet/raise given how much to add above the current pot.
static inline std::uint32_t sized_bet_target(const BetData &bd, std::uint32_t add) noexcept
{
    return (bd.currentBet == 0)
        ? std::max<std::uint32_t>(bd.minRaise, add)
        : std::max<std::uint32_t>(bd.currentBet + bd.minRaise, bd.currentBet + add);
}

// Fixed-size result of legal_actions() — avoids heap allocation in the hot path.
struct LegalActions {
    std::array<unsigned, A_COUNT> data;
    unsigned count = 0;

    void push_back(unsigned v) noexcept { data[count++] = v; }
    bool empty() const noexcept { return count == 0; }
    unsigned size() const noexcept { return count; }
    unsigned operator[](std::size_t i) const noexcept { return data[i]; }
    const unsigned* begin() const noexcept { return data.data(); }
    const unsigned* end()   const noexcept { return data.data() + count; }
    operator std::span<const unsigned>() const noexcept { return {data.data(), count}; }
};

// Returns the set of legal abstract actions for `heroIdx` in the current game state.
inline LegalActions legal_actions(const Game &g, std::size_t heroIdx, const Blinds & /*blinds*/)
{
    LegalActions actions;
    const auto &p  = g.players()[heroIdx];
    const auto &bd = g.betData();

    if (!p.alive())
        return actions;

    const std::uint32_t to_call    = (bd.currentBet > p.committed) ? (bd.currentBet - p.committed) : 0u;
    const std::uint32_t stack_total = p.committed + p.chips;
    const std::uint32_t pot         = std::max<std::uint32_t>(1, bd.pot);

    // Fold is only meaningful when there is a live bet to face
    if (to_call > 0)
        actions.push_back(A_Fold);

    // Check/Call is always available (to_engine_action handles the short-call → all-in case)
    actions.push_back(A_CheckCall);

    // Append an action only if it is not already in the list
    auto add_if_new = [&](unsigned idx)
    {
        if (std::find(actions.begin(), actions.end(), idx) == actions.end())
            actions.push_back(idx);
    };

    // A bet/raise target is "non-shove" when it is strictly between committed and stack
    auto is_non_shove_target = [&](std::uint32_t target) -> bool
    {
        return target > p.committed && target < stack_total;
    };

    // Sized bets: only offered when they land at a distinct non-all-in chip level
    const std::uint32_t half_target = sized_bet_target(bd, pot / 2);
    const std::uint32_t pot_target  = sized_bet_target(bd, pot);

    if (is_non_shove_target(half_target)) add_if_new(A_BetHalfPot);
    if (is_non_shove_target(pot_target))  add_if_new(A_BetPot);

    // All-in is surfaced when shoving is a natural option:
    //   low_spr       — stack ≤ pot+call, pot odds justify committing
    //   pot_committed — calling would leave too little to raise meaningfully
    //   short_stack   — fewer than 2 min-raises remain in stack
    const std::uint32_t remaining_after_call = p.chips - std::min(p.chips, to_call);
    const bool low_spr       = p.chips <= (bd.pot + to_call);
    const bool pot_committed = (to_call > 0) &&
                               (remaining_after_call <= std::max<std::uint32_t>(bd.minRaise, bd.pot / 2));
    const bool short_stack   = p.chips <= 2u * std::max<std::uint32_t>(1u, bd.minRaise);

    if (low_spr || pot_committed || short_stack)
        add_if_new(A_AllIn);

    return actions;
}

// Translates an abstract action index into the concrete ActionStruct expected by the game engine.
inline ActionStruct to_engine_action(unsigned idx, const Game &g, std::size_t heroIdx, const Blinds & /*blinds*/)
{
    const auto &p  = g.players()[heroIdx];
    const auto &bd = g.betData();

    // Returns the additional chips needed to reach `target`, capped at the player's stack
    auto chips_to_reach = [&](std::uint32_t target) -> std::uint32_t
    {
        if (target <= p.committed) return 0;
        const std::uint32_t need = target - p.committed;
        return (need >= p.chips) ? p.chips : need;
    };

    switch (idx)
    {
    case A_Fold:
        return ActionStruct{ActionType::Fold, 0};

    case A_CheckCall:
    {
        const std::uint32_t to_call = (bd.currentBet > p.committed) ? (bd.currentBet - p.committed) : 0u;
        if (to_call == 0)        return ActionStruct{ActionType::Check, 0};
        if (to_call >= p.chips)  return ActionStruct{ActionType::AllIn, 0};
        return ActionStruct{ActionType::Call, 0};
    }

    case A_BetHalfPot:
    case A_BetPot:
    {
        // Target respects the minRaise rule; if no current bet it's a Bet, otherwise a Raise
        const std::uint32_t pot = std::max<std::uint32_t>(1, bd.pot);
        const std::uint32_t add    = (idx == A_BetHalfPot) ? (pot / 2) : pot;
        const std::uint32_t target = sized_bet_target(bd, add);

        const std::uint32_t need = chips_to_reach(target);
        if (need == p.chips)
            return ActionStruct{ActionType::AllIn, 0};
        return (bd.currentBet == 0)
            ? ActionStruct{ActionType::Bet,   target}
            : ActionStruct{ActionType::Raise, target};
    }

    case A_AllIn:
    default:
        return ActionStruct{ActionType::AllIn, 0};
    }
}

#endif // __POKER_RL_ACTIONS_HPP__
