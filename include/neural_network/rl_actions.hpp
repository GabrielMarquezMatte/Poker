#ifndef __POKER_RL_ACTIONS_HPP__
#define __POKER_RL_ACTIONS_HPP__
#include <vector>
#include <cstdint>
#include <algorithm>
#include "../game/game.hpp"

// Discrete action indices:
enum ActIdx : unsigned
{
    A_Fold = 0,
    A_CheckCall = 1,
    A_BetHalfPot = 2,
    A_BetPot = 3,
    A_AllIn = 4,
    A_COUNT = 5
};

inline std::vector<unsigned> legal_actions(const Game &g, std::size_t heroIdx, const Blinds & /*blinds*/)
{
    std::vector<unsigned> a;
    auto const &p = g.players()[heroIdx];
    auto const &bd = g.betData();
    if (!p.alive())
        return a;

    const std::uint32_t to_call =
        (bd.currentBet > p.committed) ? (bd.currentBet - p.committed) : 0;
    const std::uint32_t stack_total = p.committed + p.chips;
    const std::uint32_t pot = std::max<std::uint32_t>(1, bd.pot);

    // Only allow fold when actually facing a bet.
    if (to_call > 0)
        a.push_back(A_Fold);

    // Check or Call (to_engine_action maps short-call to all-in automatically).
    a.push_back(A_CheckCall);

    auto add_unique = [&](unsigned idx)
    {
        if (std::find(a.begin(), a.end(), idx) == a.end())
            a.push_back(idx);
    };
    auto can_target_without_shoving = [&](std::uint32_t target) -> bool
    {
        return target > p.committed && target < stack_total;
    };

    // Add size actions only if they represent a distinct non-all-in target.
    const std::uint32_t half_target = (bd.currentBet == 0)
        ? std::max<std::uint32_t>(bd.minRaise, pot / 2)
        : std::max<std::uint32_t>(bd.currentBet + bd.minRaise, bd.currentBet + pot / 2);
    const std::uint32_t pot_target = (bd.currentBet == 0)
        ? std::max<std::uint32_t>(bd.minRaise, pot)
        : std::max<std::uint32_t>(bd.currentBet + bd.minRaise, bd.currentBet + pot);
    if (can_target_without_shoving(half_target))
        add_unique(A_BetHalfPot);
    if (can_target_without_shoving(pot_target))
        add_unique(A_BetPot);

    // Keep explicit shove for short-stack / low-SPR / pot-committed spots.
    const bool low_spr_spot = p.chips <= (bd.pot + to_call);
    const std::uint32_t remaining_after_call = p.chips - std::min(p.chips, to_call);
    const bool pot_committed =
        (to_call > 0) &&
        (remaining_after_call <= std::max<std::uint32_t>(bd.minRaise, bd.pot / 2));
    const bool short_stack = p.chips <= 2u * std::max<std::uint32_t>(1, bd.minRaise);
    if (low_spr_spot || pot_committed || short_stack)
        add_unique(A_AllIn);

    return a;
}

inline ActionStruct to_engine_action(unsigned idx, const Game &g, std::size_t heroIdx, const Blinds & /*blinds*/)
{
    auto const &p = g.players()[heroIdx];
    auto const &bd = g.betData();

    // Helper to clamp spend
    auto clamp_add = [&](std::uint32_t target) -> std::uint32_t
    {
        if (target <= p.committed)
            return 0;
        std::uint32_t need = target - p.committed;
        if (need >= p.chips)
            return p.chips; // will be AllIn add
        return need;
    };

    switch (idx)
    {
    case A_Fold:
        return ActionStruct{ActionType::Fold, 0};
    case A_CheckCall:
    {
        std::uint32_t to_call = (bd.currentBet > p.committed) ? (bd.currentBet - p.committed) : 0;
        if (to_call == 0)
            return ActionStruct{ActionType::Check, 0};
        if (to_call >= p.chips)
            return ActionStruct{ActionType::AllIn, 0};
        return ActionStruct{ActionType::Call, 0};
    }
    case A_BetHalfPot:
    case A_BetPot:
    {
        // Compute target as currentBet + size, respecting minRaise rules.
        // If no live bet, it's a Bet; else a Raise.
        std::uint32_t pot = std::max<std::uint32_t>(1, bd.pot);
        std::uint32_t add = (idx == A_BetHalfPot) ? (pot / 2) : pot;
        std::uint32_t target = (bd.currentBet == 0)
                                   ? std::max<std::uint32_t>(bd.minRaise, add)
                                   : std::max<std::uint32_t>(bd.currentBet + bd.minRaise, bd.currentBet + add);

        std::uint32_t need = clamp_add(target);
        if (need == p.chips)
            return ActionStruct{ActionType::AllIn, 0};
        return (bd.currentBet == 0)
                   ? ActionStruct{ActionType::Bet, target}
                   : ActionStruct{ActionType::Raise, target};
    }
    case A_AllIn:
    default:
        return ActionStruct{ActionType::AllIn, 0};
    }
}

#endif // __POKER_RL_ACTIONS_HPP__
