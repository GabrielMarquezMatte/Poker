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

    // Fold always legal (even vs zero bet, albeit silly)
    a.push_back(A_Fold);

    // Check or Call
    if (bd.currentBet > p.committed)
    {
        if (p.chips + p.committed >= bd.currentBet)
            a.push_back(A_CheckCall);
        else
            a.push_back(A_AllIn); // can't call fully => shove is the only way to continue
    }
    else
    {
        a.push_back(A_CheckCall); // as Check
    }

    // Bet/Raise (discretized)
    auto add_bet = [&](unsigned idx)
    { if (std::find(a.begin(), a.end(), idx)==a.end()) a.push_back(idx); };
    add_bet(A_BetHalfPot);
    add_bet(A_BetPot);
    add_bet(A_AllIn);

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