#ifndef __POKER_POT_MANAGER_HPP__
#define __POKER_POT_MANAGER_HPP__
#include <vector>
#include <cstddef>
#include <algorithm>
#include <span>
#include "player.hpp"
struct SidePot
{
    std::uint32_t amount = 0;
    std::vector<std::size_t> eligiblePlayers;
};
struct PotManager
{
    static constexpr std::vector<SidePot> build(std::span<const Player> players)
    {
        const std::size_t n = players.size();
        if (n == 0) return {};
        
        std::vector<std::uint32_t> levels;
        levels.reserve(n);
        for (const auto& p : players)
        {
            if (p.invested > 0)
            {
                levels.push_back(p.invested);
            }
        }
        if (levels.empty()) return {};
        
        std::sort(levels.begin(), levels.end());
        levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
        
        std::vector<SidePot> pots;
        pots.reserve(levels.size());
        
        std::uint32_t prevCap = 0;
        for (std::uint32_t cap : levels)
        {
            const std::uint32_t delta = cap - prevCap;
            SidePot pot;
            pot.eligiblePlayers.reserve(n);
            
            for (std::size_t i = 0; i < n; ++i)
            {
                if (players[i].invested > prevCap)
                {
                    pot.amount += delta;
                }
                if (players[i].alive() && players[i].invested >= cap)
                {
                    pot.eligiblePlayers.push_back(i);
                }
            }
            
            if (pot.amount > 0 && !pot.eligiblePlayers.empty())
            {
                pots.push_back(std::move(pot));
            }
            prevCap = cap;
        }
        return pots;
    }
};
#endif // __POKER_POT_MANAGER_HPP__