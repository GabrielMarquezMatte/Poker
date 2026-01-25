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
    static constexpr std::vector<SidePot> build(const std::span<const Player> players)
    {
        const std::size_t n = players.size();
        std::vector<std::uint32_t> contrib(n, 0);
        std::vector<bool> isActive(n, false);
        for (std::size_t i = 0; i < n; ++i)
        {
            contrib[i] = std::max(0u, players[i].invested);
            isActive[i] = players[i].alive() && !players[i].folded;
        }
        std::uint32_t prevCap = 0;
        std::vector<SidePot> pots;
        pots.reserve(n);
        while (true)
        {
            std::uint32_t cap = 0;
            for (std::uint32_t c : contrib)
            {
                if (c > prevCap)
                {
                    cap = (cap == 0) ? c : std::min(cap, c);
                }
            }
            if (cap == 0)
            {
                break;
            }
            const std::uint32_t capDelta = cap - prevCap;
            SidePot pot{};
            for (std::size_t i = 0; i < n; ++i)
            {
                if (contrib[i] > prevCap)
                {
                    pot.amount += capDelta;
                }
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                if (isActive[i] && contrib[i] >= cap)
                {
                    pot.eligiblePlayers.push_back(i);
                }
            }
            pots.push_back(std::move(pot));
            prevCap = cap;
        }
        return pots;
    }
};
#endif // __POKER_POT_MANAGER_HPP__