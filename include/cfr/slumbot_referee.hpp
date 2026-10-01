#ifndef __POKER_CFR_SLUMBOT_REFEREE_HPP__
#define __POKER_CFR_SLUMBOT_REFEREE_HPP__
#include "../hand.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

// Slumbot's game run locally (rules from its sample_api.py): tracks a hand's action string, rejects
// illegal increments and scores the hand. Chips are Slumbot's (blinds 50/100, stacks 20,000). pos 1 is
// the small blind (first preflop), 0 the big blind (first postflop).
struct SlumbotReferee
{
    std::string action;
    int street = 0, pos = 1;
    std::uint32_t streetBetTo = 100, totalBetTo = 100, lastBetSize = 50;
    std::array<std::uint32_t, 2> streetBet{100, 50}; // by pos
    std::uint32_t streetStart = 0;                   // each player's chips in before this street
    bool endsStreet = false, done = false;
    int folder = -1, allInStreet = -1; // pos that folded; street where an all-in was called

    bool apply(const std::string &incr)
    {
        const char c = incr.empty() ? '?' : incr[0];
        bool slash = false;
        if (c == 'f')
        {
            if (lastBetSize == 0)
            {
                return false;
            }
            folder = pos;
            done = true;
        }
        else if (c == 'k' || c == 'c')
        {
            if ((c == 'k') != (lastBetSize == 0))
            {
                return false;
            }
            streetBet[pos] = streetBetTo;
            if (c == 'c' && totalBetTo == 20000)
            {
                allInStreet = street; // the board runs out
                done = true;
            }
            else if (endsStreet)
            {
                done = street == 3;
                slash = !done;
                street += done ? 0 : 1;
                pos = 0;
                streetStart += done ? 0 : streetBetTo;
                streetBet = {done ? streetBet[0] : 0, done ? streetBet[1] : 0};
                streetBetTo = done ? streetBetTo : 0;
                endsStreet = false;
            }
            else
            {
                pos = 1 - pos;
                endsStreet = true;
            }
            lastBetSize = 0;
        }
        else if (c == 'b')
        {
            const std::uint32_t to = static_cast<std::uint32_t>(std::stoul(incr.substr(1)));
            if (to <= streetBetTo)
            {
                return false;
            }
            const std::uint32_t size = to - streetBetTo, remaining = 20000 - totalBetTo;
            const std::uint32_t minimum = std::min(std::max<std::uint32_t>(lastBetSize, 100), remaining);
            if (size < minimum || size > remaining)
            {
                return false;
            }
            lastBetSize = size;
            streetBetTo = to;
            streetBet[pos] = to;
            totalBetTo += size;
            pos = 1 - pos;
            endsStreet = true;
        }
        else
        {
            return false;
        }
        action += incr;
        action += slash ? "/" : "";
        return true;
    }

    inline std::uint32_t committed(int p) const noexcept { return streetStart + streetBet[static_cast<std::size_t>(p)]; }

    // Chips won by pos `p` once done, given both hands (by pos) and the five board cards. With
    // `allInEquity`, a hand that went all-in before the river scores its equity in the pot instead of
    // the runout (both hands being known, that removes the luck of the board).
    double winnings(int p, const std::array<std::uint64_t, 2> &hands, std::uint64_t board, const std::vector<std::uint64_t> &order,
                    bool allInEquity) const
    {
        const int q = 1 - p;
        if (folder >= 0)
        {
            return folder == p ? -static_cast<double>(committed(p)) : static_cast<double>(committed(q));
        }
        const double stake = committed(p); // equal at showdown
        if (allInEquity && allInStreet >= 0 && allInStreet < 3)
        {
            std::uint64_t known = 0;
            for (int i = 0; i < std::array<int, 3>{0, 3, 4}[static_cast<std::size_t>(allInStreet)]; ++i)
            {
                known |= order[static_cast<std::size_t>(i)];
            }
            return (2.0 * equity(hands[static_cast<std::size_t>(p)], hands[static_cast<std::size_t>(q)], known) - 1.0) * stake;
        }
        const auto mine = Hand::classify(Deck::from_mask(hands[static_cast<std::size_t>(p)] | board));
        const auto theirs = Hand::classify(Deck::from_mask(hands[static_cast<std::size_t>(q)] | board));
        return mine > theirs ? stake : (mine < theirs ? -stake : 0.0);
    }

    // P(win) + P(tie) / 2 of `mine` against `theirs` over every completion of `board`.
    static double equity(std::uint64_t mine, std::uint64_t theirs, std::uint64_t board)
    {
        std::vector<std::uint64_t> live;
        for (std::uint64_t c = ((1ull << 52) - 1) & ~(mine | theirs | board); c != 0; c &= c - 1)
        {
            live.push_back(c & (~c + 1));
        }
        double won = 0.0, total = 0.0;
        const auto run = [&](auto &self, std::size_t from, int missing, std::uint64_t b) -> void
        {
            if (missing == 0)
            {
                const auto a = Hand::classify(Deck::from_mask(mine | b));
                const auto o = Hand::classify(Deck::from_mask(theirs | b));
                won += a > o ? 1.0 : (a == o ? 0.5 : 0.0);
                total += 1.0;
                return;
            }
            for (std::size_t i = from; i + static_cast<std::size_t>(missing) <= live.size(); ++i)
            {
                self(self, i + 1, missing - 1, b | live[i]);
            }
        };
        run(run, 0, 5 - std::popcount(board), board);
        return won / total;
    }
};
#endif // __POKER_CFR_SLUMBOT_REFEREE_HPP__
