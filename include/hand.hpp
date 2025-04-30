#ifndef __POKER_HAND_HPP__
#define __POKER_HAND_HPP__
#include <array>
#include <span>
#include <algorithm>
#include "card.hpp"
#include "deck.hpp"
#include "classification_result.hpp"
struct Hand
{
private:
    struct StraightInfo
    {
        bool isStraight;
        Rank highCard; // valid only if isStraight==true
    };
    static constexpr std::array<StraightInfo, 1 << 13> straightTable = []()
    {
        std::array<StraightInfo, 1 << 13> tbl{}; // 13 ranks ⇒ 2^13 possible masks

        constexpr std::uint32_t LOW_STRAIGHT = static_cast<std::uint32_t>(Rank::LowStraight);

        for (std::uint32_t m = 0; m < tbl.size(); ++m)
        {
            if (m == LOW_STRAIGHT)
            {
                tbl[m] = {true, Rank::Five};
                continue;
            }
            std::uint32_t run5 = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4);
            if (!run5)
            {
                tbl[m] = {false, Rank::Two}; // highCard unused
                continue;
            }
            // highest set bit in run5 is the straight’s top card
            int high = std::bit_width(run5) - 1;
            tbl[m] = {true, static_cast<Rank>(Rank::Two << (high + 4))};
        }
        return tbl;
    }();
    static constexpr std::array<std::uint64_t, 13> rankMasks = []()
    {
        std::array<std::uint64_t, 13> m{};
        for (std::size_t r = 0; r < 13; ++r)
        {
            // suit offsets: 0,13,26,39
            m[r] = (1ULL << (r + 0 * 13)) | (1ULL << (r + 1 * 13)) | (1ULL << (r + 2 * 13)) | (1ULL << (r + 3 * 13));
        }
        return m;
    }();
    static inline constexpr std::uint64_t pext(std::uint64_t x, std::uint64_t mask) noexcept
    {
        if (!std::is_constant_evaluated())
        {
            return _pext_u64(x, mask);
        }
        std::uint64_t result = 0;
        std::uint64_t outBit = 1;
        while (mask)
        {
            std::uint64_t lowest = mask & static_cast<std::int64_t>(-static_cast<std::int64_t>(mask));
            if (x & lowest)
            {
                result |= outBit;
            }
            mask &= mask - 1;
            outBit <<= 1;
        }
        return result;
    }
    static inline constexpr std::pair<int, int> topTwoCounts(std::uint64_t handMask) noexcept
    {
        int maxC = 0, secondC = 0;
        for (int r = 0; r < 13; ++r)
        {
            // extract the 4 suit-bits for rank r into the low 4 bits
            std::uint64_t nib = pext(handMask, rankMasks[r]);
            // hardware popcount of that nibble is your count
            int c = std::popcount(nib);
            if (c > maxC)
            {
                secondC = maxC;
                maxC = c;
            }
            else if (c > secondC)
            {
                secondC = c;
            }
        }
        return {maxC, secondC};
    }
    static inline constexpr std::tuple<bool, Rank> getFlush(std::uint64_t deckMask) noexcept
    {
        std::uint32_t s0 = static_cast<std::uint32_t>(deckMask & 0x1FFFu);
        std::uint32_t s1 = static_cast<std::uint32_t>((deckMask >> 13) & 0x1FFFu);
        std::uint32_t s2 = static_cast<std::uint32_t>((deckMask >> 26) & 0x1FFFu);
        std::uint32_t s3 = static_cast<std::uint32_t>((deckMask >> 39) & 0x1FFFu);
        std::uint32_t flushMask = s0 | s1 | s2 | s3;
        bool flush = (std::popcount(s0) >= 5) || (std::popcount(s1) >= 5) || (std::popcount(s2) >= 5) || (std::popcount(s3) >= 5);
        return {flush, static_cast<Rank>(flushMask)};
    }
    static inline constexpr StraightInfo getStraight(const Rank rankMask) noexcept
    {
        return straightTable[static_cast<std::uint32_t>(rankMask)];
    }

public:
    static inline constexpr ClassificationResult classify(const Deck cards) noexcept
    {
        std::uint64_t deckMask = cards.getMask();
        auto [flush, rankValue] = getFlush(deckMask);
        auto [straight, highRank] = getStraight(rankValue);
        if (straight && flush)
        {
            if (rankValue == Rank::HighStraight)
            {
                return {Classification::RoyalFlush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten};
            }
            return {Classification::StraightFlush, highRank};
        }
        auto [maxCount, secondMaxCount] = topTwoCounts(deckMask);
        if (maxCount == 4)
        {
            return {Classification::FourOfAKind, rankValue};
        }
        if (maxCount == 3 && secondMaxCount >= 2)
        {
            return {Classification::FullHouse, rankValue};
        }
        if (flush)
        {
            return {Classification::Flush, rankValue};
        }
        if (straight)
        {
            return {Classification::Straight, highRank};
        }
        if (maxCount == 3)
        {
            return {Classification::ThreeOfAKind, rankValue};
        }
        if (maxCount != 2)
        {
            return {Classification::HighCard, rankValue};
        }
        if (secondMaxCount == 2)
        {
            return {Classification::TwoPair, rankValue};
        }
        return {Classification::Pair, rankValue};
    }
};
#endif // __POKER_HAND_HPP__