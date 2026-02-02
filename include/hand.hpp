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
        Rank highCard;
    };
    static constexpr std::array<StraightInfo, 1 << 13> straightTable = []()
    {
        std::array<StraightInfo, 1 << 13> tbl{};

        constexpr std::uint32_t lowStraight = static_cast<std::uint32_t>(Rank::LowStraight);

        for (std::uint32_t m = 0; m < tbl.size(); ++m)
        {
            std::uint32_t run5 = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4);
            if (run5)
            {
                int high = std::bit_width(run5) - 1;
                tbl[m] = {true, static_cast<Rank>(Rank::Two << (high + 4))};
                continue;
            }
            if ((m & lowStraight) == lowStraight)
            {
                tbl[m] = {true, Rank::Five};
                continue;
            }
            tbl[m] = {false, Rank::Two};
        }
        return tbl;
    }();
    static constexpr std::array<uint16_t, 1 << 13> flushTable = []()
    {
        std::array<uint16_t, 1 << 13> table{};
        for (std::uint32_t m = 0; m < (1 << 13); ++m)
        {
            table[m] = (std::popcount(m) >= 5 ? uint16_t(m) : 0);
        }
        return table;
    }();
    static constexpr std::array<std::uint16_t, 8192> hiTable = []()
    {
        std::array<std::uint16_t, 8192> table{};
        for (std::uint16_t v = 0; v < 8192; ++v)
        {
            std::uint16_t m = 0;
            for (int r = 12; r >= 0; --r)
            {
                if (v & (std::uint16_t(1u) << r))
                {
                    m = std::uint16_t(1u) << r;
                    break;
                }
            }
            table[v] = m;
        }
        return table;
    }();
    static constexpr std::array<std::uint16_t, 8192> top2Table = []()
    {
        std::array<std::uint16_t, 8192> t{};
        for (std::uint16_t v = 0; v < 8192; ++v)
        {
            std::uint16_t m = 0;
            int count = 0;
            for (int r = 12; r >= 0; --r)
            {
                if (!(v & (std::uint16_t(1u) << r)))
                {
                    continue;
                }
                m |= std::uint16_t(1u) << r;
                if (++count == 2)
                {
                    break;
                }
            }
            t[v] = m;
        }
        return t;
    }();
    static constexpr std::array<std::uint16_t, 8192> top3Table = []()
    {
        std::array<std::uint16_t, 8192> t{};
        for (std::uint16_t v = 0; v < 8192; ++v)
        {
            std::uint16_t m = 0;
            int count = 0;
            for (int r = 12; r >= 0; --r)
            {
                if (!(v & (std::uint16_t(1u) << r)))
                {
                    continue;
                }
                m |= std::uint16_t(1u) << r;
                if (++count == 3)
                {
                    break;
                }
            }
            t[v] = m;
        }
        return t;
    }();
    static inline constexpr std::pair<std::uint8_t, std::uint8_t> topTwoCounts(std::uint16_t s0, std::uint16_t s1, std::uint16_t s2, std::uint16_t s3) noexcept
    {
        const std::uint16_t all4 = s0 & s1 & s2 & s3;
        if (all4) [[unlikely]]
        {
            const std::uint16_t without4 = ~all4 & 0x1FFFu;
            const std::uint16_t s0w = s0 & without4;
            const std::uint16_t s1w = s1 & without4;
            const std::uint16_t s2w = s2 & without4;
            const std::uint16_t s3w = s3 & without4;
            const std::uint16_t three = (s0w & s1w & s2w) | (s0w & s1w & s3w) | (s0w & s2w & s3w) | (s1w & s2w & s3w);
            if (three) [[unlikely]] return {4, 3};
            const std::uint16_t two = (s0w & s1w) | (s0w & s2w) | (s0w & s3w) | (s1w & s2w) | (s1w & s3w) | (s2w & s3w);
            if (two) [[unlikely]] return {4, 2};
            if ((s0w | s1w | s2w | s3w) != 0) return {4, 1};
            return {4, 0};
        }
        const std::uint16_t three = (s0 & s1 & s2) | (s0 & s1 & s3) | (s0 & s2 & s3) | (s1 & s2 & s3);
        if (three) [[unlikely]]
        {
            const std::uint16_t without3 = ~three & 0x1FFFu;
            const std::uint16_t s0w = s0 & without3;
            const std::uint16_t s1w = s1 & without3;
            const std::uint16_t s2w = s2 & without3;
            const std::uint16_t s3w = s3 & without3;
            const std::uint16_t three2 = (s0w & s1w & s2w) | (s0w & s1w & s3w) | (s0w & s2w & s3w) | (s1w & s2w & s3w);
            if (three2) [[unlikely]] return {3, 3};
            const std::uint16_t two = (s0w & s1w) | (s0w & s2w) | (s0w & s3w) | (s1w & s2w) | (s1w & s3w) | (s2w & s3w);
            if (two) [[unlikely]] return {3, 2};
            return {3, 1};
        }
        const std::uint16_t two = (s0 & s1) | (s0 & s2) | (s0 & s3) | (s1 & s2) | (s1 & s3) | (s2 & s3);
        if (two)
        {
            const int pairCount = std::popcount(two);
            if (pairCount >= 2) return {2, 2};
            return {2, 1};
        }
        const std::uint16_t any = s0 | s1 | s2 | s3;
        return {1, static_cast<std::uint8_t>(std::popcount(any) > 1 ? 1 : 0)};
    }
    static inline constexpr std::tuple<bool, Rank> getFlush(std::uint16_t s0, std::uint16_t s1, std::uint16_t s2, std::uint16_t s3) noexcept
    {
        const std::uint16_t f0 = flushTable[s0];
        const std::uint16_t f1 = flushTable[s1];
        const std::uint16_t f2 = flushTable[s2];
        const std::uint16_t f3 = flushTable[s3];
        const std::uint16_t flushMask = f0 | f1 | f2 | f3;
        const bool isFlush = flushMask != 0;
        const std::uint16_t rankMask = isFlush ? flushMask : (s0 | s1 | s2 | s3);
        return {isFlush, static_cast<Rank>(rankMask)};
    }
    static inline constexpr StraightInfo getStraight(const Rank rankMask) noexcept
    {
        return straightTable[static_cast<std::uint16_t>(rankMask)];
    }
    static inline constexpr uint16_t makeTwoPairMask(std::uint16_t s0, std::uint16_t s1, std::uint16_t s2, std::uint16_t s3) noexcept
    {
        const std::uint16_t any = s0 | s1 | s2 | s3;
        const std::uint16_t pairs = (s0 & s1) | (s2 & s3) | ((s0 ^ s1) & (s2 ^ s3));
        const std::uint16_t pairBits = top2Table[pairs];
        const std::uint16_t kickerBit = hiTable[any & ~pairBits];
        return std::uint16_t(pairBits | kickerBit);
    }

    static inline constexpr uint16_t makePairMask(std::uint16_t s0, std::uint16_t s1, std::uint16_t s2, std::uint16_t s3) noexcept
    {
        const std::uint16_t any = s0 | s1 | s2 | s3;
        const std::uint16_t pairs = (s0 & s1) | (s2 & s3) | ((s0 ^ s1) & (s2 ^ s3));
        const std::uint16_t pairBit = hiTable[pairs];
        const std::uint16_t kickerRanks = any & ~pairBit;
        const std::uint16_t top3Kickers = top3Table[kickerRanks];
        // Encode pair rank index in high bits (multiply by 512), kickers in low bits
        // This ensures pair rank is compared first, then kickers
        const int pairRankIndex = std::countr_zero(static_cast<std::uint32_t>(pairBit));
        // Shift kickers to fit in 9 bits (bits 4-12 → 0-8)
        const std::uint16_t kickerValue = top3Kickers >> 4;
        return static_cast<std::uint16_t>(pairRankIndex << 9) | kickerValue;
    }

    static inline constexpr std::tuple<std::uint16_t, std::uint16_t, std::uint16_t, std::uint16_t> getSuitRanks(std::uint64_t deckMask) noexcept
    {
        constexpr std::uint64_t RANK_MASK = (1u << 13) - 1;
        const std::uint16_t s0 = static_cast<std::uint16_t>(deckMask & RANK_MASK);
        const std::uint16_t s1 = static_cast<std::uint16_t>((deckMask >> 13) & RANK_MASK);
        const std::uint16_t s2 = static_cast<std::uint16_t>((deckMask >> 26) & RANK_MASK);
        const std::uint16_t s3 = static_cast<std::uint16_t>((deckMask >> 39) & RANK_MASK);
        return {s0, s1, s2, s3};
    }

public:
    static inline constexpr ClassificationResult classify(const Deck cards) noexcept
    {
        std::uint64_t deckMask = cards.getMask();
        auto [s0, s1, s2, s3] = getSuitRanks(deckMask);
        auto [flush, rankValue] = getFlush(s0, s1, s2, s3);
        auto [straight, highRank] = getStraight(rankValue);
        if (straight && flush)
        {
            if (highRank == Rank::Ace)
            {
                return {Classification::RoyalFlush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten};
            }
            return {Classification::StraightFlush, highRank};
        }
        auto [maxCount, secondMaxCount] = topTwoCounts(s0, s1, s2, s3);
        if (maxCount == 4)
        {
            return {Classification::FourOfAKind, rankValue};
        }
        if (maxCount == 3 && secondMaxCount == 2)
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
            std::uint16_t twoPairMask = makeTwoPairMask(s0, s1, s2, s3);
            return {Classification::TwoPair, static_cast<Rank>(twoPairMask)};
        }
        std::uint16_t pairMask = makePairMask(s0, s1, s2, s3);
        return {Classification::Pair, static_cast<Rank>(pairMask)};
    }
};
#endif // __POKER_HAND_HPP__