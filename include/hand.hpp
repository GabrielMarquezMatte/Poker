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
    struct SuitMasks
    {
        std::uint16_t s0;
        std::uint16_t s1;
        std::uint16_t s2;
        std::uint16_t s3;
        inline constexpr std::uint16_t anySuit() const noexcept
        {
            return s0 | s1 | s2 | s3;
        }
    };
    struct CountInfo
    {
        std::uint8_t maxCount;
        std::uint8_t secondMaxCount;
        std::uint16_t pairs;
    };
    static inline constexpr CountInfo topTwoCounts(SuitMasks suits, std::uint16_t anySuit) noexcept
    {
        const std::uint16_t all4 = suits.s0 & suits.s1 & suits.s2 & suits.s3;
        if (all4) [[unlikely]]
        {
            const std::uint16_t without4 = ~all4 & 0x1FFFu;
            const std::uint16_t s0w = suits.s0 & without4;
            const std::uint16_t s1w = suits.s1 & without4;
            const std::uint16_t s2w = suits.s2 & without4;
            const std::uint16_t s3w = suits.s3 & without4;
            const std::uint16_t three = (s0w & s1w & s2w) | (s0w & s1w & s3w) | (s0w & s2w & s3w) | (s1w & s2w & s3w);
            if (three) [[unlikely]] return {4, 3, 0};
            const std::uint16_t two = (s0w & s1w) | (s2w & s3w) | ((s0w ^ s1w) & (s2w ^ s3w));
            if (two) [[unlikely]] return {4, 2, 0};
            if ((s0w | s1w | s2w | s3w) != 0) return {4, 1, 0};
            return {4, 0, 0};
        }
        const std::uint16_t three = (suits.s0 & suits.s1 & suits.s2) | (suits.s0 & suits.s1 & suits.s3) | (suits.s0 & suits.s2 & suits.s3) | (suits.s1 & suits.s2 & suits.s3);
        if (three) [[unlikely]]
        {
            const std::uint16_t without3 = ~three & 0x1FFFu;
            const std::uint16_t s0w = suits.s0 & without3;
            const std::uint16_t s1w = suits.s1 & without3;
            const std::uint16_t s2w = suits.s2 & without3;
            const std::uint16_t s3w = suits.s3 & without3;
            const std::uint16_t three2 = (s0w & s1w & s2w) | (s0w & s1w & s3w) | (s0w & s2w & s3w) | (s1w & s2w & s3w);
            if (three2) [[unlikely]] return {3, 3, 0};
            const std::uint16_t two = (s0w & s1w) | (s2w & s3w) | ((s0w ^ s1w) & (s2w ^ s3w));
            if (two) [[unlikely]] return {3, 2, 0};
            return {3, 1, 0};
        }
        const std::uint16_t two = (suits.s0 & suits.s1) | (suits.s2 & suits.s3) | ((suits.s0 ^ suits.s1) & (suits.s2 ^ suits.s3));
        if (two)
        {
            const int pairCount = std::popcount(two);
            if (pairCount >= 2) return {2, 2, two};
            return {2, 1, two};
        }
        return {1, static_cast<std::uint8_t>(std::popcount(anySuit) > 1 ? 1 : 0), 0};
    }
    static inline constexpr std::tuple<bool, Rank> getFlush(SuitMasks suits, std::uint16_t anySuit) noexcept
    {
        const std::uint16_t f0 = flushTable[suits.s0];
        const std::uint16_t f1 = flushTable[suits.s1];
        const std::uint16_t f2 = flushTable[suits.s2];
        const std::uint16_t f3 = flushTable[suits.s3];
        const std::uint16_t flushMask = f0 | f1 | f2 | f3;
        const bool isFlush = flushMask != 0;
        const std::uint16_t rankMask = isFlush ? flushMask : anySuit;
        return {isFlush, static_cast<Rank>(rankMask)};
    }
    static inline constexpr StraightInfo getStraight(const Rank rankMask) noexcept
    {
        return straightTable[static_cast<std::uint16_t>(rankMask)];
    }
    static inline constexpr uint16_t makeTwoPairMask(std::uint16_t anySuit, std::uint16_t pairs) noexcept
    {
        const std::uint16_t pairBits = top2Table[pairs];
        const std::uint16_t kickerBit = hiTable[anySuit & ~pairBits];
        return pairBits | kickerBit;
    }

    static inline constexpr uint16_t makePairMask(std::uint16_t anySuit, std::uint16_t pairs) noexcept
    {
        const std::uint16_t pairBit = hiTable[pairs];
        const std::uint16_t kickerRanks = anySuit & ~pairBit;
        const std::uint16_t top3Kickers = top3Table[kickerRanks];
        const int pairRankIndex = std::countr_zero(static_cast<std::uint32_t>(pairBit));
        const std::uint16_t kickerValue = top3Kickers >> 4;
        return static_cast<std::uint16_t>(pairRankIndex << 9) | kickerValue;
    }

    static inline constexpr SuitMasks getSuitRanks(std::uint64_t deckMask) noexcept
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
        SuitMasks suits = getSuitRanks(deckMask);
        const std::uint16_t anySuit = suits.anySuit();
        auto [flush, rankValue] = getFlush(suits, anySuit);
        auto [straight, highRank] = getStraight(rankValue);
        if (straight && flush) [[unlikely]]
        {
            if (highRank == Rank::Ace)
            {
                return {Classification::RoyalFlush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten};
            }
            return {Classification::StraightFlush, highRank};
        }
        auto [maxCount, secondMaxCount, pairs] = topTwoCounts(suits, anySuit);
        if (maxCount == 4) [[unlikely]]
        {
            return {Classification::FourOfAKind, rankValue};
        }
        if (maxCount == 3 && secondMaxCount == 2) [[unlikely]]
        {
            return {Classification::FullHouse, rankValue};
        }
        if (flush) [[unlikely]]
        {
            return {Classification::Flush, rankValue};
        }
        if (straight) [[unlikely]]
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
            std::uint16_t twoPairMask = makeTwoPairMask(anySuit, pairs);
            return {Classification::TwoPair, static_cast<Rank>(twoPairMask)};
        }
        std::uint16_t pairMask = makePairMask(anySuit, pairs);
        return {Classification::Pair, static_cast<Rank>(pairMask)};
    }
};
#endif // __POKER_HAND_HPP__