#ifndef __POKER_HAND_HPP__
#define __POKER_HAND_HPP__
#include <array>
#include <span>
#include <algorithm>
#include "card.hpp"
#include "intrinsics.hpp"
#include "deck.hpp"
#include "classification_result.hpp"
struct Hand
{
private:
    // Returns the highest set bit of x as a bitmask. x must be nonzero.
    static inline constexpr std::uint16_t highBit(std::uint16_t x) noexcept
    {
        return std::uint16_t(1) << (std::bit_width(x) - 1);
    }
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
        std::uint16_t pairs;     // pair rank bits (Pair/TwoPair), or pair rank for FullHouse
        std::uint16_t majorRank; // trips rank bit (maxCount==3) or quads rank bit (maxCount==4)
    };
    static inline constexpr CountInfo topTwoCounts(SuitMasks suits) noexcept
    {
        const std::uint16_t p01 = suits.s0 & suits.s1;
        const std::uint16_t p23 = suits.s2 & suits.s3;
        const std::uint16_t all4 = p01 & p23;
        if (all4) [[unlikely]]
        {
            const std::uint16_t without4 = ~all4 & 0x1FFFu;
            const std::uint16_t s0w = suits.s0 & without4;
            const std::uint16_t s1w = suits.s1 & without4;
            const std::uint16_t s2w = suits.s2 & without4;
            const std::uint16_t s3w = suits.s3 & without4;
            const std::uint16_t p01w = s0w & s1w;
            const std::uint16_t p23w = s2w & s3w;
            const std::uint16_t three = (p01w & (s2w | s3w)) | (p23w & (s0w | s1w));
            if (three) [[unlikely]]
                return {4, 3, 0, all4};
            const std::uint16_t two = p01w | p23w | ((s0w ^ s1w) & (s2w ^ s3w));
            if (two) [[unlikely]]
                return {4, 2, 0, all4};
            if ((s0w | s1w | s2w | s3w) != 0)
                return {4, 1, 0, all4};
            return {4, 0, 0, all4};
        }
        const std::uint16_t three = (p01 & (suits.s2 | suits.s3)) | (p23 & (suits.s0 | suits.s1));
        if (three) [[unlikely]]
        {
            if (three & (three - 1u)) [[unlikely]]
            {
                const std::uint16_t tripsRank = highBit(three);
                const std::uint16_t pairRank = static_cast<std::uint16_t>(three & ~tripsRank);
                return {3, 2, pairRank, tripsRank};
            }
            const std::uint16_t without3 = ~three & 0x1FFFu;
            const std::uint16_t s0w = suits.s0 & without3;
            const std::uint16_t s1w = suits.s1 & without3;
            const std::uint16_t s2w = suits.s2 & without3;
            const std::uint16_t s3w = suits.s3 & without3;
            const std::uint16_t two = (s0w & s1w) | (s2w & s3w) | ((s0w ^ s1w) & (s2w ^ s3w));
            if (two) [[unlikely]]
                return {3, 2, two, three};
            return {3, 1, 0, three};
        }
        const std::uint16_t two = p01 | p23 | ((suits.s0 ^ suits.s1) & (suits.s2 ^ suits.s3));
        if (two)
        {
            if (two & (two - 1u))
                return {2, 2, two, 0};
            return {2, 1, two, 0};
        }
        return {1, 0, 0, 0};
    }
    static constexpr std::array<std::uint16_t, 1 << 13> straightTable = []()
    {
        std::array<std::uint16_t, 1 << 13> tbl{};
        constexpr std::uint32_t lowStraight = static_cast<std::uint32_t>(Rank::LowStraight);
        for (std::uint32_t m = 0; m < tbl.size(); ++m)
        {
            std::uint32_t run5 = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4);
            if (run5)
            {
                int high = std::bit_width(run5) - 1;
                tbl[m] = static_cast<std::uint16_t>(Rank::Two << (high + 4));
                continue;
            }
            if ((m & lowStraight) == lowStraight)
            {
                tbl[m] = static_cast<std::uint16_t>(Rank::Five);
                continue;
            }
            tbl[m] = 0;
        }
        return tbl;
    }();
    static constexpr std::array<uint16_t, 1 << 13> flushTable = []()
    {
        std::array<uint16_t, 1 << 13> table{};
        for (std::uint32_t m = 0; m < (1 << 13); ++m)
            table[m] = (std::popcount(m) >= 5 ? uint16_t(m) : 0);
        return table;
    }();
    static inline constexpr std::uint16_t getFlush(SuitMasks suits) noexcept
    {
        return flushTable[suits.s0] | flushTable[suits.s1] | flushTable[suits.s2] | flushTable[suits.s3];
    }
    static inline constexpr std::uint16_t makeTwoPairMask(std::uint16_t anySuit, std::uint16_t pairs) noexcept
    {
        const std::uint16_t pairBits = keepTopBits(pairs, 2);
        const std::uint16_t kickerBit = highBit(anySuit & ~pairBits);
        return pairBits | kickerBit;
    }

    static inline constexpr std::uint16_t makePairMask(std::uint16_t anySuit, std::uint16_t pairs) noexcept
    {
        std::uint16_t k = anySuit ^ pairs;
        const int excess = std::popcount(k) - 3;
        if (excess > 0)
            k &= k - 1u;
        if (excess > 1)
            k &= k - 1u;
        const int pairRankIndex = std::countr_zero(static_cast<std::uint32_t>(pairs));
        return static_cast<std::uint16_t>(pairRankIndex << 9) | (k >> 4);
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
        const std::uint16_t flushMask = getFlush(suits);
        const std::uint16_t straightVal = straightTable[anySuit];
        auto [maxCount, secondMaxCount, pairs, majorRank] = topTwoCounts(suits);
        if (flushMask) [[unlikely]]
        {
            const std::uint16_t straightValFlush = straightTable[flushMask];
            if (straightValFlush) [[unlikely]]
            {
                if (straightValFlush == static_cast<std::uint16_t>(Rank::Ace))
                    return {Classification::RoyalFlush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten};
                return {Classification::StraightFlush, static_cast<Rank>(straightValFlush)};
            }
            if (maxCount == 4) [[unlikely]]
            {
                const std::uint16_t kickerBit = highBit(static_cast<std::uint16_t>(anySuit) & ~majorRank);
                return {Classification::FourOfAKind, static_cast<Rank>(majorRank | kickerBit)};
            }
            if (maxCount == 3 && secondMaxCount == 2) [[unlikely]]
            {
                const int tripsIdx = std::countr_zero(static_cast<std::uint32_t>(majorRank));
                const int pairIdx = std::bit_width(static_cast<std::uint32_t>(pairs)) - 1;
                return {Classification::FullHouse, static_cast<Rank>((tripsIdx << 4) | pairIdx)};
            }
            return {Classification::Flush, static_cast<Rank>(keepTopBits(flushMask, 5))};
        }
        if (maxCount == 4) [[unlikely]]
        {
            const std::uint16_t kickerBit = highBit(static_cast<std::uint16_t>(anySuit) & ~majorRank);
            return {Classification::FourOfAKind, static_cast<Rank>(majorRank | kickerBit)};
        }
        if (maxCount == 3 && secondMaxCount == 2) [[unlikely]]
        {
            const int tripsIdx = std::countr_zero(static_cast<std::uint32_t>(majorRank));
            const int pairIdx = std::bit_width(static_cast<std::uint32_t>(pairs)) - 1;
            return {Classification::FullHouse, static_cast<Rank>((tripsIdx << 4) | pairIdx)};
        }
        if (straightVal) [[unlikely]]
        {
            return {Classification::Straight, static_cast<Rank>(straightVal)};
        }
        if (maxCount == 3)
        {
            std::uint16_t k = static_cast<std::uint16_t>(anySuit) & ~majorRank;
            const int excess = std::popcount(k) - 2;
            if (excess > 0)
                k &= k - 1u;
            if (excess > 1)
                k &= k - 1u;
            return {Classification::ThreeOfAKind, static_cast<Rank>(majorRank | k)};
        }
        if (maxCount != 2)
        {
            std::uint16_t hc = static_cast<std::uint16_t>(anySuit);
            const int excess = std::popcount(hc) - 5;
            if (excess > 0)
                hc &= hc - 1u;
            if (excess > 1)
                hc &= hc - 1u;
            return {Classification::HighCard, static_cast<Rank>(hc)};
        }
        if (secondMaxCount == 2)
        {
            return {Classification::TwoPair, static_cast<Rank>(makeTwoPairMask(anySuit, pairs))};
        }
        return {Classification::Pair, static_cast<Rank>(makePairMask(anySuit, pairs))};
    }
};
#endif // __POKER_HAND_HPP__