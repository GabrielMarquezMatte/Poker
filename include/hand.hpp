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
        {
            table[m] = (std::popcount(m) >= 5 ? uint16_t(m) : 0);
        }
        return table;
    }();
    static inline constexpr std::uint16_t getFlush(SuitMasks suits) noexcept
    {
        return flushTable[suits.s0] | flushTable[suits.s1] | flushTable[suits.s2] | flushTable[suits.s3];
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
        const SuitMasks suits = getSuitRanks(cards.getMask());
        const std::uint16_t anySuit = suits.anySuit();
        const std::uint16_t flushMask = getFlush(suits);
        const std::uint16_t straightVal = straightTable[anySuit];

        // Rank masks by multiplicity: quads (4 suits), three (>= 3), two (>= 2).
        const std::uint16_t p01 = suits.s0 & suits.s1;
        const std::uint16_t p23 = suits.s2 & suits.s3;
        const std::uint16_t quads = p01 & p23;
        const std::uint16_t three = (p01 & (suits.s2 | suits.s3)) | (p23 & (suits.s0 | suits.s1));
        const std::uint16_t two = p01 | p23 | ((suits.s0 ^ suits.s1) & (suits.s2 ^ suits.s3));
        const std::uint16_t pairsOnly = two & ~three;
        const bool fullHouse = (three != 0) & ((static_cast<std::uint16_t>(three & (three - 1u)) | pairsOnly) != 0);

        // Rare categories: branches are almost never taken, so they predict well.
        if (flushMask) [[unlikely]]
        {
            const std::uint16_t straightValFlush = straightTable[flushMask];
            if (straightValFlush) [[unlikely]]
            {
                if (straightValFlush == static_cast<std::uint16_t>(Rank::Ace))
                {
                    return {Classification::RoyalFlush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten};
                }
                return {Classification::StraightFlush, straightValFlush};
            }
        }
        if (quads) [[unlikely]]
        {
            return {Classification::FourOfAKind, quads, highBit(anySuit & ~quads)};
        }
        if (fullHouse) [[unlikely]]
        {
            const std::uint16_t trips = highBit(three);
            return {Classification::FullHouse, trips, highBit((three & ~trips) | pairsOnly)};
        }
        if (flushMask) [[unlikely]]
        {
            return {Classification::Flush, keepTopBits(flushMask, 5)};
        }
        if (straightVal) [[unlikely]]
        {
            return {Classification::Straight, straightVal};
        }

        // HighCard / Pair / TwoPair / ThreeOfAKind (~90% of hands) share one branch-free path:
        // which of them it is changes from hand to hand, so branching on it mispredicts a lot.
        // Here `three` has at most one bit and pairsOnly is empty when it does.
        const std::uint16_t lowestPair = pairsOnly & (~pairsOnly + 1u);
        const std::uint16_t topPairs = pairsOnly ^ (std::popcount(pairsOnly) > 2 ? lowestPair : 0);
        const bool isTrips = three != 0;
        const std::uint16_t primary = isTrips ? three : topPairs;
        const int pairCount = std::popcount(topPairs);
        const int madeCards = isTrips ? 3 : 2 * pairCount;
        const std::uint32_t category = isTrips ? 3u : static_cast<std::uint32_t>(pairCount);
        return ClassificationResult::fromIndex(category, primary, keepTopBits(anySuit & ~primary, 5 - madeCards));
    }
};
#endif // __POKER_HAND_HPP__
