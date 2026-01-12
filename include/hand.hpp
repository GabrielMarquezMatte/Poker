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
    static constexpr std::array<std::uint64_t, 13> rankMasks = []()
    {
        std::array<std::uint64_t, 13> m{};
        for (std::size_t r = 0; r < 13; ++r)
        {
            m[r] = (1ULL << (r + 0 * 13)) | (1ULL << (r + 1 * 13)) | (1ULL << (r + 2 * 13)) | (1ULL << (r + 3 * 13));
        }
        return m;
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
    static constexpr std::array<std::uint8_t, 16> popCountTable = []()
    {
        std::array<std::uint8_t, 16> table{};
        for (std::uint32_t i = 0; i < table.size(); ++i)
        {
            table[i] = static_cast<std::uint8_t>(std::popcount(i));
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
    static inline constexpr std::pair<std::uint8_t, std::uint8_t> topTwoCounts(std::uint64_t handMask) noexcept
    {
        std::uint8_t maxC = 0, secondC = 0;
        for (int r = 0; r < 13; ++r)
        {
            std::uint64_t nib = pext(handMask, rankMasks[r]);
            std::uint8_t c = popCountTable[nib];
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
        const std::uint16_t s0 = static_cast<std::uint16_t>(deckMask & 0x1FFFu);
        const std::uint16_t s1 = static_cast<std::uint16_t>((deckMask >> 13) & 0x1FFFu);
        const std::uint16_t s2 = static_cast<std::uint16_t>((deckMask >> 26) & 0x1FFFu);
        const std::uint16_t s3 = static_cast<std::uint16_t>((deckMask >> 39) & 0x1FFFu);
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
    static inline constexpr uint16_t makeTwoPairMask(std::uint64_t deckMask) noexcept
    {
        constexpr std::uint64_t RANK_MASK = (1u << 13) - 1;
        const std::uint16_t a = std::uint16_t((deckMask)&RANK_MASK);
        const std::uint16_t b = std::uint16_t((deckMask >> 13) & RANK_MASK);
        const std::uint16_t c = std::uint16_t((deckMask >> 26) & RANK_MASK);
        const std::uint16_t d = std::uint16_t((deckMask >> 39) & RANK_MASK);
        const std::uint16_t any = std::uint16_t(a | b | c | d);
        const std::uint16_t pairs = std::uint16_t((a & b) | (c & d) | ((a ^ b) & (c ^ d)));
        const std::uint16_t pairBits = top2Table[pairs];
        const std::uint16_t kickerBit = hiTable[std::uint16_t(any & ~pairBits)];
        return std::uint16_t(pairBits | kickerBit);
    }

public:
    static inline constexpr ClassificationResult classify(const Deck cards) noexcept
    {
        std::uint64_t deckMask = cards.getMask();
        auto [flush, rankValue] = getFlush(deckMask);
        auto [straight, highRank] = getStraight(rankValue);
        if (straight && flush)
        {
            if (highRank == Rank::Ace)
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
            std::uint16_t twoPairMask = makeTwoPairMask(deckMask);
            return {Classification::TwoPair, static_cast<Rank>(twoPairMask)};
        }
        return {Classification::Pair, rankValue};
    }
};
#endif // __POKER_HAND_HPP__