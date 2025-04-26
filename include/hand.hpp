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
            if (run5)
            {
                // highest set bit in run5 is the straight’s top card
                int high = std::bit_width(run5) - 1;
                tbl[m] = {true, static_cast<Rank>(Rank::Two << (high + 4))};
                continue;
            }
            tbl[m] = {false, Rank::Two}; // highCard unused
        }
        return tbl;
    }();
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
    static inline constexpr std::array<int, 13> processCards(const Deck cards) noexcept
    {
        std::array<int, 13> counts{}; // Inicializa com 0
        // Assume que o primeiro naipe seja o de referência para verificar flush.
        for (const Card card : cards)
        {
            Rank cardRank = card.getRank();
            // Atualiza contagem. Supondo que getRankIndex retorne um índice de 0 a 12.
            counts[getRankIndex(cardRank)]++;
        }
        return counts;
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
        std::array<int, 13> counts = processCards(cards);
        int maxCount = 0;
        int secondMaxCount = 0;
        for (int count : counts)
        {
            if (count > maxCount)
            {
                secondMaxCount = maxCount;
                maxCount = count;
                continue;
            }
            if (count > secondMaxCount)
            {
                secondMaxCount = count;
            }
        }
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