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
    static inline constexpr std::array<int, 13> processCards(const Deck cards)
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
    static inline constexpr std::pair<bool, Rank> getStraight(const Rank rankMask)
    {
        // Special Ace-low case: 2,3,4,5,A
        if (rankMask == Rank::LowStraight)
        {
            return {true, Rank::Five};
        }
        std::uint32_t m = static_cast<std::uint32_t>(rankMask);
        std::uint32_t run5 = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4);
        if (run5 != 0)
        {
            int highCard = std::bit_width(run5) - 1;
            return {true, static_cast<Rank>(Rank::Two << (highCard + 4))};
        }
        int highCard = std::bit_width(m) - 1;
        return {false, static_cast<Rank>(Rank::Two << highCard)};
    }

public:
    static inline constexpr ClassificationResult classify(const Deck cards)
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