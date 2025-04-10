#ifndef __POKER_HAND_HPP__
#define __POKER_HAND_HPP__
#include <array>
#include <span>
#include <algorithm>
#include "card.hpp"
struct Hand
{
private:
    static inline constexpr Rank getRanks(const std::span<const Card, 5> cards)
    {
        Rank ranks = Rank::Two;
        for (const Card card : cards)
        {
            ranks = ranks | card.getRank();
        }
        return ranks;
    }
    static inline constexpr std::pair<bool, Rank> getStraight(const std::span<const Card, 5> cards)
    {
        // Special Ace-low case: 2,3,4,5,A
        Rank cardRanks = getRanks(cards);
        if (cardRanks == Rank::LowStraight)
        {
            return {true, Rank::Five};
        }
        for (int i = 0; i <= 8; ++i)
        {
            // Cria uma máscara com 5 bits consecutivos iniciando em i: (0x1F = 0b11111)
            uint32_t straightSequence = 0x1Fu << i;
            if ((static_cast<uint32_t>(cardRanks) & straightSequence) == straightSequence)
            {
                // O rank mais alto da sequência será i+4.
                return {true, static_cast<Rank>(1 << (i + 4))};
            }
        }
        int highestBit = std::bit_width(static_cast<uint32_t>(cardRanks)) - 1;
        return {false, static_cast<Rank>(1 << highestBit)};
    }
public:
    static inline constexpr ClassificationResult classify(const std::span<const Card, 5> cards)
    {
        bool flush = (cards[0].getSuit() == cards[1].getSuit() &&
                      cards[1].getSuit() == cards[2].getSuit() &&
                      cards[2].getSuit() == cards[3].getSuit() &&
                      cards[3].getSuit() == cards[4].getSuit());
        // Check for a straight.
        auto [straight, straightHigh] = getStraight(cards);
        if (flush && straight)
        {
            // For a royal flush the high card is Ace.
            if (straightHigh == Rank::Ace)
            {
                return {Classification::RoyalFlush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten};
            }
            return {Classification::StraightFlush, straightHigh};
        }
        // Build a frequency table for ranks.
        std::array<std::pair<Rank, int>, 13> rankCounts;
        for (const auto &card : cards)
        {
            std::size_t index = getRankIndex(card.getRank());
            int count = rankCounts[index].second;
            if (count == 0)
            {
                rankCounts[index] = {card.getRank(), 1};
                continue;
            }
            rankCounts[index].second = count + 1;
        }
        // Sort by count (descending), then rank (descending).
        std::sort(rankCounts.begin(), rankCounts.end(), [](const std::pair<Rank, int> &a, const std::pair<Rank, int> &b)
                  {
                    if (a.second != b.second)
                    {
                        return a.second > b.second;
                    }
                    return static_cast<uint32_t>(a.first) > static_cast<uint32_t>(b.first); });
        if (rankCounts[0].second == 4)
        {
            return {Classification::FourOfAKind, getRanks(cards)};
        }
        if (rankCounts[0].second == 3 && rankCounts[1].second >= 2)
        {
            return {Classification::FullHouse, getRanks(cards)};
        }
        if (flush)
        {
            return {Classification::Flush, getRanks(cards)};
        }
        if (straight)
        {
            return {Classification::Straight, straightHigh};
        }
        if (rankCounts[0].second == 3)
        {
            return {Classification::ThreeOfAKind, getRanks(cards)};
        }
        if (rankCounts[0].second != 2)
        {
            return {Classification::HighCard, getRanks(cards)};
        }
        if (rankCounts[1].second == 2)
        {
            return {Classification::TwoPair, getRanks(cards)};
        }
        return {Classification::Pair, getRanks(cards)};
    }
};
#endif // __POKER_HAND_HPP__