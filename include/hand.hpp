#ifndef __POKER_HAND_HPP__
#define __POKER_HAND_HPP__
#include <array>
#include <span>
#include <algorithm>
#include "card.hpp"
struct Hand
{
private:
    std::array<Card, 5> m_cards;
    inline constexpr Rank getRanks() const
    {
        Rank ranks = Rank::Two;
        for (const Card card : m_cards)
        {
            ranks = ranks | card.getRank();
        }
        return ranks;
    }
    inline constexpr std::pair<bool, Rank> getStraight() const
    {
        // Special Ace-low case: 2,3,4,5,A
        Rank cardRanks = getRanks();
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
        return {false, m_cards[4].getRank()};
    }
public:
    inline constexpr Hand() = default;
    // Construct a hand from 5 cards and sort them in ascending order
    inline constexpr Hand(Card c1, Card c2, Card c3, Card c4, Card c5): m_cards{c1, c2, c3, c4, c5}
    {
        std::sort(m_cards.begin(), m_cards.end(), [](const Card &a, const Card &b)
                  { return static_cast<int>(a.getRank()) < static_cast<int>(b.getRank()); });
    }
    inline constexpr Hand(const std::array<Card, 5> &cards)
        : m_cards(cards)
    {
        std::sort(m_cards.begin(), m_cards.end(), [](const Card &a, const Card &b)
                  { return static_cast<int>(a.getRank()) < static_cast<int>(b.getRank()); });
    }
    inline constexpr Hand(const std::span<const Card> cards)
    {
        if (cards.size() != 5)
        {
            throw std::invalid_argument("Hand must have exactly 5 cards.");
        }
        std::copy(cards.begin(), cards.end(), m_cards.begin());
        std::sort(m_cards.begin(), m_cards.end(), [](const Card &a, const Card &b)
                  { return static_cast<int>(a.getRank()) < static_cast<int>(b.getRank()); });
    }
    inline constexpr std::span<const Card, 5> cards() const
    {
        return m_cards;
    }
    inline constexpr ClassificationResult classify() const
    {
        bool flush = (m_cards[0].getSuit() == m_cards[1].getSuit() &&
                      m_cards[1].getSuit() == m_cards[2].getSuit() &&
                      m_cards[2].getSuit() == m_cards[3].getSuit() &&
                      m_cards[3].getSuit() == m_cards[4].getSuit());
        // Check for a straight.
        auto [straight, straightHigh] = getStraight();
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
        for (const auto &card : m_cards)
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
            return {Classification::FourOfAKind, getRanks()};
        }
        if (rankCounts[0].second == 3 && rankCounts[1].second >= 2)
        {
            return {Classification::FullHouse, getRanks()};
        }
        if (flush)
        {
            return {Classification::Flush, getRanks()};
        }
        if (straight)
        {
            return {Classification::Straight, straightHigh};
        }
        if (rankCounts[0].second == 3)
        {
            return {Classification::ThreeOfAKind, getRanks()};
        }
        if (rankCounts[0].second != 2)
        {
            return {Classification::HighCard, getRanks()};
        }
        if (rankCounts[1].second == 2)
        {
            return {Classification::TwoPair, getRanks()};
        }
        return {Classification::Pair, getRanks()};
    }
};
#endif // __POKER_HAND_HPP__