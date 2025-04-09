#ifndef __POKER_GAME_HPP__
#define __POKER_GAME_HPP__
#include "classification_result.hpp"
#include "hand.hpp"
struct Player
{
private:
    std::array<Card, 2> m_cards;
    std::array<Card, 5> m_table;
public:
    inline constexpr Player() = default;
    inline constexpr Player(Card c1, Card c2, const std::span<const Card> table)
        : m_cards{c1, c2}
    {
        if (table.size() < 5)
        {
            throw std::invalid_argument("Table must have at least 5 cards.");
        }
        std::copy(table.begin(), table.begin() + 5, m_table.begin());
    }
    inline constexpr Player(Card c1, Card c2, const std::array<Card, 5> &table)
        : m_cards{c1, c2}, m_table(table)
    {
    }
    inline constexpr std::span<const Card, 2> cards() const
    {
        return m_cards;
    }
    inline constexpr std::span<const Card, 5> table() const
    {
        return m_table;
    }
    inline constexpr ClassificationResult classify() const
    {
        std::array<Card, 7> cards;
        std::copy(m_table.begin(), m_table.end(), cards.begin());
        std::copy(m_cards.begin(), m_cards.end(), cards.begin() + 5);
        std::sort(cards.begin(), cards.end(), [](const Card &a, const Card &b)
                  { return static_cast<int>(a.getRank()) < static_cast<int>(b.getRank()); });
        ClassificationResult best = {Classification::HighCard, Rank::Two};
        for (std::size_t i = 0; i < 7; ++i)
        {
            for (std::size_t j = i + 1; j < 7; ++j)
            {
                std::array<Card, 5> hand;
                std::size_t k = 0;
                for (std::size_t l = 0; l < 7; ++l)
                {
                    if (l != i && l != j)
                    {
                        hand[k++] = cards[l];
                    }
                }
                ClassificationResult candidate = Hand(hand).classify();
                if (candidate > best)
                {
                    best = candidate;
                }
            }
        }
        return best;
    }
    inline constexpr bool operator==(const Player &other) const
    {
        return m_cards == other.m_cards && m_table == other.m_table;
    }
    inline constexpr bool operator!=(const Player &other) const
    {
        return !(*this == other);
    }
};
template <std::size_t N>
struct Game
{
private:
    static_assert(N > 1, "Number of players must be greater than 1.");
    static_assert(N <= 10, "Number of players must be less than or equal to 10.");
    static_assert(N >= 2, "Number of players must be at least 2.");
private:
    std::array<Player, N> m_players;
    std::array<Card, 5> m_table;
public:
    constexpr Game(const std::array<Player, N> &players, const std::array<Card, 5> &table)
        : m_players(players), m_table(table)
    {
    }
    constexpr Game(const std::array<Player, N> &players, const std::span<const Card> table) : m_players(players)
    {
        if (table.size() < 5)
        {
            throw std::invalid_argument("Table must have at least 5 cards.");
        }
        std::copy(table.begin(), table.begin() + 5, m_table.begin());
    }
    inline constexpr std::array<std::pair<Player, ClassificationResult>, N> classify() const
    {
        std::array<std::pair<Player, ClassificationResult>, N> results;
        for (int i = 0; i < N; ++i)
        {
            results[i] = {m_players[i], m_players[i].classify()};
        }
        std::sort(results.begin(), results.end(), [](const std::pair<Player, ClassificationResult> &a, const std::pair<Player, ClassificationResult> &b)
                  { return a.second > b.second; });
        return results;
    }
};
#endif // __POKER_GAME_HPP__