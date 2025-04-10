#ifndef __POKER_DECK_HPP__
#define __POKER_DECK_HPP__
#include <array>
#include <span>
#include <pcg_random.hpp>
#include "card.hpp"
struct Deck
{
private:
    std::array<Card, 52> m_cards;
    std::size_t m_deckSize = 52;
    inline void createAndShuffleDeck(pcg64 &rng)
    {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 13; ++j)
            {
                m_cards[i * 13 + j] = {static_cast<Suit>(1 << i), static_cast<Rank>(1 << j)};
            }
        }
        std::shuffle(m_cards.begin(), m_cards.end(), rng);
    }
public:
    Deck(pcg64 &rng)
    {
        createAndShuffleDeck(rng);
    }
    Deck(Deck &&) = default;
    Deck &operator=(Deck &&) = default;
    Deck(const Deck &) = delete;
    Deck &operator=(const Deck &) = delete;
    inline constexpr std::span<const Card> cards() const
    {
        return std::span<const Card>(m_cards.data(), m_deckSize);
    }
    // Deal the next 2 cards (without reshuffling).
    inline std::optional<std::span<const Card, 2>> dealToPlayer()
    {
        if (m_deckSize < 2)
        {
            return std::nullopt;
        }
        m_deckSize -= 2;
        return std::span<const Card, 2>{m_cards.data() + (52 - m_deckSize - 2), 2};
    }
    // Deal the next 5 cards (without reshuffling).
    inline std::optional<std::span<const Card, 5>> dealToTable()
    {
        if (m_deckSize < 5)
        {
            return std::nullopt;
        }
        m_deckSize -= 5;
        return std::span<const Card, 5>{m_cards.data() + (52 - m_deckSize - 5), 5};
    }
    template <std::size_t N>
    inline std::optional<std::span<const Card, N>> dealCards()
    {
        if (m_deckSize < N)
        {
            return std::nullopt;
        }
        m_deckSize -= N;
        return std::span<const Card, N>{m_cards.data() + (52 - m_deckSize - N), N};
    }
    inline std::optional<std::span<const Card>> dealCards(std::size_t numCards)
    {
        if (m_deckSize < numCards)
        {
            return std::nullopt;
        }
        m_deckSize -= numCards;
        return std::span<const Card>{m_cards.data() + (52 - m_deckSize - numCards), numCards};
    }
    inline constexpr void removeCards(const std::span<const Card> cardsToRemove)
    {
        for (const auto &card : cardsToRemove)
        {
            auto it = std::find(m_cards.begin(), m_cards.end(), card);
            if (it != m_cards.end())
            {
                *it = m_cards[--m_deckSize];
            }
        }
    }
};
#endif // __POKER_DECK_HPP__