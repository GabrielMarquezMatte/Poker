#ifndef __POKER_DECK_HPP__
#define __POKER_DECK_HPP__
#include <array>
#include <span>
#include <pcg_random.hpp>
#include <random>
#include "card.hpp"
struct Deck
{
private:
    std::int64_t m_cardsBitmask;
    static inline constexpr std::uint64_t calculateCardMask(const Card card)
    {
        std::size_t rankIndex = getRankIndex(card.getRank());
        std::size_t suitIndex = getSuitIndex(card.getSuit());
        return (1ULL << (rankIndex + (suitIndex * 13)));
    }
    static inline constexpr Card calculateCardFromMask(const std::uint64_t mask)
    {
        int count = std::countr_zero(mask);
        std::int64_t rankIndex = count % 13;
        std::int64_t suitIndex = count / 13;
        return Card(static_cast<Suit>(1 << suitIndex), static_cast<Rank>(1 << rankIndex));
    }

public:
    struct DeckIterator
    {
        std::int64_t m_mask;
        inline constexpr DeckIterator(std::int64_t mask) : m_mask(mask) {}
        inline constexpr bool operator!=(const DeckIterator &other) const
        {
            return m_mask != other.m_mask;
        }
        inline constexpr Card operator*() const
        {
            return calculateCardFromMask(m_mask & -m_mask);
        }
        inline constexpr DeckIterator &operator++()
        {
            m_mask &= ~(m_mask & -m_mask);
            return *this;
        }
    };
    static inline constexpr Deck createFullDeck()
    {
        Deck deck;
        deck.m_cardsBitmask = 0xFFFFFFFFFFFFF;
        return deck;
    }
    static inline constexpr Deck emptyDeck()
    {
        Deck deck;
        deck.m_cardsBitmask = 0;
        return deck;
    }
    static inline constexpr Deck createDeck(const std::initializer_list<Deck> decks)
    {
        Deck deck = Deck::emptyDeck();
        for (const Deck d : decks)
        {
            deck.m_cardsBitmask |= d.m_cardsBitmask;
        }
        return deck;
    }
    static inline constexpr Deck parseHand(const std::string_view str)
    {
        // Split from spaces
        Deck deck = Deck::emptyDeck();
        std::size_t start = 0;
        std::size_t end = str.find(' ', start);
        while (end != std::string_view::npos)
        {
            auto card = Card::parseCard(str.substr(start, end - start));
            if (card.has_value())
            {
                deck.addCard(card.value());
            }
            start = end + 1;
            end = str.find(' ', start);
        }
        auto card = Card::parseCard(str.substr(start, end - start));
        if (card.has_value())
        {
            deck.addCard(card.value());
        }
        return deck;
    }
    inline constexpr std::int64_t getMask() const
    {
        return m_cardsBitmask;
    }
    inline constexpr void addCard(const Card card)
    {
        m_cardsBitmask |= calculateCardMask(card);
    }
    inline constexpr void removeCards(const Deck deck)
    {
        m_cardsBitmask &= ~deck.m_cardsBitmask;
    }
    inline std::optional<Card> popRandomCard(pcg64 &rng)
    {
        if (m_cardsBitmask == 0)
        {
            return std::nullopt;
        }
        std::int64_t tmp = m_cardsBitmask;
        std::int64_t bit = 0;
        std::uniform_int_distribution<std::size_t> dist(0, std::popcount(static_cast<std::uint64_t>(tmp)) - 1);
        size_t idx = dist(rng);
        for (size_t i = 0; i <= idx; ++i)
        {
            bit = tmp & -tmp;
            tmp ^= bit;
        }
        m_cardsBitmask &= ~bit;
        return calculateCardFromMask(bit);
    }
    inline constexpr std::optional<Card> dealCard(std::size_t index) const
    {
        if (index >= 52)
        {
            return std::nullopt;
        }
        std::int64_t mask = 1ULL << index;
        if ((m_cardsBitmask & mask) == 0)
        {
            return std::nullopt;
        }
        return calculateCardFromMask(mask);
    }
    inline constexpr std::optional<Card> at(std::size_t index) const
    {
        if (index >= size())
        {
            return std::nullopt;
        }
        std::size_t count = 0;
        std::int64_t mask = m_cardsBitmask;
        while (count < index)
        {
            mask &= ~(mask & -mask);
            ++count;
        }
        return calculateCardFromMask(mask & -mask);
    }
    inline constexpr std::size_t size() const
    {
        return std::popcount(static_cast<std::uint64_t>(m_cardsBitmask));
    }
    inline constexpr DeckIterator begin() const
    {
        return DeckIterator(m_cardsBitmask);
    }
    inline constexpr DeckIterator end() const
    {
        return DeckIterator(0);
    }
};
#endif // __POKER_DECK_HPP__