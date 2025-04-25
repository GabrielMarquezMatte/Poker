#ifndef __POKER_DECK_HPP__
#define __POKER_DECK_HPP__
#include <array>
#include <span>
#include <pcg_random.hpp>
#include <random>
#include <immintrin.h>
#include "card.hpp"
struct Deck
{
private:
    std::int64_t m_cardsBitmask;
    static constexpr std::array<Card, 52> m_deck = []()
    {
        std::array<Card, 52> deck;
        std::size_t index = 0;
        for (int suit = 0; suit < 4; ++suit)
        {
            for (int rank = 0; rank < 13; ++rank)
            {
                deck[index++] = Card(static_cast<Suit>(1 << suit), static_cast<Rank>(1 << rank));
            }
        }
        return deck;
    }();
    static inline constexpr std::uint64_t calculateCardMask(const Card card)
    {
        std::size_t rankIndex = getRankIndex(card.getRank());
        std::size_t suitIndex = getSuitIndex(card.getSuit());
        return (1ULL << (rankIndex + (suitIndex * 13)));
    }
    static inline constexpr Card calculateCardFromMask(const std::uint64_t mask)
    {
        int count = std::countr_zero(mask);
        return m_deck[count];
    }

    static inline constexpr std::uint64_t pdep(std::uint64_t x, std::uint64_t mask)
    {
        if (!std::is_constant_evaluated())
        {
            return _pdep_u64(x, mask);
        }
        std::uint64_t result = 0;
        for (std::size_t i = 0; i < 64; ++i)
        {
            if (mask & (1ULL << i))
            {
                result |= (x & (1ULL << i)) ? (1ULL << i) : 0;
            }
        }
        return result;
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
    static inline constexpr Deck createDeck(const std::initializer_list<Card> cards)
    {
        Deck deck = Deck::emptyDeck();
        for (const Card card : cards)
        {
            deck.addCard(card);
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
    inline Card popRandomCard(pcg64 &rng)
    {
        std::int64_t tmp = m_cardsBitmask;
        std::uniform_int_distribution<std::size_t> dist(0, std::popcount(static_cast<std::uint64_t>(tmp)) - 1);
        std::uint64_t bit = pdep(1ULL << dist(rng), tmp);
        m_cardsBitmask &= ~bit;
        return calculateCardFromMask(bit);
    }
    inline constexpr std::optional<Card> at(std::size_t index) const
    {
        if (index >= size())
        {
            return std::nullopt;
        }
        std::uint64_t tmp = m_cardsBitmask;
        std::uint64_t bit = pdep(1ULL << index, tmp);
        return calculateCardFromMask(bit);
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