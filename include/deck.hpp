#ifndef __POKER_DECK_HPP__
#define __POKER_DECK_HPP__
#include <array>
#include <random>
#include <immintrin.h>
#include "card.hpp"
#include "random.hpp"
struct Deck
{
private:
    std::uint64_t m_cardsBitmask;
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
    static inline constexpr std::uint64_t calculateCardMask(const Card card) noexcept
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

    static inline constexpr std::uint64_t pdep(std::uint64_t x, std::uint64_t mask) noexcept
    {
        if (!std::is_constant_evaluated())
        {
            return _pdep_u64(x, mask);
        }
        std::uint64_t res = 0;
        for (std::uint64_t m = mask; m; m &= m - 1)
        {
            std::uint64_t lowest = m & -static_cast<std::int64_t>(m);
            if (x & 1)
            {
                res |= lowest;
            }
            x >>= 1;
        }
        return res;
    }

public:
    struct DeckIterator
    {
        std::uint64_t m_mask;
        inline constexpr DeckIterator(std::uint64_t mask) noexcept : m_mask(mask) {}
        inline constexpr bool operator!=(const DeckIterator &other) const noexcept
        {
            return m_mask != other.m_mask;
        }
        inline constexpr Card operator*() const noexcept
        {
            return calculateCardFromMask(m_mask & -static_cast<std::int64_t>(m_mask));
        }
        inline constexpr DeckIterator &operator++() noexcept
        {
            m_mask &= ~(m_mask & -static_cast<std::int64_t>(m_mask));
            return *this;
        }
    };
    static inline constexpr Deck createFullDeck() noexcept
    {
        Deck deck;
        deck.m_cardsBitmask = 0xFFFFFFFFFFFFF;
        return deck;
    }
    static inline constexpr Deck emptyDeck() noexcept
    {
        Deck deck;
        deck.m_cardsBitmask = 0;
        return deck;
    }
    static inline constexpr Deck createDeck(const std::initializer_list<Deck> decks) noexcept
    {
        Deck deck = Deck::emptyDeck();
        for (const Deck d : decks)
        {
            deck.m_cardsBitmask |= d.m_cardsBitmask;
        }
        return deck;
    }
    static inline constexpr Deck createDeck(const std::vector<Deck> &decks) noexcept
    {
        Deck deck = Deck::emptyDeck();
        for (const Deck d : decks)
        {
            deck.m_cardsBitmask |= d.m_cardsBitmask;
        }
        return deck;
    }
    static inline constexpr Deck createDeck(const std::initializer_list<Card> cards) noexcept
    {
        Deck deck = Deck::emptyDeck();
        for (const Card card : cards)
        {
            deck.addCard(card);
        }
        return deck;
    }
    static inline constexpr Deck createDeck(const std::vector<Card> &cards) noexcept
    {
        Deck deck = Deck::emptyDeck();
        for (const Card card : cards)
        {
            deck.addCard(card);
        }
        return deck;
    }
    static inline constexpr Deck parseHand(const std::string_view str) noexcept
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
    inline constexpr std::int64_t getMask() const noexcept
    {
        return m_cardsBitmask;
    }
    inline constexpr void addCard(const Card card) noexcept
    {
        m_cardsBitmask |= calculateCardMask(card);
    }
    inline constexpr void addCards(const Deck deck) noexcept
    {
        m_cardsBitmask |= deck.m_cardsBitmask;
    }
    inline constexpr void removeCards(const Deck deck) noexcept
    {
        m_cardsBitmask &= ~deck.m_cardsBitmask;
    }
    inline constexpr void removeCard(const Card card) noexcept
    {
        m_cardsBitmask &= ~calculateCardMask(card);
    }
    inline constexpr Deck popRandomCards(omp::XoroShiro128Plus &rng, std::size_t count) noexcept
    {
        const std::size_t total = size();
        if (count >= total)
        {
            Deck all = *this;
            m_cardsBitmask = 0;
            return all;
        }
        std::uint64_t mask = m_cardsBitmask;
        std::uint64_t resultMask = 0;
        omp::FastUniformIntDistribution<std::uint8_t> dist;
        for (std::size_t i = 0; i < count; ++i)
        {
            int remaining = std::popcount(mask);
            std::uint8_t index = dist(rng, decltype(dist)::param_type(0, static_cast<std::uint8_t>(remaining - 1)));
            std::uint64_t bit = pdep(1ULL << index, mask);
            resultMask |= bit;
            mask &= ~bit;
        }
        Deck result = Deck::emptyDeck();
        result.m_cardsBitmask = resultMask;
        m_cardsBitmask &= ~resultMask;
        return result;
    }
    inline Card popRandomCard(omp::XoroShiro128Plus &rng)
    {
        std::uint64_t tmp = m_cardsBitmask;
        omp::FastUniformIntDistribution<std::size_t> dist(0, std::popcount(tmp) - 1);
        std::uint64_t bit = pdep(1ULL << dist(rng), tmp);
        m_cardsBitmask &= ~bit;
        return calculateCardFromMask(bit);
    }
    inline constexpr Card popCard() noexcept
    {
        std::uint64_t tmp = m_cardsBitmask;
        std::uint64_t bit = tmp & -static_cast<std::int64_t>(tmp);
        m_cardsBitmask &= ~bit;
        return calculateCardFromMask(bit);
    }
    inline constexpr Deck popCards(std::size_t count) noexcept
    {
        std::size_t sz = size();
        if (count >= sz)
        {
            Deck all = *this;
            m_cardsBitmask = 0;
            return all;
        }
        std::uint64_t indexMask = (1ull << count) - 1;
        std::uint64_t resultMask = pdep(indexMask, m_cardsBitmask);
        m_cardsBitmask &= ~resultMask;
        Deck result = Deck::emptyDeck();
        result.m_cardsBitmask = resultMask;
        return result;
    }
    inline constexpr std::optional<Card> at(std::size_t index) const noexcept
    {
        if (index >= size())
        {
            return std::nullopt;
        }
        std::uint64_t tmp = m_cardsBitmask;
        std::uint64_t bit = pdep(1ULL << index, tmp);
        return calculateCardFromMask(bit);
    }
    inline constexpr std::size_t size() const noexcept
    {
        return std::popcount(static_cast<std::uint64_t>(m_cardsBitmask));
    }
    inline constexpr DeckIterator begin() const noexcept
    {
        return DeckIterator(m_cardsBitmask);
    }
    inline constexpr DeckIterator end() const noexcept
    {
        return DeckIterator(0);
    }
    inline constexpr bool operator==(const Deck &other) const noexcept
    {
        return m_cardsBitmask == other.m_cardsBitmask;
    }
    inline constexpr bool operator!=(const Deck &other) const noexcept
    {
        return m_cardsBitmask != other.m_cardsBitmask;
    }
};

inline std::ostream &operator<<(std::ostream &os, const Deck &deck)
{
    os << "Deck: ";
    for (const auto &card : deck)
    {
        os << card << " ";
    }
    return os;
}
#endif // __POKER_DECK_HPP__