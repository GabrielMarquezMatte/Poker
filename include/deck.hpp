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
                deck[index++] = Card(static_cast<Suit>(1ull << suit), static_cast<Rank>(1ull << rank));
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

    constexpr explicit Deck(std::uint64_t mask) : m_cardsBitmask(mask) {}
    static inline constexpr Deck from_mask(std::uint64_t m) noexcept { return Deck(m); }

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
    inline constexpr Deck() = default;
    static inline constexpr Deck createFullDeck() noexcept
    {
        return Deck::from_mask((1ull << 52ull) - 1ull);
    }
    static inline constexpr Deck emptyDeck() noexcept
    {
        return Deck::from_mask(0);
    }
    static inline constexpr Deck createDeck(const std::initializer_list<Deck> decks) noexcept
    {
        std::uint64_t mask = 0;
        for (const Deck& d : decks)
        {
            mask |= d.m_cardsBitmask;
        }
        return Deck::from_mask(mask);
    }
    static inline constexpr Deck createDeck(const std::vector<Deck> &decks) noexcept
    {
        std::uint64_t mask = 0;
        for (const Deck& d : decks)
        {
            mask |= d.m_cardsBitmask;
        }
        return Deck::from_mask(mask);
    }
    static inline constexpr Deck createDeck(const std::initializer_list<Card> cards) noexcept
    {
        std::uint64_t mask = 0;
        for (const Card& card : cards)
        {
            mask |= Deck::calculateCardMask(card);
        }
        return Deck::from_mask(mask);
    }
    static inline constexpr Deck createDeck(const std::vector<Card> &cards) noexcept
    {
        std::uint64_t mask = 0;
        for (const Card& card : cards)
        {
            mask |= Deck::calculateCardMask(card);
        }
        return Deck::from_mask(mask);
    }
    static inline constexpr Deck parseHand(const std::string_view str) noexcept
    {
        // Split from spaces
        std::uint64_t mask = 0;
        std::size_t start = 0;
        std::size_t end = str.find(' ', start);
        while (end != std::string_view::npos)
        {
            auto card = Card::parseCard(str.substr(start, end - start));
            if (card.has_value())
            {
                mask |= Deck::calculateCardMask(card.value());
            }
            start = end + 1;
            end = str.find(' ', start);
        }
        auto card = Card::parseCard(str.substr(start, end - start));
        if (card.has_value())
        {
            mask |= Deck::calculateCardMask(card.value());
        }
        return Deck::from_mask(mask);
    }
    inline constexpr std::uint64_t getMask() const noexcept
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
    template<typename TRng>
    inline constexpr Deck popPair(TRng &rng) noexcept
    {
        std::uint64_t tmp = m_cardsBitmask;
        std::size_t count = size();
        std::uint64_t rand_val = rng();
        std::uint32_t idx1 = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(rand_val)) * count) >> 32;
        std::uint32_t idx2 = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(rand_val >> 32)) * (count - 1)) >> 32;
        if (idx2 >= idx1) 
        {
            idx2++;
        }
        std::uint64_t pdep_mask = (1ULL << idx1) | (1ULL << idx2);
        std::uint64_t result = pdep(pdep_mask, tmp);
        m_cardsBitmask &= ~result;
        return Deck::from_mask(result);
    }
    template<typename TRng>
    inline constexpr Deck popRandomCards(TRng &rng, std::size_t count) noexcept
    {
        const std::size_t total = size();
        if (count >= total)
        {
            Deck all = *this;
            m_cardsBitmask = 0;
            return all;
        }
        const bool choose_complement = (count > (total >> 1));
        std::size_t k = choose_complement ? (total - count) : count;
        std::uint64_t mask = m_cardsBitmask;
        std::uint64_t chosen = 0;
        omp::FastUniformIntDistribution<std::uint8_t> dist;
        std::size_t remaining = total;
        for (std::size_t i = 0; i < k; ++i, --remaining)
        {
            std::uint8_t idx = dist(rng, {0, static_cast<std::uint8_t>(remaining - 1)});
            std::uint64_t bit = pdep(1ull << idx, mask);
            chosen |= bit;
            mask &= ~bit;
        }
        if (choose_complement)
        {
            Deck out = Deck::from_mask(m_cardsBitmask & ~chosen);
            m_cardsBitmask = chosen;
            return out;
        }
        Deck out = Deck::from_mask(chosen);
        m_cardsBitmask = m_cardsBitmask & ~chosen;
        return out;
    }
    template<typename TRng>
    inline Card popRandomCard(TRng &rng)
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
        return Deck::from_mask(resultMask);
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
        return std::popcount(m_cardsBitmask);
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