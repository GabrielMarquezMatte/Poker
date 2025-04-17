#ifndef __POKER_CARD_HPP__
#define __POKER_CARD_HPP__
#include <optional>
#include <string_view>
#include <vector>
#include <ostream>
#include "deck.hpp"
#include "card_enums.hpp"
struct Card
{
private:
    std::uint32_t m_mask;
    static inline constexpr std::optional<Rank> parseRank(const char value)
    {
        switch (value)
        {
        case '2':
            return Rank::Two;
        case '3':
            return Rank::Three;
        case '4':
            return Rank::Four;
        case '5':
            return Rank::Five;
        case '6':
            return Rank::Six;
        case '7':
            return Rank::Seven;
        case '8':
            return Rank::Eight;
        case '9':
            return Rank::Nine;
        case 'T':
        case 't':
            return Rank::Ten;
        case 'J':
        case 'j':
            return Rank::Jack;
        case 'Q':
        case 'q':
            return Rank::Queen;
        case 'K':
        case 'k':
            return Rank::King;
        case 'A':
        case 'a':
            return Rank::Ace;
        default:
            return std::nullopt;
        }
    }
    static inline constexpr std::optional<Suit> parseSuit(const char value)
    {
        switch (value)
        {
        case 'H':
        case 'h':
            return Suit::Hearts;
        case 'D':
        case 'd':
            return Suit::Diamonds;
        case 'C':
        case 'c':
            return Suit::Clubs;
        case 'S':
        case 's':
            return Suit::Spades;
        default:
            return std::nullopt;
        }
    }

public:
    inline constexpr Card() = default;
    inline constexpr Card(Suit suit, Rank rank) : m_mask(static_cast<std::uint32_t>(suit) | static_cast<std::uint32_t>(rank << 4)) {}
    inline constexpr Suit getSuit() const
    {
        return static_cast<Suit>(m_mask & 0xF);
    }
    inline constexpr Rank getRank() const
    {
        return static_cast<Rank>(m_mask >> 4);
    }
    inline constexpr std::uint32_t getMask() const
    {
        return m_mask;
    }
    inline constexpr bool operator==(const Card &other) const
    {
        return m_mask == other.m_mask;
    }
    static inline constexpr std::optional<Card> parseCard(const std::string_view str)
    {
        if (str.size() != 2)
        {
            return std::nullopt;
        }
        auto rank = parseRank(str[0]);
        auto suit = parseSuit(str[1]);
        if (!rank || !suit)
        {
            return std::nullopt;
        }
        return Card(*suit, *rank);
    }
};

inline std::ostream &operator<<(std::ostream &os, const Card &card)
{
    static constexpr std::array<std::string_view, 13> suits = {"Hearts", "Diamonds", "Clubs", "Spades"};
    return os << card.getRank() << " of " << card.getSuit();
}
#endif // __POKER_CARD_HPP__