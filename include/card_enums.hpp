#ifndef __POKER_RANK_HPP__
#define __POKER_RANK_HPP__
#include <cstdint>
#include <bit>
#include <cstddef>
#include <array>
#include <ostream>
enum class Suit
{
    Hearts = 1 << 0,
    Diamonds = 1 << 1,
    Clubs = 1 << 2,
    Spades = 1 << 3,
};
enum class Rank
{
    Two = 1 << 0,
    Three = 1 << 1,
    Four = 1 << 2,
    Five = 1 << 3,
    Six = 1 << 4,
    Seven = 1 << 5,
    Eight = 1 << 6,
    Nine = 1 << 7,
    Ten = 1 << 8,
    Jack = 1 << 9,
    Queen = 1 << 10,
    King = 1 << 11,
    Ace = 1 << 12,
    LowStraight = Two | Three | Four | Five | Ace,
    HighStraight = Ten | Jack | Queen | King | Ace,
};
enum class Classification
{
    HighCard,
    Pair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
    RoyalFlush,
};
inline constexpr Rank operator|(Rank lhs, Rank rhs)
{
    return static_cast<Rank>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline constexpr Rank operator&(Rank lhs, Rank rhs)
{
    return static_cast<Rank>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline constexpr Rank operator^(Rank lhs, Rank rhs)
{
    return static_cast<Rank>(static_cast<uint32_t>(lhs) ^ static_cast<uint32_t>(rhs));
}
inline constexpr Rank operator~(Rank lhs)
{
    return static_cast<Rank>(~static_cast<uint32_t>(lhs));
}
inline constexpr Rank operator>>(Rank lhs, uint32_t rhs)
{
    return static_cast<Rank>(static_cast<uint32_t>(lhs) >> rhs);
}
inline constexpr Rank operator<<(Rank lhs, uint32_t rhs)
{
    return static_cast<Rank>(static_cast<uint32_t>(lhs) << rhs);
}
inline constexpr Rank operator-(Rank lhs, uint32_t rhs)
{
    return static_cast<Rank>(static_cast<uint32_t>(lhs) - rhs);
}
inline constexpr std::size_t getRankIndex(Rank rank)
{
    return std::countr_zero(static_cast<uint32_t>(rank));
}
inline constexpr std::size_t getSuitIndex(Suit suit)
{
    return std::countr_zero(static_cast<uint32_t>(suit));
}

inline std::ostream &operator<<(std::ostream &os, const Suit suit)
{
    static constexpr std::array<std::string_view, 4> suits = {"Hearts", "Diamonds", "Clubs", "Spades"};
    return os << suits[getSuitIndex(suit)];
}
inline std::ostream &operator<<(std::ostream &os, const Rank rank)
{
    static constexpr std::array<std::string_view, 13> ranks = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"};
    return os << ranks[getRankIndex(rank)];
}
#endif // __POKER_RANK_HPP__