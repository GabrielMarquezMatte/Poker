#ifndef __POKER_PLAYER_HPP__
#define __POKER_PLAYER_HPP__
#include "../deck.hpp"
#include "poker_enums.hpp"
struct Player
{
    std::size_t id = 0;
    Deck hole{};
    std::uint32_t chips = 0;
    std::uint32_t committed = 0;
    std::uint32_t invested = 0;
    bool folded = false;
    bool all_in = false;
    bool has_hole = false;
    inline constexpr bool eligible() const noexcept { return alive() && !all_in; }
    inline constexpr bool alive() const noexcept { return !folded && chips >= 0 && has_hole; } // can be all-in but still in the hand
    constexpr Player() = default;
    constexpr Player(std::size_t id_, int chips_) : id(id_), chips(chips_) {}
};
struct ActionStruct
{
    ActionType type;
    std::uint32_t amount = 0;
};
#endif // __POKER_PLAYER_HPP__