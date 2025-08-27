#ifndef __POKER_PLAYER_HPP__
#define __POKER_PLAYER_HPP__
#include "../deck.hpp"
#include "poker_enums.hpp"
struct Player
{
    std::size_t id = 0;
    int chips = 0;
    bool folded = false;
    bool all_in = false;
    int committed = 0;
    Deck hole{};
    bool has_hole = false;
    inline constexpr bool eligible() const noexcept { return !folded && !all_in; }
    inline constexpr bool alive() const noexcept { return !folded; } // can be all-in but still in the hand
};
struct ActionStruct
{
    ActionType type;
    int amount = 0;
};
#endif // __POKER_PLAYER_HPP__