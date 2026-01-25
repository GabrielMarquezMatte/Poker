#ifndef __POKER_BLINDS_HPP__
#define __POKER_BLINDS_HPP__
#include <cstdint>
struct Blinds
{
    std::uint32_t smallBlind = 50;
    std::uint32_t bigBlind = 100;
};
#endif // __POKER_BLINDS_HPP__