#ifndef __POKER_GAME_PLAYER_HPP__
#define __POKER_GAME_PLAYER_HPP__
#include "../executioners/interface.hpp"
#include "../player.hpp"
#include "../executioners/random_executioner.hpp"
#include "../executioners/console_executioner.hpp"
#include <memory>
struct GamePlayer
{
    std::unique_ptr<IExecutioner> executioner;
    Player* player;
    bool isActive;
};
#endif // __POKER_GAME_PLAYER_HPP__