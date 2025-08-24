#ifndef __POKER_EXECUTIONERS_INTERFACE_HPP__
#define __POKER_EXECUTIONERS_INTERFACE_HPP__
#include "../player.hpp"
struct BetSizes
{
    float minBet;
    float currentBet;
};
class IExecutioner
{
public:
    virtual Action getAction(const Player *player, const GameState gameState, const BetSizes &betSizes, float contribution) = 0;
    virtual float getBetAmount(const Player *player, const GameState gameState, const Action action, const BetSizes &betSizes, float contribution) = 0;
    virtual ~IExecutioner() = default;
};
#endif // __POKER_EXECUTIONERS_INTERFACE_HPP__