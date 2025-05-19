#ifndef __POKER_ENUMS_HPP__
#define __POKER_ENUMS_HPP__
enum class GameState
{
    PreFlop,
    Flop,
    Turn,
    River,
    Showdown,
    End
};
enum class Action
{
    Fold,
    Call,
    Raise,
    Check,
    AllIn,
};
#endif // __POKER_ENUMS_HPP__