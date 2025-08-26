#ifndef __POKER_ENUMS_HPP__
#define __POKER_ENUMS_HPP__
enum class GameState
{
    PreDeal,
    PreFlop,
    Flop,
    Turn,
    River,
    Showdown,
    Finished,
    End = Finished,
};
enum class ActionType
{
    Fold,
    Check,
    Call,
    Bet,
    Raise,
    AllIn
};
#endif // __POKER_ENUMS_HPP__