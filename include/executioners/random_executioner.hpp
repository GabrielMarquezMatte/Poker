#ifndef __POKER_EXECUTIONERS_RANDOM_EXECUTIONER_HPP__
#define __POKER_EXECUTIONERS_RANDOM_EXECUTIONER_HPP__
#include "interface.hpp"
#include "../random.hpp"
#include "../poker_enums.hpp"
#include "../player.hpp"
class RandomExecutioner : public IExecutioner
{
private:
    omp::XoroShiro128Plus m_random;
    omp::FastUniformIntDistribution<int> m_distribution{0, 1};
    std::uniform_real_distribution<float> m_realDist{0.0f, 1.0f};
    ActionType chooseAction(const Player *player, const ActionType defaultAction, bool isCalling, const BetSizes &betSizes, float contribution)
    {
        if (player->getChips() == 0)
        {
            return defaultAction;
        }
        bool executeDefault = m_distribution(m_random) == 0;
        if (executeDefault)
        {
            return defaultAction;
        }
        float toCall = betSizes.currentBet - contribution;
        if (player->getChips() < toCall)
        {
            return ActionType::AllIn;
        }
        if (!isCalling)
        {
            return ActionType::Raise;
        }
        bool shouldRaise = m_distribution(m_random) == 0;
        if (!shouldRaise)
        {
            return ActionType::Call;
        }
        return ActionType::Raise;
    }

public:
    RandomExecutioner(const omp::XoroShiro128Plus &random) noexcept : m_random(random) {}
    ActionType getAction(const Player *player, const GameState, const BetSizes &betSizes, float contribution) override
    {
        if (betSizes.currentBet == 0)
        {
            return chooseAction(player, ActionType::Check, false, betSizes, contribution);
        }
        return chooseAction(player, ActionType::Call, true, betSizes, contribution);
    }
    float getBetAmount(const Player *player, const GameState, const ActionType action, const BetSizes &betSizes, float contribution) override
    {
        switch (action)
        {
        case ActionType::Fold:
        case ActionType::Check:
            return 0.0f;
        case ActionType::Call:
            return betSizes.currentBet - contribution;
        case ActionType::Raise:
            return m_realDist(m_random) * player->getChips();
        case ActionType::AllIn:
            return player->getChips();
        default:
            return 0.0f;
        }
    }
};
#endif // __POKER_EXECUTIONERS_RANDOM_EXECUTIONER_HPP__