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
    Action chooseAction(const Player *player, const Action defaultAction, bool isCalling, const BetSizes &betSizes, float contribution)
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
            return Action::AllIn;
        }
        if (!isCalling)
        {
            return Action::Raise;
        }
        bool shouldRaise = m_distribution(m_random) == 0;
        if (!shouldRaise)
        {
            return Action::Call;
        }
        return Action::Raise;
    }

public:
    RandomExecutioner(const omp::XoroShiro128Plus &random) noexcept : m_random(random) {}
    Action getAction(const Player *player, const GameState, const BetSizes &betSizes, float contribution) override
    {
        if (betSizes.currentBet == 0)
        {
            return chooseAction(player, Action::Check, false, betSizes, contribution);
        }
        return chooseAction(player, Action::Call, true, betSizes, contribution);
    }
    float getBetAmount(const Player *player, const GameState, const Action action, const BetSizes &betSizes, float contribution) override
    {
        switch (action)
        {
        case Action::Fold:
        case Action::Check:
            return 0.0f;
        case Action::Call:
            return betSizes.currentBet - contribution;
        case Action::Raise:
            return m_realDist(m_random) * player->getChips();
        case Action::AllIn:
            return player->getChips();
        default:
            return 0.0f;
        }
    }
};
#endif // __POKER_EXECUTIONERS_RANDOM_EXECUTIONER_HPP__