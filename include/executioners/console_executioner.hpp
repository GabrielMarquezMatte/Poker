#ifndef __POKER_EXECUTIONERS_CONSOLE_EXECUTIONER_HPP__
#define __POKER_EXECUTIONERS_CONSOLE_EXECUTIONER_HPP__
#include "interface.hpp"
#include <iostream>
#include "../poker_enums.hpp"
#include "../player.hpp"
class ConsoleExecutioner : public IExecutioner
{
public:
    Action getAction(const Player *player, const GameState, const BetSizes &betSizes, float) override
    {
        std::cout << "Player " << player->getChips() << " chips. Current bet: " << betSizes.currentBet << "\n";
        std::cout << "Choose action (0: Fold, 1: Call, 2: Raise, 3: Check, 4: AllIn): ";
        int choice;
        std::cin >> choice;
        switch (choice)
        {
        case 0:
            return Action::Fold;
        case 1:
            return Action::Call;
        case 2:
            return Action::Raise;
        case 3:
            return Action::Check;
        case 4:
            return Action::AllIn;
        default:
            return Action::Fold;
        }
    }
    float getBetAmount(const Player *player, const GameState gameState, const Action action, const BetSizes &betSizes, float contribution) override
    {
        switch (action)
        {
        case Action::Fold:
        case Action::Check:
            return 0.0f;
        case Action::AllIn:
            return player->getChips();
        case Action::Call:
            return std::min(betSizes.currentBet, player->getChips());
        case Action::Raise:
        {
            std::cout << "Enter raise amount (min: " << betSizes.minBet << "): ";
            float amount;
            std::cin >> amount;
            if (amount < betSizes.minBet || amount > player->getChips())
            {
                std::cout << "Invalid raise amount. Trying again.\n";
                return getBetAmount(player, gameState, action, betSizes, contribution);
            }
            return amount;
        }
        default:
            return 0.0f;
        }
    }
};
#endif // __POKER_EXECUTIONERS_CONSOLE_EXECUTIONER_HPP__