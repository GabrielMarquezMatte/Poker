#ifndef __POKER_EXECUTIONER_HPP__
#define __POKER_EXECUTIONER_HPP__
#include "random.hpp"
#include "poker_enums.hpp"
#include "table.hpp"
#include "player.hpp"
#include "game.hpp"
#include <cstddef>
#include <iostream>
#include <vector>
#include <variant>

template <typename T>
concept IExecutioner = requires(T t, const Player &player, const Table &table, const GameState gameState, const Action action) {
    { t.getAction(player, table, gameState) } -> std::same_as<Action>;
    { t.getBetAmount(player, table, gameState, action) } -> std::same_as<float>;
};

class RandomExecutioner
{
private:
    omp::XoroShiro128Plus m_random;
    omp::FastUniformIntDistribution<int> m_distribution;
    Action chooseAction(const Player &player, const Table &table, const Action defaultAction, bool isCalling)
    {
        if (player.getChips() == 0)
        {
            return defaultAction;
        }
        bool executeDefault = m_distribution(m_random) == 0;
        if (executeDefault)
        {
            return defaultAction;
        }
        if (player.getChips() < table.getMinBet())
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
    constexpr RandomExecutioner(const omp::XoroShiro128Plus &random) noexcept : m_random(random), m_distribution(0, 1) {}
    Action getAction(const Player &player, const Table &table, const GameState)
    {
        if (table.getCurrentBet() == 0)
        {
            return chooseAction(player, table, Action::Check, false);
        }
        return chooseAction(player, table, Action::Fold, true);
    }
    float getBetAmount(const Player &player, const Table &table, const GameState, const Action action)
    {
        switch (action)
        {
        case Action::Fold:
        case Action::Check:
            return 0.0f;
        case Action::Call:
            return table.getCurrentBet();
        case Action::Raise:
        {
            const float chips = player.getChips();
            std::uniform_real_distribution<float> distribution(0.0f, chips);
            return distribution(m_random);
        }
        case Action::AllIn:
            return player.getChips();
        default:
            return 0.0f;
        }
    }
};
static_assert(IExecutioner<RandomExecutioner>);

class ConsoleExecutioner
{
public:
    Action getAction(const Player &player, const Table &table, const GameState)
    {
        std::cout << "Player " << player.getChips() << " chips. Current bet: " << table.getCurrentBet() << "\n";
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
    float getBetAmount(const Player &player, const Table &table, const GameState gameState, const Action action)
    {
        switch (action)
        {
        case Action::Fold:
        case Action::Check:
            return 0.0f;
        case Action::AllIn:
            return player.getChips();
        case Action::Call:
            return table.getCurrentBet();
        case Action::Raise:
        {
            std::cout << "Enter raise amount (min: " << table.getMinBet() << "): ";
            float amount;
            std::cin >> amount;
            if (amount < table.getMinBet() || amount > player.getChips())
            {
                std::cout << "Invalid raise amount. Trying again.\n";
                return getBetAmount(player, table, gameState, action);
            }
            return amount;
        }
        default:
            return 0.0f;
        }
    }
};
static_assert(IExecutioner<ConsoleExecutioner>);

class Game
{
private:
    struct GamePlayer
    {
        bool isActive = true;
        Player* player;
        std::variant<RandomExecutioner, ConsoleExecutioner> executioner;
    };
    GameState m_gameState;
    Table m_table;
    std::vector<GamePlayer> m_players;
    inline constexpr void changeGameState() noexcept
    {
        switch (m_gameState)
        {
        case GameState::PreFlop:
            m_gameState = GameState::Flop;
            break;
        case GameState::Flop:
            m_gameState = GameState::Turn;
            break;
        case GameState::Turn:
            m_gameState = GameState::River;
            break;
        case GameState::River:
            m_gameState = GameState::Showdown;
            break;
        case GameState::Showdown:
            m_gameState = GameState::End;
            break;
        default:
            break;
        }
    }
    void finishGame()
    {
        struct Result
        {
            ClassificationResult result;
            std::size_t index;
        };
        std::vector<Result> results;
        results.reserve(m_players.size());
        for (std::size_t i = 0; i < m_players.size(); ++i)
        {
            auto &[isActive, player, executioner] = m_players[i];
            if (!isActive)
            {
                continue;
            }
            ClassificationResult result = Hand::classify(Deck::createDeck({player->getCards(), m_table.getTableCards()}));
            results.push_back({result, i});
        }
        std::sort(results.begin(), results.end(), [](const Result &a, const Result &b)
                  { return a.result > b.result; });
        Result winner = results[0];
        std::size_t winners = std::count_if(results.begin(), results.end(), [&](const Result &result)
                                            { return result.result == winner.result; });
        float share = m_table.getPot() / winners;
        for (std::size_t i = 0; i < winners; ++i)
        {
            m_players[results[i].index].player->addChips(share);
        }
    }
    inline constexpr void resetGame() noexcept
    {
        m_gameState = GameState::PreFlop;
        m_table.resetTable();
        m_players.clear();
    }

public:
    constexpr Game() : m_gameState(GameState::PreFlop), m_table() {}
    inline constexpr void addPlayer(Player *player, const std::variant<RandomExecutioner, ConsoleExecutioner> &executioner)
    {
        assert(player->getCards().size() == 2 && "Player must have 2 cards");
        assert(player->getChips() > 0 && "Player must have chips");
        m_players.push_back({true, player, executioner});
    }
    inline constexpr void addCards(const Deck cards)
    {
        assert(cards.size() <= 5 && "Table can have at most 5 cards");
        m_table.addCards(cards);
    }
    bool playRound()
    {
        if (m_gameState == GameState::End)
        {
            finishGame();
            resetGame();
            return false;
        }
        std::size_t betIndex = 0;
        std::size_t numPlayers = m_players.size();
        while (betIndex < numPlayers)
        {
            auto &[isActive, player, executioner] = m_players[betIndex];
            if (!isActive)
            {
                betIndex++;
                continue;
            }
            Action action = std::visit([&](auto &exec)
                                       { return exec.getAction(*player, m_table, m_gameState); }, executioner);
            if (action == Action::Fold)
            {
                isActive = false;
                betIndex++;
                continue;
            }
            float betAmount = std::visit([&](auto &exec)
                                         { return exec.getBetAmount(*player, m_table, m_gameState, action); }, executioner);
            if (betAmount > 0)
            {
                player->removeChips(betAmount);
                m_table.addToPot(betAmount);
            }
            betIndex++;
        }
        changeGameState();
        return true;
    }
};

#endif // __POKER_EXECUTIONER_HPP__