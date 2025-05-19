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

struct BetSizes
{
    float minBet;
    float currentBet;
};

template <typename T>
concept IExecutioner = requires(T t, const Player &player, const GameState gameState, const Action action, const BetSizes& betSizes)
{
    { t.getAction(player, gameState, betSizes) } -> std::same_as<Action>;
    { t.getBetAmount(player, gameState, action, betSizes) } -> std::same_as<float>;
};

class RandomExecutioner
{
private:
    omp::XoroShiro128Plus m_random;
    omp::FastUniformIntDistribution<int> m_distribution{0, 1};
    std::uniform_real_distribution<float> m_realDist{0.0f, 1.0f};
    Action chooseAction(const Player &player, const Action defaultAction, bool isCalling, const BetSizes& betSizes)
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
        float toCall = betSizes.currentBet - player.getContribution();
        if (player.getChips() < toCall)
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
    Action getAction(const Player &player, const GameState, const BetSizes& betSizes)
    {
        if (betSizes.currentBet == 0)
        {
            return chooseAction(player, Action::Check, false, betSizes);
        }
        return chooseAction(player, Action::Call, true, betSizes);
    }
    float getBetAmount(const Player &player, const GameState, const Action action, const BetSizes& betSizes)
    {
        switch (action)
        {
        case Action::Fold:
        case Action::Check:
            return 0.0f;
        case Action::Call:
            return betSizes.currentBet - player.getContribution();
        case Action::Raise:
            return m_realDist(m_random) * player.getChips();
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
    Action getAction(const Player &player, const GameState, const BetSizes& betSizes)
    {
        std::cout << "Player " << player.getChips() << " chips. Current bet: " << betSizes.currentBet << "\n";
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
    float getBetAmount(const Player &player, const GameState gameState, const Action action, const BetSizes& betSizes)
    {
        switch (action)
        {
        case Action::Fold:
        case Action::Check:
            return 0.0f;
        case Action::AllIn:
            return player.getChips();
        case Action::Call:
            return std::min(betSizes.currentBet, player.getChips());
        case Action::Raise:
        {
            std::cout << "Enter raise amount (min: " << betSizes.minBet << "): ";
            float amount;
            std::cin >> amount;
            if (amount < betSizes.minBet || amount > player.getChips())
            {
                std::cout << "Invalid raise amount. Trying again.\n";
                return getBetAmount(player, gameState, action, betSizes);
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
        Player *player;
        std::variant<RandomExecutioner, ConsoleExecutioner> executioner;
    };
    GameState m_gameState;
    Table m_table;
    std::vector<GamePlayer> m_players;
    std::size_t m_dealerIndex = 0;
    BetSizes m_betSizes;
    static constexpr float smallBlind = 0.01f;
    static constexpr float bigBlind = 0.02f;
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
            ClassificationResult res;
            std::size_t idx;
        };
        std::vector<Result> results;
        results.reserve(m_players.size());
        for (std::size_t i = 0; i < m_players.size(); ++i)
        {
            auto &player = m_players[i];
            if (!player.isActive)
            {
                continue;
            }
            auto score = Hand::classify(Deck::createDeck({player.player->getCards(), m_table.getTableCards()}));
            results.push_back({score, i});
        }
        if (results.empty())
        {
            return;
        }
        std::sort(results.begin(), results.end(), [](auto const &a, auto const &b)
                  { return a.res > b.res; });
        auto best = results[0].res;
        auto winners = std::count_if(results.begin(), results.end(), [&](auto const &r)
                                     { return r.res == best; });
        float share = m_table.getPot() / winners;
        for (auto const &r : results)
        {
            if (r.res == best)
            {
                m_players[r.idx].player->addChips(share);
            }
        }
    }
    inline constexpr void resetGame() noexcept
    {
        m_gameState = GameState::PreFlop;
        m_table.resetTable();
        for (auto &player : m_players)
        {
            player.isActive = true;
            player.player->setContribution(0.0f);
        }
    }
    inline constexpr void postBlinds()
    {
        std::size_t sb = (m_dealerIndex + 1) % m_players.size();
        std::size_t bb = (m_dealerIndex + 2) % m_players.size();
        m_players[sb].player->removeChips(smallBlind);
        m_players[bb].player->removeChips(bigBlind);
        m_table.addToPot(smallBlind);
        m_table.addToPot(bigBlind);
    }
    inline constexpr float getHighestBet() const
    {
        float highestBet = 0.0f;
        for (const auto &[isActive, player, executioner] : m_players)
        {
            if (isActive && player->getContribution() > highestBet)
            {
                highestBet = player->getContribution();
            }
        }
        return highestBet;
    }
    inline constexpr void removeFromPlayer(GamePlayer &gp, float amount)
    {
        gp.player->removeChips(amount);
        m_table.addToPot(amount);
    }
    inline constexpr void prepareRound()
    {
        m_betSizes.currentBet = getHighestBet();
        m_betSizes.minBet = m_betSizes.currentBet + bigBlind;
    }
    void bettingRound()
    {
        std::size_t n = m_players.size();
        std::size_t activeCount = std::count_if(m_players.begin(), m_players.end(), [](auto const &gp)
                                                { return gp.isActive; });
        std::size_t actor = (m_dealerIndex + 3) % n;
        std::size_t lastToAct = (actor + n - 1) % n;
        float highestBet = getHighestBet();
        while (true)
        {
            auto &gp = m_players[actor];
            if (!gp.isActive)
            {
                actor = (actor + 1) % n;
                continue;
            }
            float toCall = highestBet - gp.player->getContribution();
            Action action = std::visit([&](auto &e)
                                       { return e.getAction(*gp.player, m_gameState, m_betSizes); }, gp.executioner);
            float betAmount = std::visit([&](auto &e)
                                         { return e.getBetAmount(*gp.player, m_gameState, action, m_betSizes); }, gp.executioner);

            switch (action)
            {
            case Action::Fold:
                gp.isActive = false;
                --activeCount;
                break;

            case Action::Call:
                betAmount = std::min(betAmount, toCall);
                removeFromPlayer(gp, betAmount);
                break;

            case Action::Raise:
            {
                float raiseAmt = std::max(betAmount, toCall + smallBlind);
                removeFromPlayer(gp, raiseAmt);
                highestBet = gp.player->getContribution();
                lastToAct = (actor + n - 1) % n;
                break;
            }

            case Action::AllIn:
            {
                float allInAmt = gp.player->getChips();
                removeFromPlayer(gp, allInAmt);
                if (gp.player->getContribution() > highestBet)
                {
                    highestBet = gp.player->getContribution();
                    lastToAct = (actor + n - 1) % n;
                }
                break;
            }
            }

            if (activeCount <= 1 || actor == lastToAct)
            {
                break;
            }

            actor = (actor + 1) % n;
        }
    }

public:
    constexpr Game() : m_gameState(GameState::PreFlop), m_table() {}
    inline constexpr void addPlayer(Player *player, std::variant<RandomExecutioner, ConsoleExecutioner> executioner)
    {
        assert(player->getCards().size() == 2 && "Player must have 2 cards");
        assert(player->getChips() > 0 && "Player must have chips");
        m_players.push_back({true, player, std::move(executioner)});
    }
    inline constexpr void addCards(const Deck cards)
    {
        assert(cards.size() <= 5 && "Table can have at most 5 cards");
        m_table.addCards(cards);
    }
    bool playRound()
    {
        if (m_gameState == GameState::PreFlop)
        {
            postBlinds();
            m_dealerIndex = (m_dealerIndex + 1) % m_players.size();
            changeGameState();
            return true;
        }
        if (m_gameState == GameState::End)
        {
            finishGame();
            resetGame();
            return false;
        }
        bettingRound();
        changeGameState();
        return true;
    }
};

#endif // __POKER_EXECUTIONER_HPP__