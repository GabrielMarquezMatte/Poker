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

struct GamePlayer
{
    bool isActive = true;
    float contribution = 0.0f;
    Player *player;
    std::variant<RandomExecutioner, ConsoleExecutioner> executioner;
};

template <typename T>
concept IExecutioner = requires(T t, const GamePlayer &player, const GameState gameState, const Action action, const BetSizes &betSizes) {
    { t.getAction(player, gameState, betSizes) } -> std::same_as<Action>;
    { t.getBetAmount(player, gameState, action, betSizes) } -> std::same_as<float>;
};

class RandomExecutioner
{
private:
    omp::XoroShiro128Plus m_random;
    omp::FastUniformIntDistribution<int> m_distribution{0, 1};
    std::uniform_real_distribution<float> m_realDist{0.0f, 1.0f};
    Action chooseAction(const GamePlayer &player, const Action defaultAction, bool isCalling, const BetSizes &betSizes)
    {
        if (player.player->getChips() == 0)
        {
            return defaultAction;
        }
        bool executeDefault = m_distribution(m_random) == 0;
        if (executeDefault)
        {
            return defaultAction;
        }
        float toCall = betSizes.currentBet - player.contribution;
        if (player.player->getChips() < toCall)
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
    Action getAction(const GamePlayer &player, const GameState, const BetSizes &betSizes)
    {
        if (betSizes.currentBet == 0)
        {
            return chooseAction(player, Action::Check, false, betSizes);
        }
        return chooseAction(player, Action::Call, true, betSizes);
    }
    float getBetAmount(const GamePlayer &player, const GameState, const Action action, const BetSizes &betSizes)
    {
        switch (action)
        {
        case Action::Fold:
        case Action::Check:
            return 0.0f;
        case Action::Call:
            return betSizes.currentBet - player.contribution;
        case Action::Raise:
            return m_realDist(m_random) * player.player->getChips();
        case Action::AllIn:
            return player.player->getChips();
        default:
            return 0.0f;
        }
    }
};
static_assert(IExecutioner<RandomExecutioner>);

class ConsoleExecutioner
{
public:
    Action getAction(const GamePlayer &player, const GameState, const BetSizes &betSizes)
    {
        std::cout << "Player " << player.player->getChips() << " chips. Current bet: " << betSizes.currentBet << "\n";
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
    float getBetAmount(const GamePlayer &player, const GameState gameState, const Action action, const BetSizes &betSizes)
    {
        switch (action)
        {
        case Action::Fold:
        case Action::Check:
            return 0.0f;
        case Action::AllIn:
            return player.player->getChips();
        case Action::Call:
            return std::min(betSizes.currentBet, player.player->getChips());
        case Action::Raise:
        {
            std::cout << "Enter raise amount (min: " << betSizes.minBet << "): ";
            float amount;
            std::cin >> amount;
            if (amount < betSizes.minBet || amount > player.player->getChips())
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
    struct Result
    {
        ClassificationResult res;
        std::size_t idx;
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
        m_gameState = m_gameState == GameState::End ? GameState::PreFlop : static_cast<GameState>(static_cast<int>(m_gameState) + 1);
    }
    void processPot(Pot &pot, std::span<Result> results)
    {
        auto players = pot.getPlayers();
        std::vector<Result> potResults;
        potResults.reserve(players.size());
        for (auto const &result : results)
        {
            auto it = std::find(players.begin(), players.end(), m_players[result.idx].player);
            if (it != players.end())
            {
                potResults.push_back(result);
            }
        }
        if (potResults.empty())
        {
            return;
        }
        auto bestInPot = std::max_element(potResults.begin(), potResults.end(), [](const Result &a, const Result &b)
                                          { return a.res < b.res; })->res;
        auto numWinners = std::count_if(potResults.begin(), potResults.end(), [&](const Result &result)
                                        { return result.res == bestInPot; });
        float share = pot.getAmount() / numWinners;
        for (auto &result : potResults)
        {
            if (result.res == bestInPot)
            {
                m_players[result.idx].player->addChips(share);
            }
        }
    }
    void finishGame()
    {
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
        for (auto &pot : m_table.getPots())
        {
            auto players = pot.getPlayers();
            std::vector<Result> potResults;
            potResults.reserve(players.size());
            for (auto const &result : results)
            {
                auto it = std::find(players.begin(), players.end(), m_players[result.idx].player);
                if (it != players.end())
                {
                    potResults.push_back(result);
                }
            }
            if (potResults.empty())
            {
                continue;
            }
            auto bestInPot = std::max_element(potResults.begin(), potResults.end(), [](const Result &a, const Result &b)
                                              { return a.res < b.res; })
                                 ->res;
            auto numWinners = std::count_if(potResults.begin(), potResults.end(), [&](const Result &result)
                                            { return result.res == bestInPot; });
            float share = pot.getAmount() / numWinners;
            for (auto &result : potResults)
            {
                if (result.res == bestInPot)
                {
                    m_players[result.idx].player->addChips(share);
                }
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
            player.contribution = 0.0f;
        }
    }
    inline constexpr void postBlinds()
    {
        std::size_t sb = (m_dealerIndex + 1) % m_players.size();
        std::size_t bb = (m_dealerIndex + 2) % m_players.size();
        m_players[sb].player->removeChips(smallBlind);
        m_players[bb].player->removeChips(bigBlind);
        auto &firstPot = m_table.getPots().back();
        firstPot.addAmount(smallBlind);
        firstPot.addAmount(bigBlind);
    }
    inline constexpr float getHighestBet() const
    {
        float highestBet = 0.0f;
        for (const auto &[isActive, contribution, player, executioner] : m_players)
        {
            if (isActive && contribution > highestBet)
            {
                highestBet = contribution;
            }
        }
        return highestBet;
    }
    inline constexpr void removeFromPlayer(GamePlayer &gp, float amount)
    {
        gp.player->removeChips(amount);
        gp.contribution += amount;
        m_table.getPots().back().addAmount(amount);
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
            float toCall = highestBet - gp.contribution;
            Action action = std::visit([&](auto &e)
                                       { return e.getAction(gp, m_gameState, m_betSizes); }, gp.executioner);
            float betAmount = std::visit([&](auto &e)
                                         { return e.getBetAmount(gp, m_gameState, action, m_betSizes); }, gp.executioner);

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
                highestBet = gp.contribution;
                lastToAct = (actor + n - 1) % n;
                break;
            }

            case Action::AllIn:
            {
                float allInAmt = gp.player->getChips();
                removeFromPlayer(gp, allInAmt);
                if (gp.contribution > highestBet)
                {
                    highestBet = gp.contribution;
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
        m_players.push_back({true, 0.0f, player, std::move(executioner)});
    }
    inline constexpr void initializePots()
    {
        m_table.addPot(0.0f, {});
        auto pots = m_table.getPots();
        for (auto &player : m_players)
        {
            player.contribution = 0.0f;
            player.isActive = true;
            for (auto &pot : pots)
            {
                pot.addPlayer(player.player);
            }
        }
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