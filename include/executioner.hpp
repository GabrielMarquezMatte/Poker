#ifndef __POKER_EXECUTIONER_HPP__
#define __POKER_EXECUTIONER_HPP__
#include "random.hpp"
#include "poker_enums.hpp"
#include "table.hpp"
#include "player.hpp"
#include "game.hpp"
#include "game/game_player.hpp"
#include "game/pot_manager.hpp"
#include "executioners/interface.hpp"
#include <cstddef>
#include <iostream>
#include <vector>

class Game
{
private:
    struct Result
    {
        ClassificationResult res;
        std::size_t idx;
    };
    Table m_table;
    PotManager m_potManager;
    std::vector<GamePlayer> m_players;
    std::size_t m_dealerIndex = 0;
    BetSizes m_betSizes;
    GameState m_gameState;
    inline constexpr GameState nextState(GameState s) noexcept
    {
        sizeof(Game);
        sizeof(Table);
        sizeof(PotManager);
        sizeof(std::vector<GamePlayer>);
        sizeof(BetSizes);
        sizeof(GameState);
        switch (s)
        {
        case GameState::PreFlop:
            return GameState::Flop;
        case GameState::Flop:
            return GameState::Turn;
        case GameState::Turn:
            return GameState::River;
        case GameState::River:
            return GameState::End;
        case GameState::End:
            return GameState::PreFlop;
        }
        return GameState::End;
    }
    void finishGame()
    {
        m_potManager.distribute(m_table, m_players);
        for (auto &player : m_players)
        {
            player.isActive = false;
        }
        m_gameState = GameState::End;
    }
    inline constexpr void resetGame() noexcept
    {
        m_gameState = GameState::PreFlop;
        m_table.resetTable();
        for (auto &player : m_players)
        {
            player.isActive = true;
        }
    }
    inline constexpr void postBlinds(const float smallBlind, const float bigBlind) noexcept
    {
        std::size_t sb = (m_dealerIndex + 1) % m_players.size();
        std::size_t bb = (m_dealerIndex + 2) % m_players.size();
        m_potManager.collectBets(m_table, m_players[sb].player, smallBlind);
        m_potManager.collectBets(m_table, m_players[bb].player, bigBlind);
    }
    inline constexpr float getHighestBet() const
    {
        float highestBet = 0.0f;
        for (const auto &[isActive, player, executioner] : m_players)
        {
            float contribution = m_potManager.getContribution(player);
            if (isActive && contribution > highestBet)
            {
                highestBet = contribution;
            }
        }
        return highestBet;
    }
    inline constexpr void prepareRound(const float bigBlind)
    {
        m_betSizes.currentBet = getHighestBet();
        m_betSizes.minBet = m_betSizes.currentBet + bigBlind;
    }
    void bettingRound(const float smallBlind)
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
            float contribution = m_potManager.getContribution(gp.player);
            float toCall = highestBet - contribution;
            Action action = gp.executioner->getAction(gp.player, m_gameState, m_betSizes, contribution);
            float betAmount = gp.executioner->getBetAmount(gp.player, m_gameState, action, m_betSizes, contribution);

            switch (action)
            {
            case Action::Fold:
                gp.isActive = false;
                --activeCount;
                break;

            case Action::Call:
                betAmount = std::min(betAmount, toCall);
                if (betAmount > 0)
                {
                    m_potManager.collectBets(m_table, gp.player, betAmount);
                }
                break;

            case Action::Raise:
            {
                float raiseAmt = std::max(betAmount, toCall + smallBlind);
                m_potManager.collectBets(m_table, gp.player, raiseAmt);
                highestBet = contribution + raiseAmt;
                lastToAct = (actor + n - 1) % n;
                break;
            }

            case Action::AllIn:
            {
                float allInAmt = gp.player->getChips();
                m_potManager.collectBets(m_table, gp.player, allInAmt);
                if (contribution > highestBet)
                {
                    highestBet = contribution;
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
    inline constexpr void addPlayer(std::unique_ptr<IExecutioner> &&executioner, Player *player)
    {
        assert(player->getCards().size() == 2 && "Player must have 2 cards");
        assert(player->getChips() > 0 && "Player must have chips");
        m_players.push_back({std::move(executioner), player, true});
    }
    inline constexpr void addCards(const Deck cards)
    {
        assert(cards.size() <= 5 && "Table can have at most 5 cards");
        m_table.addCards(cards);
    }
    bool playRound(const float smallBlind, const float bigBlind)
    {
        if (m_gameState == GameState::PreFlop)
        {
            m_potManager.initialize(m_table, m_players);
            postBlinds(smallBlind, bigBlind);
            m_dealerIndex = (m_dealerIndex + 1) % m_players.size();
            m_gameState = GameState::Flop;
            return true;
        }
        if (m_gameState == GameState::End)
        {
            finishGame();
            resetGame();
            return false;
        }
        prepareRound(bigBlind);
        bettingRound(smallBlind);
        m_gameState = nextState(m_gameState);
        return true;
    }
};

#endif // __POKER_EXECUTIONER_HPP__