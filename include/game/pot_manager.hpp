#ifndef __POKER_GAME_POT_MANAGER_HPP__
#define __POKER_GAME_POT_MANAGER_HPP__
#include "../poker_enums.hpp"
#include "../player.hpp"
#include "../table.hpp"
#include "game_player.hpp"
#include "../executioners/interface.hpp"
#include <unordered_map>

class PotManager
{
private:
    std::vector<std::pair<Player *, float>> m_contributions;
    static inline constexpr std::vector<Player *> getWinners(const std::span<const GamePlayer> players, const Pot &pot, const Deck communityCards) noexcept
    {
        struct Res
        {
            ClassificationResult score;
            Player *p;
        };
        std::vector<Res> contenders;
        contenders.reserve(players.size());
        for (auto const &gp : players)
        {
            if (!gp.isActive)
            {
                continue;
            }
            if (!pot.containsPlayer(gp.player))
            {
                continue;
            }
            auto cards = gp.player->getCards();
            auto score = Hand::classify(Deck::createDeck({cards, communityCards}));
            contenders.push_back({score, gp.player});
        }
        if (contenders.empty())
        {
            return {};
        }
        auto bestScore = std::max_element(
                             contenders.begin(), contenders.end(),
                             [](auto const &a, auto const &b)
                             { return a.score < b.score; })
                             ->score;
        std::vector<Player *> winners;
        winners.reserve(contenders.size());
        for (auto const &r : contenders)
        {
            if (r.score == bestScore)
            {
                winners.push_back(r.p);
            }
        }
        return winners;
    }

public:
    inline constexpr float getContribution(Player *player) const noexcept
    {
        for (const auto &[p, contribution] : m_contributions)
        {
            if (p == player)
            {
                return contribution;
            }
        }
        return 0.0f;
    }
    inline constexpr void initialize(Table &table, const std::span<GamePlayer> players) noexcept
    {
        m_contributions.clear();
        m_contributions.reserve(players.size());
        std::vector<Player *> allPlayers;
        allPlayers.reserve(players.size());
        for (auto const &gp : players)
        {
            m_contributions.push_back({gp.player, 0.0f});
            allPlayers.push_back(gp.player);
        }
        table.addPot(0.0f, allPlayers);
    }
    inline constexpr void collectBets(Table &table, Player *player, float amount) noexcept
    {
        assert(amount > 0.0f && "Bet amount must be positive");
        auto it = std::find_if(m_contributions.begin(), m_contributions.end(),
                               [player](const auto &pair) { return pair.first == player; });
        if (it != m_contributions.end())
        {
            it->second += amount;
        }
        auto pots = table.getPots();
        if (!pots.empty())
        {
            pots.back().addAmount(amount);
        }
    }
    constexpr void distribute(Table &table, const std::span<const GamePlayer> players) noexcept
    {
        std::vector<Player *> remaining;
        remaining.reserve(m_contributions.size());
        for (auto &kv : m_contributions)
        {
            if (kv.second > 0.0f)
            {
                remaining.push_back(kv.first);
            }
        }
        while (!remaining.empty())
        {
            float minAmt = std::numeric_limits<float>::max();
            for (auto *p : remaining)
            {
                auto it = std::find_if(m_contributions.begin(), m_contributions.end(),
                                       [p](const auto &pair) { return pair.first == p; });
                minAmt = std::min(minAmt, it->second);
            }
            float potAmount = minAmt * static_cast<float>(remaining.size());
            if (potAmount <= 0.0f)
            {
                break;
            }
            auto tablePots = table.getPots();
            if (tablePots.empty() || tablePots.back().getAmount() > potAmount)
            {
                table.addPot(potAmount, remaining);
            }
            else
            {
                tablePots.back().addAmount(potAmount);
            }
            std::vector<Player *> next;
            next.reserve(remaining.size());
            for (auto *p : remaining)
            {
                auto it = std::find_if(m_contributions.begin(), m_contributions.end(),
                                       [p](const auto &pair) { return pair.first == p; });
                float &contribution = it->second;
                contribution -= minAmt;
                if (contribution > 0.0f)
                {
                    next.push_back(p);
                }
            }
            remaining.swap(next);
        }
        auto community = table.getTableCards();
        for (auto &pot : table.getPots())
        {
            auto winners = getWinners(players, pot, community);
            float share = pot.getAmount() / static_cast<float>(winners.size());
            for (auto *w : winners)
            {
                w->addChips(share);
            }
        }
    }
};

#endif // __POKER_GAME_POT_MANAGER_HPP__
