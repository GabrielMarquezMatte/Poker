#ifndef __POKER_GAME_EXECUTION_HPP__
#define __POKER_GAME_EXECUTION_HPP__
#include "player.hpp"
#include "blinds.hpp"
#include "poker_enums.hpp"
#include "pot_manager.hpp"
#include "../classification_result.hpp"
#include "../hand.hpp"
#include <span>
#include <cstdlib>
struct BetData
{
    std::uint32_t pot = 0;
    std::uint32_t currentBet = 0;
    std::uint32_t minRaise = 0;
};
struct PlayersData
{
    std::size_t dealer = 0;
    std::size_t current = 0;
    std::size_t lastAggressor = 0;
    std::size_t toAct = 0;
    constexpr PlayersData() = default;
    constexpr PlayersData(std::size_t numberOfPlayers) : dealer(0), current(numberOfPlayers), lastAggressor(numberOfPlayers), toAct(0) {}
};
class Game
{
private:
    Blinds m_blinds;
    GameState m_state = GameState::PreDeal;
    PlayersData m_playersData;
    BetData m_betData;
    Deck m_board = Deck::emptyDeck();
    std::vector<Player> m_players;
    Deck m_deck = Deck::createFullDeck();
    inline constexpr std::size_t numberOfPlayers() const noexcept { return m_players.size(); }
    inline constexpr std::size_t leftOf(std::size_t i) const noexcept { return (i + 1) % numberOfPlayers(); }
    inline constexpr std::size_t countEligible() const noexcept
    {
        return std::count_if(m_players.begin(), m_players.end(), [](Player const &p)
                             { return p.eligible(); });
    }

    inline constexpr std::size_t countAlive() const noexcept
    {
        return std::count_if(m_players.begin(), m_players.end(), [](Player const &p)
                             { return p.alive(); });
    }

    inline constexpr std::size_t countEligibleExcluding(std::size_t idx) const noexcept
    {
        return std::count_if(m_players.begin(), m_players.end(), [idx](Player const &p)
                             { return p.eligible() && p.id != idx; });
    }

    template <typename Pred>
    inline constexpr std::size_t nextPlayerFrom(std::size_t i, Pred pred) const noexcept
    {
        const std::size_t n = numberOfPlayers();
        for (std::size_t k = 1; k <= n; ++k)
        {
            std::size_t idx = (i + k) % n;
            if (pred(m_players[idx])) return idx;
        }
        return n;
    }

    inline constexpr std::size_t nextEligibleFrom(std::size_t i) const noexcept
    {
        return nextPlayerFrom(i, [](const Player &p) noexcept { return p.eligible(); });
    }

    inline constexpr std::size_t nextAliveFrom(std::size_t i) const noexcept
    {
        return nextPlayerFrom(i, [](const Player &p) noexcept { return p.alive(); });
    }

    inline constexpr void commit(Player &player, std::uint32_t amount) noexcept
    {
        std::uint32_t pay = std::min(amount, player.chips);
        player.chips -= pay;
        player.committed += pay;
        player.invested += pay;
        m_betData.pot += pay;
        if (player.chips == 0)
        {
            player.all_in = true;
        }
    }

    inline constexpr void resetBettingRound() noexcept
    {
        m_betData.currentBet = 0;
        m_betData.minRaise = m_blinds.bigBlind; // usual convention
        m_playersData.lastAggressor = numberOfPlayers();
        for (auto &p : m_players)
        {
            p.committed = 0;
        }
    }

    inline constexpr bool onlyOneAliveWins() noexcept
    {
        if (countAlive() != 1)
        {
            return false;
        }
        for (auto &player : m_players)
        {
            if (!player.alive())
            {
                continue;
            }
            player.chips += m_betData.pot;
            m_betData.pot = 0;
            m_state = GameState::Finished;
            return true;
        }
        return false;
    }

    template <typename TRng>
    inline constexpr void dealBoard(TRng &rng, std::size_t count) noexcept
    {
        m_board.addCards(m_deck.popRandomCards(rng, count));
    }
    template <typename TRng>
    inline constexpr void executeRound(TRng &rng, GameState newState, std::size_t cardsToDeal) noexcept
    {
        m_state = newState;
        dealBoard(rng, cardsToDeal);
        resetBettingRound();
        m_playersData.current = nextEligibleFrom(m_playersData.dealer);
        m_playersData.toAct = countEligible();
    }
    template <typename TRng>
    inline constexpr void advanceStreet(TRng &rng) noexcept
    {
        if (onlyOneAliveWins())
        {
            return;
        }
        switch (m_state)
        {
        case GameState::PreFlop:
            executeRound(rng, GameState::Flop, 3);
            break;
        case GameState::Flop:
            executeRound(rng, GameState::Turn, 1);
            break;
        case GameState::Turn:
            executeRound(rng, GameState::River, 1);
            break;
        default:
            break;
        }
    }
    inline constexpr void showdownAndPayout() noexcept
    {
        const std::size_t n = numberOfPlayers();
        // Folded players keep the zero result, so refund pots (only folded contributors) split evenly.
        std::vector<ClassificationResult> hands(n, ClassificationResult{});
        for (std::size_t i = 0; i < n; ++i)
        {
            if (m_players[i].alive())
            {
                hands[i] = Hand::classify(Deck::createDeck({m_players[i].hole, m_board}));
            }
        }

        for (auto const &pot : PotManager::build(m_players))
        {
            ClassificationResult best{};
            for (std::size_t pi : pot.eligiblePlayers)
            {
                best = std::max(best, hands[pi]);
            }
            std::vector<std::size_t> winners;
            for (std::size_t pi : pot.eligiblePlayers)
            {
                if (hands[pi] == best)
                {
                    winners.push_back(pi);
                }
            }
            std::uint32_t share = pot.amount / static_cast<std::uint32_t>(winners.size());
            std::uint32_t rem = pot.amount % static_cast<std::uint32_t>(winners.size());
            for (std::size_t wi = 0; wi < winners.size(); ++wi)
            {
                m_players[winners[wi]].chips += share + (wi < rem ? 1 : 0);
            }
        }
        m_betData.pot = 0;
        m_state = GameState::Finished;
    }

    template <typename TRng>
    inline constexpr bool bettingRoundMaybeComplete(TRng &rng) noexcept
    {
        if (m_playersData.toAct > 0)
        {
            return false;
        }
        switch (m_state)
        {
        case GameState::PreFlop:
        case GameState::Flop:
        case GameState::Turn:
            advanceStreet(rng);
            return m_state == GameState::Finished;
        case GameState::River:
            showdownAndPayout();
            return true;
        default:
            return false;
        }
    }

    template <typename TRng>
    inline constexpr void nextTurn(TRng &rng) noexcept
    {
        if (m_state == GameState::Finished)
        {
            return;
        }
        if (m_state == GameState::Showdown)
        {
            showdownAndPayout();
            return;
        }
        std::size_t n = numberOfPlayers();
        std::size_t nxt = nextEligibleFrom(m_playersData.current == n ? m_playersData.dealer : m_playersData.current);
        m_playersData.current = nxt;
        if (nxt == n)
        {
            m_playersData.toAct = 0;
            bettingRoundMaybeComplete(rng);
        }
    }

    template <typename TRng>
    inline constexpr bool advanceAndCheckComplete(TRng &rng) noexcept
    {
        if (m_playersData.toAct > 0)
        {
            --m_playersData.toAct;
        }
        nextTurn(rng);
        if (m_state == GameState::Finished)
        {
            return true;
        }
        return bettingRoundMaybeComplete(rng) && m_state == GameState::Finished;
    }

public:
    constexpr Game(Blinds blinds) noexcept : m_blinds(blinds) {}
    inline constexpr Player &addPlayer(std::uint32_t chips) noexcept
    {
        return m_players.emplace_back(m_players.size(), chips);
    }
    template <typename TRng>
    inline constexpr void startNewHand(TRng &rng) noexcept
    {
        const bool handPlayed = m_state != GameState::PreDeal;
        m_board = Deck::emptyDeck();
        m_betData.pot = 0;
        m_state = GameState::PreDeal;
        m_deck = Deck::createFullDeck();
        for (auto &p : m_players)
        {
            p.folded = false;
            p.all_in = false;
            p.committed = 0;
            p.invested = 0;
            p.has_hole = false;
        }
        for (std::size_t i = 0; i < numberOfPlayers(); ++i)
        {
            if (m_players[i].chips <= 0)
            {
                m_players[i].folded = true;
                continue;
            }
            m_players[i].hole = m_deck.popRandomCards(rng, 2);
            m_players[i].has_hole = true;
        }

        if (handPlayed || !m_players[m_playersData.dealer].alive())
        {
            m_playersData.dealer = nextAliveFrom(m_playersData.dealer);
        }
        // Heads-up the dealer posts the small blind.
        const std::size_t sb = countAlive() == 2 ? m_playersData.dealer : nextAliveFrom(m_playersData.dealer);
        const std::size_t bb = nextAliveFrom(sb);
        commit(m_players[sb], m_blinds.smallBlind);
        commit(m_players[bb], m_blinds.bigBlind);
        m_betData.currentBet = std::min(m_players[bb].committed, m_blinds.bigBlind);
        m_betData.minRaise = m_blinds.bigBlind;
        m_playersData.lastAggressor = bb;
        m_playersData.current = nextEligibleFrom(bb);
        m_playersData.toAct = countEligibleExcluding(bb);
        m_state = GameState::PreFlop;
        onlyOneAliveWins();
    }
    inline constexpr GameState state() const noexcept { return m_state; }
    inline constexpr bool hasCurrentActor() const noexcept { return m_playersData.current != numberOfPlayers(); }
    inline constexpr const Player &currentPlayer() const noexcept { return m_players[m_playersData.current]; }
    inline constexpr const BetData &betData() const noexcept { return m_betData; }
    inline constexpr const Deck &board() const noexcept { return m_board; }
    inline constexpr std::span<const Player> players() const noexcept { return m_players; }
    inline constexpr std::size_t dealer() const noexcept { return m_playersData.dealer; }
    inline constexpr std::span<Player> mutablePlayers() noexcept { return m_players; }
    inline constexpr void resetPlayerChips(std::uint32_t chips) noexcept
    {
        for (auto &p : m_players)
        {
            p.chips = chips;
        }
    }
    template <typename TRng>
    constexpr bool applyAction(TRng &rng, const ActionStruct &a) noexcept
    {
        if (m_state == GameState::Finished)
        {
            return true;
        }

        if (m_state == GameState::Showdown)
        {
            showdownAndPayout();
            return true;
        }
        while (m_playersData.current == numberOfPlayers() || !m_players[m_playersData.current].eligible())
        {
            nextTurn(rng);
            if (m_state == GameState::Finished)
            {
                return true;
            }
            if (m_playersData.current == numberOfPlayers())
            {
                return false;
            }
        }
        Player &current = m_players[m_playersData.current];
        const std::uint32_t toCall = m_betData.currentBet - current.committed;

        switch (a.type)
        {
        case ActionType::Fold:
        {
            current.folded = true;
            return onlyOneAliveWins() || advanceAndCheckComplete(rng);
        }

        case ActionType::Check:
        {
            if (toCall != 0)
            {
                return false; // illegal: cannot check when facing a bet
            }
            return advanceAndCheckComplete(rng);
        }

        case ActionType::Call:
        {
            commit(current, toCall);
            return advanceAndCheckComplete(rng);
        }

        // With currentBet == 0 a bet and a raise follow the same rules.
        case ActionType::Bet:
        case ActionType::Raise:
        case ActionType::AllIn:
        {
            const std::uint32_t stack = current.committed + current.chips;
            const std::uint32_t target = a.type == ActionType::AllIn
                                             ? stack
                                             : std::min(std::max(m_betData.currentBet + m_betData.minRaise, a.amount), stack);
            commit(current, target - current.committed);
            if (target <= m_betData.currentBet)
            {
                return advanceAndCheckComplete(rng); // all-in for no more than a call
            }
            const std::uint32_t raiseSize = target - m_betData.currentBet;
            m_betData.currentBet = target;
            if (raiseSize >= m_betData.minRaise)
            {
                m_betData.minRaise = raiseSize; // a short all-in does not lower the min raise
            }
            m_playersData.lastAggressor = m_playersData.current;
            m_playersData.toAct = countEligibleExcluding(m_playersData.current);
            nextTurn(rng);
            return m_state == GameState::Finished || (bettingRoundMaybeComplete(rng) && m_state == GameState::Finished);
        }
        }
        return (m_state == GameState::Finished);
    }
};
#endif // __POKER_GAME_HPP__