#ifndef __POKER_GAME_EXECUTION_HPP__
#define __POKER_GAME_EXECUTION_HPP__
#include "player.hpp"
#include "blinds.hpp"
#include "poker_enums.hpp"
#include "pot_manager.hpp"
#include "../classification_result.hpp"
#include "../hand.hpp"
#include <span>
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
    constexpr PlayersData(std::size_t numberOfPlayers) : dealer(0), current(numberOfPlayers), lastAggressor(numberOfPlayers), toAct(0) {}
};
class Game
{
private:
    Blinds m_blinds;
    GameState m_state = GameState::PreDeal;
    PlayersData m_playersData;
    BetData m_betData;
    Deck m_board;
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

    inline constexpr std::size_t nextEligibleFrom(std::size_t i) const noexcept
    {
        const std::size_t n = numberOfPlayers();
        for (std::size_t k = 1; k <= n; ++k)
        {
            std::size_t idx = (i + k) % n;
            if (m_players[idx].eligible())
            {
                return idx;
            }
        }
        return n;
    }

    inline constexpr std::size_t nextAliveFrom(std::size_t i) const noexcept
    {
        const std::size_t n = numberOfPlayers();
        for (std::size_t k = 1; k <= n; ++k)
        {
            std::size_t idx = (i + k) % n;
            if (m_players[idx].alive())
            {
                return idx;
            }
        }
        return n;
    }

    inline constexpr void commit(Player &player, std::uint32_t amount) noexcept
    {
        amount = std::max(0u, amount);
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
        case GameState::River:
            m_state = GameState::Showdown;
            break;
        default:
            break;
        }
    }
    inline constexpr void showdownAndPayout() noexcept
    {
        Deck table = m_board;
        struct Entry
        {
            std::size_t idx;
            ClassificationResult res;
        };
        std::vector<Entry> entries;
        entries.reserve(numberOfPlayers());
        for (std::size_t i = 0; i < numberOfPlayers(); ++i)
        {
            auto const &p = m_players[i];
            if (!p.alive())
            {
                continue;
            }
            auto res = Hand::classify(Deck::createDeck({p.hole, table}));
            entries.push_back({i, res});
        }
        auto pots = PotManager::build(m_players);
        for (auto const &pot : pots)
        {
            if (pot.amount <= 0 || pot.eligiblePlayers.empty())
            {
                continue;
            }
            ClassificationResult best{};
            bool first = true;
            for (std::size_t pi : pot.eligiblePlayers)
            {
                auto it = std::find_if(entries.begin(), entries.end(),
                                       [&](auto const &e)
                                       { return e.idx == pi; });
                if (it == entries.end())
                {
                    continue;
                }
                if (first || it->res > best)
                {
                    best = it->res;
                    first = false;
                }
            }
            if (first)
            {
                continue;
            }
            std::vector<std::size_t> winners;
            for (std::size_t pi : pot.eligiblePlayers)
            {
                auto it = std::find_if(entries.begin(), entries.end(),
                                       [&](auto const &e)
                                       { return e.idx == pi; });
                if (it != entries.end() && it->res == best)
                {
                    winners.push_back(pi);
                }
            }
            if (winners.empty())
            {
                continue;
            }
            int winnerCount = static_cast<int>(winners.size());
            int share = pot.amount / winnerCount;
            int rem = pot.amount - share * winnerCount;
            std::sort(winners.begin(), winners.end());
            for (int wi = 0; wi < winnerCount; ++wi)
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
            m_state = GameState::Showdown;
            return false;
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
        return bettingRoundMaybeComplete(rng) && m_state == GameState::Finished;
    }

public:
    constexpr Game(Blinds blinds) noexcept : m_blinds(blinds), m_playersData{numberOfPlayers()} {}
    inline constexpr Player &addPlayer(std::uint32_t chips) noexcept
    {
        return m_players.emplace_back(m_players.size(), chips);
    }
    template <typename TRng>
    inline constexpr void startNewHand(TRng &rng) noexcept
    {
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
        for (int r = 0; r < 2; ++r)
        {
            for (int i = 0; i < numberOfPlayers(); ++i)
            {
                if (m_players[i].chips <= 0)
                {
                    m_players[i].folded = true;
                    continue;
                }
                m_players[i].hole = m_deck.popRandomCards(rng, 2);
                m_players[i].has_hole = true;
            }
        }

        std::size_t sb = nextAliveFrom(m_playersData.dealer);
        std::size_t bb = nextAliveFrom(sb);
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
        if (m_playersData.current == numberOfPlayers())
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
        if (!current.eligible())
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
        if (m_playersData.current == numberOfPlayers())
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
        auto amount_to_call = [&]() -> std::uint32_t
        { return std::max(0u, m_betData.currentBet - current.committed); };
        auto can_check = [&]() -> bool
        { return amount_to_call() == 0; };

        switch (a.type)
        {
        case ActionType::Fold:
        {
            current.folded = true;
            return onlyOneAliveWins() && advanceAndCheckComplete(rng);
        }

        case ActionType::Check:
        {
            if (can_check())
            {
                return advanceAndCheckComplete(rng);
            }
            int need = amount_to_call();
            if (need <= 0)
            {
                return advanceAndCheckComplete(rng);
            }
            commit(current, need);
            return advanceAndCheckComplete(rng);
        }

        case ActionType::Call:
        {
            int need = amount_to_call();
            if (need == 0)
            {
                return advanceAndCheckComplete(rng);
            }
            commit(current, need);
            return advanceAndCheckComplete(rng);
        }

        case ActionType::Bet:
        {
            std::uint32_t amt = std::max(a.amount, m_betData.minRaise);
            if (m_betData.currentBet != 0)
            {
                std::uint32_t target = std::max(m_betData.currentBet + m_betData.minRaise, a.amount);
                std::uint32_t add = std::max(0u, target - current.committed);
                commit(current, add);
                std::uint32_t raise_size = std::max(0u, target - m_betData.currentBet);
                m_betData.currentBet = std::max(m_betData.currentBet, target);
                if (raise_size > 0)
                {
                    m_betData.minRaise = raise_size;
                }
                m_playersData.lastAggressor = m_playersData.current;
                m_playersData.toAct = countEligibleExcluding(m_playersData.current);
                nextTurn(rng);
                return false;
            }

            std::uint32_t target = amt;
            std::uint32_t add = std::max(0u, target - current.committed);
            commit(current, add);
            m_betData.currentBet = target;
            m_betData.minRaise = amt;
            m_playersData.lastAggressor = m_playersData.current;
            m_playersData.toAct = countEligibleExcluding(m_playersData.current);
            nextTurn(rng);
            return false;
        }
        case ActionType::Raise:
        {
            std::uint32_t target = std::max(m_betData.currentBet + m_betData.minRaise, a.amount);
            std::uint32_t add = std::max(0u, target - current.committed);
            commit(current, add);
            std::uint32_t raise_size = std::max(0u, target - m_betData.currentBet);
            m_betData.currentBet = std::max(m_betData.currentBet, target);
            if (raise_size > 0)
            {
                m_betData.minRaise = raise_size;
            }
            m_playersData.lastAggressor = m_playersData.current;
            m_playersData.toAct = countEligibleExcluding(m_playersData.current);
            nextTurn(rng);
            return false;
        }
        case ActionType::AllIn:
        {
            std::uint32_t target = current.committed + current.chips;
            std::uint32_t add = std::max(0u, target - current.committed);
            commit(current, add);
            if (target <= m_betData.currentBet)
            {
                return advanceAndCheckComplete(rng);
            }
            std::uint32_t raise_size = target - m_betData.currentBet;
            m_betData.currentBet = target;
            if (raise_size >= m_betData.minRaise)
            {
                m_betData.minRaise = raise_size;
            }
            m_playersData.lastAggressor = m_playersData.current;
            m_playersData.toAct = countEligibleExcluding(m_playersData.current);
            nextTurn(rng);
            return bettingRoundMaybeComplete(rng) && m_state == GameState::Finished;
        }
        }
        return (m_state == GameState::Finished);
    }
};
#endif // __POKER_GAME_HPP__