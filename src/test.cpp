#include "../include/card.hpp"
#include "../include/deck.hpp"
#include "../include/classification_result.hpp"
#include "../include/hand.hpp"
#include "../include/random.hpp"
#include "../include/poker_enums.hpp"
#include "../include/game.hpp"
struct ActionStruct
{
    ActionType type;
    int amount = 0;
};
struct Player
{
    int id = 0;
    int chips = 0;
    bool folded = false;
    bool all_in = false;
    int committed = 0;
    Deck hole{};
    bool has_hole = false;
    inline constexpr bool eligible() const noexcept { return !folded && !all_in; }
    inline constexpr bool alive() const noexcept { return !folded; } // can be all-in but still in the hand
};
struct Blinds
{
    int smallBlind = 50;
    int bigBlind = 100;
};
struct BetData
{
    int pot = 0;
    int currentBet = 0;
    int minRaise = 0;
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
        for (std::size_t k = 1; k <= numberOfPlayers(); ++k)
        {
            std::size_t idx = (i + k) % numberOfPlayers();
            if (m_players[idx].eligible())
            {
                return idx;
            }
        }
        return numberOfPlayers(); // none
    }

    inline constexpr std::size_t nextAliveFrom(std::size_t i) const noexcept
    {
        for (std::size_t k = 1; k <= numberOfPlayers(); ++k)
        {
            std::size_t idx = (i + k) % numberOfPlayers();
            if (m_players[idx].alive())
            {
                return idx;
            }
        }
        return numberOfPlayers(); // none
    }

    inline constexpr void commit(Player &player, int amount) noexcept
    {
        amount = std::max(0, amount);
        int pay = std::min(amount, player.chips);
        player.chips -= pay;
        player.committed += pay;
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
        ClassificationResult best{};
        bool first = true;
        for (auto const &e : entries)
        {
            if (first || e.res > best)
            {
                best = e.res;
                first = false;
            }
        }
        std::vector<std::size_t> winners;
        winners.reserve(entries.size());
        for (auto const &e : entries)
        {
            if (e.res == best)
            {
                winners.push_back(e.idx);
            }
        }
        int winnerSize = static_cast<int>(winners.size());
        int share = m_betData.pot / std::max(1, winnerSize);
        int rem = m_betData.pot - share * winnerSize;
        for (int wi = 0; wi < winnerSize; ++wi)
        {
            m_players[winners[wi]].chips += share + (wi < rem ? 1 : 0);
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
        // Everyone who could act has responded to the last bet/raise (or all checked)
        // Advance street or showdown.
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
        // Next eligible player
        std::size_t n = numberOfPlayers();
        std::size_t nxt = nextEligibleFrom(m_playersData.current == n ? m_playersData.dealer : m_playersData.current);
        m_playersData.current = nxt;
        // If no eligible player (all folded or all-in), close betting round
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
        // advance
        nextTurn(rng);
        return bettingRoundMaybeComplete(rng) && m_state == GameState::Finished;
    }

public:
    constexpr Game(Blinds blinds) noexcept : m_blinds(blinds), m_playersData{numberOfPlayers()} {}
    inline constexpr Player &addPlayer(int id, int chips) noexcept
    {
        return m_players.emplace_back(id, chips);
    }
    template <typename TRng>
    inline constexpr void startNewHand(TRng &rng) noexcept
    {
        // Reset per-hand state
        m_board = Deck::emptyDeck();
        m_betData.pot = 0;
        m_state = GameState::PreDeal;
        m_deck = Deck::createFullDeck();
        for (auto &p : m_players)
        {
            p.folded = false;
            p.all_in = false;
            p.committed = 0;
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

        // Post blinds (skip folded/broke seats)
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
    inline constexpr const Player &currentPlayer() const noexcept { return m_players[m_playersData.current]; }
    inline constexpr const BetData &betData() const noexcept { return m_betData; }
    inline constexpr const Deck &board() const noexcept { return m_board; }
    inline constexpr std::span<const Player> players() const noexcept { return m_players; }
    template <typename TRng>
    constexpr bool applyAction(TRng &rng, const ActionStruct &a) noexcept
    {
        // If game is already done, just say so
        if (m_state == GameState::Finished)
        {
            return true;
        }

        // If we're at showdown, resolve and finish
        if (m_state == GameState::Showdown)
        {
            showdownAndPayout();
            return true;
        }

        // Sanity: ensure someone is to act. If not, compute nextTurn() once.
        Player &current = m_players[m_playersData.current];
        if (m_playersData.current == numberOfPlayers() || !current.eligible())
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
        // Helper lambdas
        auto amount_to_call = [&]() -> int
        { return std::max(0, m_betData.currentBet - current.committed); };
        auto can_check = [&]() -> bool
        { return amount_to_call() == 0; };

        switch (a.type)
        {
        case ActionType::Fold:
        {
            current.folded = true;
            // Folding is a valid response to a bet/raise or to a zero-bet round.
            return onlyOneAliveWins() && advanceAndCheckComplete(rng);
        }

        case ActionType::Check:
        {
            if (can_check())
            {
                // Valid check
                return advanceAndCheckComplete(rng);
            }
            int need = amount_to_call();
            if (need <= 0)
            {
                return advanceAndCheckComplete(rng);
            }
            commit(current, need);
            if (current.all_in)
            { /* no side pots here */
            }
            return advanceAndCheckComplete(rng);
        }

        case ActionType::Call:
        {
            int need = amount_to_call();
            if (need == 0)
            {
                // Calling zero == check
                return advanceAndCheckComplete(rng);
            }
            commit(current, need);
            if (current.all_in)
            { /* no side pots here */
            }
            return advanceAndCheckComplete(rng);
        }

        case ActionType::Bet:
        {
            // Allowed only when current_bet == 0
            int amt = std::max(a.amount, m_betData.minRaise);
            if (m_betData.currentBet != 0)
            {
                // Treat as raise if there is an existing bet
                // fallthrough via Raise semantics below
                int target = std::max(m_betData.currentBet + m_betData.minRaise, a.amount);
                int add = std::max(0, target - current.committed);
                commit(current, add);
                if (current.all_in)
                { /* no side pots here */
                }
                int raise_size = std::max(0, target - m_betData.currentBet);
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

            int target = amt; // first bet sets current_bet to bet size
            int add = std::max(0, target - current.committed);
            commit(current, add);
            if (current.all_in)
            { /* no side pots here */
            }
            m_betData.currentBet = target;
            m_betData.minRaise = amt; // first bet defines min_raise going forward
            m_playersData.lastAggressor = m_playersData.current;
            m_playersData.toAct = countEligibleExcluding(m_playersData.current);
            nextTurn(rng);
            return false;
        }
        case ActionType::Raise:
        {
            // Require current_bet > 0
            int target = std::max(m_betData.currentBet + m_betData.minRaise, a.amount);
            int add = std::max(0, target - current.committed);
            commit(current, add);
            if (current.all_in)
            { /* no side pots here */
            }
            int raise_size = std::max(0, target - m_betData.currentBet);
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
            // Treat as bet/raise/call depending on committed + chips
            int target = current.committed + current.chips; // push all chips
            int add = std::max(0, target - current.committed);
            commit(current, add);
            // Update bet/raise bookkeeping if this actually exceeds current_bet
            if (target <= m_betData.currentBet)
            {
                // For now, treat as Call if possible.
                return advanceAndCheckComplete(rng);
            }
            int raise_size = target - m_betData.currentBet;
            m_betData.currentBet = target;
            // All-in can be for less than min-raise; allow but don't update min_raise unless it meets it
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

#include <iostream>

int main()
{
    omp::XoroShiro128Plus rng(std::random_device{}());
    Blinds blinds{50, 100};
    Game g(blinds);
    sizeof(rng);
    g.addPlayer(0, 10000);
    g.addPlayer(1, 10000);
    g.addPlayer(2, 10000);
    g.startNewHand(rng);
    BS::thread_pool<BS::tp::none> threadPool{8};
    while (g.state() != GameState::Finished)
    {
        const Player &p = g.currentPlayer();
        const BetData &bd = g.betData();
        const Deck &board = g.board();
        const GameState state = g.state();
        if (state == GameState::Flop)
        {
            std::cerr << "Player " << p.id << " with hand " << p.hole << "is on the flop.\n";
        }
        std::cerr << "Street: " << static_cast<int>(g.state()) << ", Player "
                  << p.id << " to act. Chips: " << p.chips << ", Committed: "
                  << p.committed << ", Pot: " << bd.pot << ", Current Bet: " 
                  << bd.currentBet << ", Min Raise: " << bd.minRaise 
                  << ", Probability of Winning: " << probabilityOfWinning(p.hole, board, 100'000, 3, threadPool) << '\n';
        ActionStruct a{ActionType::Check, 0};
        if (bd.currentBet > p.committed)
        {
            a = {ActionType::Call, 0};
        }
        if (g.applyAction(rng, a))
        {
            break;
        }
    }
    for (const Player &p : g.players())
    {
        std::cerr << "Player " << p.id << ": Chips: " << p.chips << ", Committed: " << p.committed << "\n";
    }
    return 0;
}