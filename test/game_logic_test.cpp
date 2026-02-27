#include "../include/game/game.hpp"
#include "../include/game/pot_manager.hpp"
#include <numeric>
#include <gtest/gtest.h>

// ========== Compile-time Player tests ==========

static constexpr Player makeConstexprPlayer(std::size_t id, std::uint32_t chips, bool folded, bool hasHole, bool allIn)
{
    Player p{id, chips};
    p.folded = folded;
    p.has_hole = hasHole;
    p.all_in = allIn;
    return p;
}

TEST(PlayerTest, AliveWhenNotFoldedAndHasHole)
{
    static constexpr Player p = makeConstexprPlayer(0, 1000, false, true, false);
    static_assert(p.alive() == true);
}

TEST(PlayerTest, NotAliveWhenFolded)
{
    static constexpr Player p = makeConstexprPlayer(0, 1000, true, true, false);
    static_assert(p.alive() == false);
}

TEST(PlayerTest, NotAliveWhenNoHole)
{
    static constexpr Player p = makeConstexprPlayer(0, 1000, false, false, false);
    static_assert(p.alive() == false);
}

TEST(PlayerTest, AliveWithZeroChips)
{
    static constexpr Player p = makeConstexprPlayer(0, 0, false, true, false);
    static_assert(p.alive() == true);
}

TEST(PlayerTest, EligibleWhenAliveAndNotAllIn)
{
    static constexpr Player p = makeConstexprPlayer(0, 1000, false, true, false);
    static_assert(p.eligible() == true);
}

TEST(PlayerTest, NotEligibleWhenAllIn)
{
    static constexpr Player p = makeConstexprPlayer(0, 1000, false, true, true);
    static_assert(p.eligible() == false);
}

TEST(PlayerTest, NotEligibleWhenFolded)
{
    static constexpr Player p = makeConstexprPlayer(0, 1000, true, true, false);
    static_assert(p.eligible() == false);
}

TEST(PlayerTest, NotEligibleWhenNoHole)
{
    static constexpr Player p = makeConstexprPlayer(0, 1000, false, false, false);
    static_assert(p.eligible() == false);
}

TEST(BlindsTest, SmallBlindValue)
{
    static_assert(Blinds{50, 100}.smallBlind == 50);
}

TEST(BlindsTest, BigBlindValue)
{
    static_assert(Blinds{50, 100}.bigBlind == 100);
}

TEST(BetDataTest, DefaultPotIsZero)
{
    static_assert(BetData{}.pot == 0);
}

TEST(BetDataTest, DefaultCurrentBetIsZero)
{
    static_assert(BetData{}.currentBet == 0);
}

TEST(BetDataTest, DefaultMinRaiseIsZero)
{
    static_assert(BetData{}.minRaise == 0);
}

static inline constexpr int sum_chips(std::span<const Player> players)
{
    return std::accumulate(players.begin(), players.end(), 0, [](int sum, const Player &p)
                           { return sum + static_cast<int>(p.chips); });
}

static inline constexpr std::ptrdiff_t count_alive(std::span<const Player> players)
{
    return std::count_if(players.begin(), players.end(), [](const Player &p)
                         { return p.alive(); });
}

static inline constexpr Game make_game(std::size_t n, std::uint32_t chips_each, const Blinds &b)
{
    Game g(b);
    for (std::size_t i = 0; i < n; ++i)
    {
        g.addPlayer(chips_each);
    }
    return g;
}

template <class TRng>
static inline constexpr void play_all_check_call(Game &g, TRng &rng)
{
    while (g.state() != GameState::Finished)
    {
        const Player &p = g.currentPlayer();
        const BetData &bd = g.betData();
        ActionStruct a{ActionType::Check, 0};
        if (bd.currentBet > p.committed)
        {
            a = {ActionType::Call, 0};
        }
        g.applyAction(rng, a);
    }
}

// ========== Basic Setup Tests ==========

TEST(GameSetup, PostsBlindsCorrectly)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    EXPECT_EQ(g.betData().pot, blinds.smallBlind + blinds.bigBlind);
    EXPECT_EQ(g.betData().currentBet, blinds.bigBlind);
    EXPECT_EQ(g.betData().minRaise, blinds.bigBlind);
    EXPECT_GE(count_alive(g.players()), 2);
}

TEST(GameSetup, DealsExactlyTwoCardsPerPlayer)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{123};
    auto g = make_game(4, 10000, blinds);
    g.startNewHand(rng);

    for (const auto &p : g.players())
    {
        EXPECT_EQ(p.hole.size(), 2) << "Player " << p.id << " should have exactly 2 cards";
    }
}

TEST(GameSetup, ChipsConservedAfterBlinds)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    int total_before = sum_chips(g.players());
    g.startNewHand(rng);
    int total_after = sum_chips(g.players()) + static_cast<int>(g.betData().pot);
    EXPECT_EQ(total_before, total_after);
}

// ========== Fold Logic Tests ==========

TEST(FoldLogic, AllFoldToOneWinner)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    int initial_total = sum_chips(g.players()) + static_cast<int>(g.betData().pot);

    while (g.state() != GameState::Finished && count_alive(g.players()) > 1)
    {
        bool finished = g.applyAction(rng, {ActionType::Fold, 0});
        if (count_alive(g.players()) == 1)
        {
            EXPECT_TRUE(finished) << "Game should be finished when only one player remains";
            break;
        }
    }

    EXPECT_EQ(g.state(), GameState::Finished);
    EXPECT_EQ(count_alive(g.players()), 1);
    
    int final_total = sum_chips(g.players());
    EXPECT_EQ(initial_total, final_total) << "Chips should be conserved";
}

TEST(FoldLogic, FoldReturnsCorrectValue)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(2, 10000, blinds);
    g.startNewHand(rng);

    bool finished = g.applyAction(rng, {ActionType::Fold, 0});
    EXPECT_TRUE(finished) << "Heads-up fold should immediately end the game";
    EXPECT_EQ(g.state(), GameState::Finished);
}

// ========== Check/Call Logic Tests ==========

TEST(CheckCallLogic, CheckWhenNoBet)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    while (g.state() == GameState::PreFlop)
    {
        g.applyAction(rng, {ActionType::Call, 0});
    }

    EXPECT_EQ(g.state(), GameState::Flop);
    
    std::uint32_t pot_before = g.betData().pot;
    g.applyAction(rng, {ActionType::Check, 0});
    EXPECT_EQ(g.betData().pot, pot_before) << "Check should not add to pot";
}

TEST(CheckCallLogic, CallMatchesBet)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    std::uint32_t pot_before = g.betData().pot;
    std::uint32_t call_amount = g.betData().currentBet - g.currentPlayer().committed;
    
    g.applyAction(rng, {ActionType::Call, 0});
    
    EXPECT_EQ(g.betData().pot, pot_before + call_amount);
}

// ========== Bet/Raise Logic Tests ==========

TEST(BetRaiseLogic, BetSetsCurrentBet)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    while (g.state() == GameState::PreFlop)
    {
        g.applyAction(rng, {ActionType::Call, 0});
    }

    EXPECT_EQ(g.betData().currentBet, 0);
    g.applyAction(rng, {ActionType::Bet, 200});
    EXPECT_EQ(g.betData().currentBet, 200);
}

TEST(BetRaiseLogic, RaiseIncreasesCurrentBet)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    std::uint32_t initial_bet = g.betData().currentBet;
    g.applyAction(rng, {ActionType::Raise, 300});
    EXPECT_GT(g.betData().currentBet, initial_bet);
}

TEST(BetRaiseLogic, MinRaiseEnforced)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    g.applyAction(rng, {ActionType::Raise, 1});
    EXPECT_GE(g.betData().currentBet, blinds.bigBlind + blinds.bigBlind);
}

// ========== All-In Logic Tests ==========

TEST(AllInLogic, AllInCommitsAllChips)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(2, 500, blinds);
    g.startNewHand(rng);

    g.applyAction(rng, {ActionType::AllIn, 0});
    
    EXPECT_EQ(g.players()[0].chips + g.players()[1].chips + g.betData().pot, 1000);
}

TEST(AllInLogic, ShortStackAllInDoesNotReopenAction)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.addPlayer(50); // Short stack player
    g.startNewHand(rng);

    while (g.state() == GameState::PreFlop)
    {
        g.applyAction(rng, {ActionType::Call, 0});
    }
    
    EXPECT_EQ(g.state(), GameState::Flop);
}

// ========== Street Progression Tests ==========

TEST(StreetProgression, ProgressesThroughAllStreets)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(2, 10000, blinds);
    g.startNewHand(rng);

    EXPECT_EQ(g.state(), GameState::PreFlop);
    EXPECT_EQ(g.board().size(), 0);

    play_all_check_call(g, rng);

    EXPECT_EQ(g.state(), GameState::Finished);
}

TEST(StreetProgression, FlopDealsThreeCards)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    while (g.state() == GameState::PreFlop)
    {
        g.applyAction(rng, {ActionType::Call, 0});
    }

    EXPECT_EQ(g.state(), GameState::Flop);
    EXPECT_EQ(g.board().size(), 3);
}

TEST(StreetProgression, TurnDealsOneCard)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    while (g.state() != GameState::Turn && g.state() != GameState::Finished)
    {
        const Player &p = g.currentPlayer();
        ActionStruct a = (g.betData().currentBet > p.committed) 
            ? ActionStruct{ActionType::Call, 0} 
            : ActionStruct{ActionType::Check, 0};
        g.applyAction(rng, a);
    }

    if (g.state() == GameState::Turn)
    {
        EXPECT_EQ(g.board().size(), 4);
    }
}

TEST(StreetProgression, RiverDealsOneCard)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    while (g.state() != GameState::River && g.state() != GameState::Finished)
    {
        const Player &p = g.currentPlayer();
        ActionStruct a = (g.betData().currentBet > p.committed) 
            ? ActionStruct{ActionType::Call, 0} 
            : ActionStruct{ActionType::Check, 0};
        g.applyAction(rng, a);
    }

    if (g.state() == GameState::River)
    {
        EXPECT_EQ(g.board().size(), 5);
    }
}

// ========== Chip Conservation Tests ==========

TEST(ChipConservation, ChipsConservedThroughEntireHand)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(4, 10000, blinds);
    
    int initial_total = sum_chips(g.players());
    g.startNewHand(rng);
    
    play_all_check_call(g, rng);
    
    int final_total = sum_chips(g.players());
    EXPECT_EQ(initial_total, final_total) << "Total chips should be conserved";
}

TEST(ChipConservation, ChipsConservedWithBetting)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    
    int initial_total = sum_chips(g.players());
    g.startNewHand(rng);

    int action_count = 0;
    while (g.state() != GameState::Finished && action_count < 100)
    {
        ActionStruct a;
        if (action_count % 5 == 0)
        {
            a = {ActionType::Raise, 200};
        }
        else if (g.betData().currentBet > g.currentPlayer().committed)
        {
            a = {ActionType::Call, 0};
        }
        else
        {
            a = {ActionType::Check, 0};
        }
        g.applyAction(rng, a);
        action_count++;
    }

    int final_total = sum_chips(g.players());
    EXPECT_EQ(initial_total, final_total);
}

// ========== Pot Manager Tests (Compile-time) ==========

static constexpr Player makePlayer(std::size_t id, std::uint32_t chips, std::uint32_t invested, bool folded = false, bool hasHole = true)
{
    Player p;
    p.id = id;
    p.chips = chips;
    p.invested = invested;
    p.folded = folded;
    p.has_hole = hasHole;
    return p;
}

TEST(PotManagerTest, EmptyPlayersReturnsEmptyPots)
{
    static_assert([]() {
        std::vector<Player> players;
        auto pots = PotManager::build(players);
        return pots.empty();
    }());
}

TEST(PotManagerTest, TwoPlayersEqualInvestment)
{
    static_assert([]() {
        std::vector<Player> players = {
            makePlayer(0, 900, 100),
            makePlayer(1, 900, 100),
        };
        auto pots = PotManager::build(players);
        return pots.size() == 1 && pots[0].amount == 200 && pots[0].eligiblePlayers.size() == 2;
    }());
}

TEST(PotManagerTest, AllInCreatesSidePot)
{
    static_assert([]() {
        std::vector<Player> players = {
            makePlayer(0, 0, 50),
            makePlayer(1, 900, 100),
            makePlayer(2, 900, 100),
        };
        players[0].all_in = true;
        auto pots = PotManager::build(players);
        return pots.size() == 2 
            && pots[0].amount == 150 && pots[0].eligiblePlayers.size() == 3
            && pots[1].amount == 100 && pots[1].eligiblePlayers.size() == 2;
    }());
}

TEST(PotManagerTest, FoldedPlayerNotEligible)
{
    static_assert([]() {
        std::vector<Player> players = {
            makePlayer(0, 900, 100, true),
            makePlayer(1, 900, 100),
            makePlayer(2, 900, 100),
        };
        auto pots = PotManager::build(players);
        if (pots.size() != 1 || pots[0].amount != 300 || pots[0].eligiblePlayers.size() != 2)
            return false;
        for (std::size_t idx : pots[0].eligiblePlayers)
            if (idx == 0) return false;
        return true;
    }());
}

TEST(PotManagerTest, TotalEqualsInvestments)
{
    static_assert([]() {
        std::vector<Player> players = {
            makePlayer(0, 0, 150),
            makePlayer(1, 0, 300),
            makePlayer(2, 500, 450),
            makePlayer(3, 200, 200, true),
        };
        players[0].all_in = true;
        players[1].all_in = true;
        auto pots = PotManager::build(players);
        std::uint32_t total = 0;
        for (const auto &pot : pots)
            total += pot.amount;
        return total == 150 + 300 + 450 + 200;
    }());
}

// ========== Edge Cases ==========

TEST(EdgeCases, HeadsUpBlinds)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(2, 10000, blinds);
    g.startNewHand(rng);

    EXPECT_EQ(g.betData().pot, 150);
    EXPECT_EQ(count_alive(g.players()), 2);
}

TEST(EdgeCases, PlayerWithZeroChipsFolds)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    Game g(blinds);
    g.addPlayer(10000);
    g.addPlayer(0);
    g.addPlayer(10000);
    g.startNewHand(rng);

    EXPECT_EQ(count_alive(g.players()), 2);
}

TEST(EdgeCases, MultipleHandsChipsConserved)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);

    int initial_total = sum_chips(g.players());

    for (int hand = 0; hand < 5; ++hand)
    {
        g.startNewHand(rng);
        play_all_check_call(g, rng);
    }

    int final_total = sum_chips(g.players());
    EXPECT_EQ(initial_total, final_total);
}

TEST(EdgeCases, FinishedGameReturnsTrue)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(2, 10000, blinds);
    g.startNewHand(rng);

    g.applyAction(rng, {ActionType::Fold, 0});
    EXPECT_EQ(g.state(), GameState::Finished);

    bool result = g.applyAction(rng, {ActionType::Check, 0});
    EXPECT_TRUE(result);
}

// ========== Bug Fix Tests ==========

TEST(BugFix_CheckReject, CheckFacingBetReturnsFalse)
{
    // Bug: Check silently committed chips when facing a bet.
    // Fixed: returns false immediately without touching game state.
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    // On PreFlop the first actor (UTG) faces the big blind.
    const Player &actor = g.currentPlayer();
    ASSERT_GT(g.betData().currentBet, actor.committed)
        << "Pre-condition: actor must be facing a live bet";

    const std::size_t actor_id   = actor.id;
    const std::uint32_t chips_before = actor.chips;
    const std::uint32_t pot_before   = g.betData().pot;
    const GameState state_before     = g.state();

    bool result = g.applyAction(rng, {ActionType::Check, 0});

    EXPECT_FALSE(result)
        << "Illegal check must return false without advancing the game";
    EXPECT_EQ(g.players()[actor_id].chips, chips_before)
        << "Actor chips must not change on a rejected check";
    EXPECT_EQ(g.betData().pot, pot_before)
        << "Pot must not change on a rejected check";
    EXPECT_EQ(g.state(), state_before)
        << "Game state must not change on a rejected check";
}

TEST(BugFix_CheckReject, CheckAfterCallingIsLegal)
{
    // Sanity check: after everyone calls to even the bet, Check is legal.
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);

    // Advance past PreFlop so all committed amounts are reset.
    while (g.state() == GameState::PreFlop)
    {
        g.applyAction(rng, {ActionType::Call, 0});
    }
    if (g.state() == GameState::Finished)
    {
        GTEST_SKIP() << "Hand finished before Flop";
    }

    // Now currentBet == 0, so Check is legal.
    ASSERT_EQ(g.betData().currentBet, 0u);
    const std::uint32_t pot_before = g.betData().pot;
    g.applyAction(rng, {ActionType::Check, 0});
    EXPECT_EQ(g.betData().pot, pot_before) << "Check must not add to pot";
}

TEST(BugFix_RiverCompletion, LastRiverActionReturnsTrueDirectly)
{
    // Bug: River set state=Showdown and returned false, forcing a spurious
    // extra applyAction call before the showdown ran.
    // Fixed: showdownAndPayout() is invoked directly; last River action returns true.
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(2, 10000, blinds);
    g.startNewHand(rng);

    // Play through PreFlop, Flop, and Turn.
    while (g.state() != GameState::River && g.state() != GameState::Finished)
    {
        const Player &p = g.currentPlayer();
        ActionStruct a = (g.betData().currentBet > p.committed)
            ? ActionStruct{ActionType::Call, 0}
            : ActionStruct{ActionType::Check, 0};
        g.applyAction(rng, a);
    }

    if (g.state() != GameState::River)
    {
        GTEST_SKIP() << "Hand finished before River";
    }

    // Play River actions. The last one must return true in the same call.
    bool finished_on_action = false;
    while (g.state() == GameState::River)
    {
        const Player &p = g.currentPlayer();
        ActionStruct a = (g.betData().currentBet > p.committed)
            ? ActionStruct{ActionType::Call, 0}
            : ActionStruct{ActionType::Check, 0};
        if (g.applyAction(rng, a))
        {
            finished_on_action = true;
            break;
        }
    }

    EXPECT_TRUE(finished_on_action)
        << "The last River action must return true without an extra call";
    EXPECT_EQ(g.state(), GameState::Finished);
    EXPECT_EQ(g.betData().pot, 0u)
        << "showdownAndPayout must clear the pot";
}

// ========== Showdown Payout Tests ==========

TEST(ShowdownPayout, PotIsZeroAfterShowdown)
{
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{7};
    auto g = make_game(3, 10000, blinds);
    g.startNewHand(rng);
    play_all_check_call(g, rng);

    EXPECT_EQ(g.state(), GameState::Finished);
    EXPECT_EQ(g.betData().pot, 0u) << "Pot must be fully distributed after showdown";
}

TEST(ShowdownPayout, TotalChipsPreservedAfterShowdown)
{
    // Chips must be conserved when a real showdown (not a fold) decides the winner.
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{7};
    auto g = make_game(4, 5000, blinds);
    const int initial_total = sum_chips(g.players());
    g.startNewHand(rng);
    play_all_check_call(g, rng);

    EXPECT_EQ(g.state(), GameState::Finished);
    EXPECT_EQ(sum_chips(g.players()), initial_total);
}

TEST(ShowdownPayout, WinnerChipsIncrease)
{
    // At least one player must end up with more chips than they had at hand start.
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(2, 10000, blinds);

    // Snapshot chips right before the hand (players start equal).
    g.startNewHand(rng);
    const std::uint32_t chips_sb = g.players()[0].chips + g.players()[0].committed;
    const std::uint32_t chips_bb = g.players()[1].chips + g.players()[1].committed;

    play_all_check_call(g, rng);

    // After showdown, one (or both in a tie) player has gained chips from the pot.
    const bool someone_gained =
        (g.players()[0].chips > chips_sb) || (g.players()[1].chips > chips_bb);
    EXPECT_TRUE(someone_gained) << "At least one player must gain chips from the pot";
}

// ========== All-In Fast-Forward Tests ==========

TEST(AllInFastForward, BothPlayersAllInCompletesGame)
{
    // When all players go all-in the board must be dealt automatically and the
    // game must reach Finished without any further player decisions.
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{42};
    auto g = make_game(2, 400, blinds);
    const int initial_total = sum_chips(g.players());
    g.startNewHand(rng);

    // Drive applyAction until Finished; the AllIn action is ignored on
    // street-transition calls where current == n.
    while (g.state() != GameState::Finished)
    {
        g.applyAction(rng, {ActionType::AllIn, 0});
    }

    EXPECT_EQ(g.state(), GameState::Finished);
    EXPECT_EQ(g.betData().pot, 0u);
    EXPECT_EQ(sum_chips(g.players()), initial_total);
}

TEST(AllInFastForward, ShortStackForcedAllInSidePotChipsConserved)
{
    // A player too short to cover the big blind is forced all-in on the blind.
    // This creates a side pot between the two full-stacked players.
    // Verifies that the multi-pot payout conserves total chips end-to-end.
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{13};
    Game g(Blinds{50, 100});
    g.addPlayer(10000);
    g.addPlayer(10000);
    g.addPlayer(60); // covers only part of the big blind — forced all-in
    const int initial_total = sum_chips(g.players());
    g.startNewHand(rng);
    play_all_check_call(g, rng);

    EXPECT_EQ(g.state(), GameState::Finished);
    EXPECT_EQ(g.betData().pot, 0u);
    EXPECT_EQ(sum_chips(g.players()), initial_total);
}

TEST(AllInFastForward, ThreeWayAllInChipsConserved)
{
    // All three players shove on PreFlop — no player actions on later streets.
    constexpr Blinds blinds{50, 100};
    omp::XoroShiro128Plus rng{99};
    auto g = make_game(3, 500, blinds);
    const int initial_total = sum_chips(g.players());
    g.startNewHand(rng);

    while (g.state() != GameState::Finished)
    {
        g.applyAction(rng, {ActionType::AllIn, 0});
    }

    EXPECT_EQ(g.state(), GameState::Finished);
    EXPECT_EQ(g.betData().pot, 0u);
    EXPECT_EQ(sum_chips(g.players()), initial_total);
}
