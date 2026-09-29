#include <gtest/gtest.h>
#include <array>
#include "../include/deck.hpp"
#include "../include/hand.hpp"
#include "../include/game.hpp"

static BS::thread_pool<BS::tp::none> threadPool(std::thread::hardware_concurrency());
inline double calculateProbability(const std::string_view playerHand, const std::string_view boardCards, std::size_t numSimulations, std::size_t numPlayers)
{
    Deck player = Deck::parseHand(playerHand);
    Deck board = Deck::parseHand(boardCards);
    return probabilityOfWinning(player, board, numSimulations, numPlayers, threadPool);
}

TEST(ExecutionTests, RoyalFlushTest)
{
    double probability = calculateProbability("as ks", "qs js ts 2h 3d", 500'000, 8);
    EXPECT_EQ(probability, 1.0);
}

TEST(ExecutionTests, RoyalFlushOnTheBoard)
{
    double probability = calculateProbability("2c 7d", "ts js qs ks as", 500'000, 8);
    EXPECT_EQ(probability, 1.0);
}

TEST(ExecutionTests, UnbeatableQuads)
{
    double probability = calculateProbability("as 2c", "ad ah ac kc qd", 500'000, 8);
    EXPECT_EQ(probability, 1.0);
}

TEST(ExecutionTests, HighFlush)
{
    double probability = calculateProbability("as 7h", "ks qs 9s 2s 3d", 500'000, 8);
    EXPECT_GE(probability, 0.99);
    EXPECT_LE(probability, 1.0);
}

TEST(ExecutionTests, QHighFlushVsTheFieldOnA4SpadeBoard)
{
    double probability = calculateProbability("qs 7h", "ks 7s 4s 2s 3d", 200'000, 8);
    EXPECT_GE(probability, 0.60);
    EXPECT_LE(probability, 0.70);
}

TEST(ExecutionTests, TwoPairOnTheBoardKickerWars)
{
    // Opponent overpairs (JJ+) make a higher two pair (e.g. QQ TT 9) and beat TT 99 A.
    double probability = calculateProbability("ac ks", "td 9c 9s th 2h", 500'000, 8);
    EXPECT_GE(probability, 0.165);
    EXPECT_LE(probability, 0.185);
}

TEST(ExecutionTests, StraightOnPairedBoardVsTheField) // Nome corrigido
{
    double probability = calculateProbability("jh 6h", "qs 8d ts td 9c", 500'000, 8);
    EXPECT_GE(probability, 0.74);
    EXPECT_LE(probability, 0.75);
}

TEST(ExecutionTests, JHighFlushVsTheField)
{
    double probability = calculateProbability("jh 6h", "qs 8h th 2h 3d", 500'000, 8);
    EXPECT_GE(probability, 0.86);
    EXPECT_LE(probability, 0.89);
}
TEST(ExactEnumeration, RiverHeadsUpCoversEveryOpponentHand)
{
    const GameStatistics stats = exactGameStatistics(Deck::parseHand("ac ks"), Deck::parseHand("td 9c 9s th 2h"), 2);
    EXPECT_EQ(stats.totalGames(), 990u); // C(45, 2)
}

TEST(ExactEnumeration, MatchesSimulationMultiway)
{
    // 3-way on the turn: exact counts every deal, simulation must agree within sampling error.
    const Deck player = Deck::parseHand("jh 6h");
    const Deck board = Deck::parseHand("qs 8h th 2h");
    const GameStatistics exact = exactGameStatistics(player, board, 3);
    EXPECT_EQ(exact.totalGames(), 46u * 990u * 903u); // river card, then two ordered hole pairs
    omp::XoroShiro128Plus rng{7};
    const GameStatistics simulated = computeRandomGameStatistics(rng, player, board, 2'000'000, 3);
    const auto ratio = [](std::size_t part, const GameStatistics &s)
    { return static_cast<double>(part) / static_cast<double>(s.totalGames()); };
    EXPECT_NEAR(ratio(exact.wins, exact), ratio(simulated.wins, simulated), 0.002);
    EXPECT_NEAR(ratio(exact.ties, exact), ratio(simulated.ties, simulated), 0.002);
}

TEST(ExactEnumeration, SmallCasesUseExactPath)
{
    // River heads-up has 990 deals, far below the simulation budget, so the answer is exact.
    const Deck player = Deck::parseHand("ac ks");
    const Deck board = Deck::parseHand("td 9c 9s th 2h");
    const GameStatistics exact = exactGameStatistics(player, board, 2);
    EXPECT_EQ(calculateProbability("ac ks", "td 9c 9s th 2h", 1'000'000, 2),
              static_cast<double>(exact.wins + exact.ties) / static_cast<double>(exact.totalGames()));
}
