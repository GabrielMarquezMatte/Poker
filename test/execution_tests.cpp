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
    double probability = calculateProbability("ac ks", "td 9c 9s th 2h", 500'000, 8);
    EXPECT_GE(probability, 0.165);
    EXPECT_LE(probability, 0.185);
}

TEST(ExecutionTests, StraightOnPairedBoardVsTheField) 
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
    EXPECT_EQ(stats.totalGames(), 990u); 
}

TEST(ExactEnumeration, MatchesSimulationMultiway)
{
    const Deck player = Deck::parseHand("jh 6h");
    const Deck board = Deck::parseHand("qs 8h th 2h");
    const GameStatistics exact = exactGameStatistics(player, board, 3);
    EXPECT_EQ(exact.totalGames(), 46u * 990u * 903u); 
    omp::XoroShiro128Plus rng{7};
    const GameStatistics simulated = computeRandomGameStatistics(rng, player, board, 2'000'000, 3);
    const auto ratio = [](std::size_t part, const GameStatistics &s)
    { return static_cast<double>(part) / static_cast<double>(s.totalGames()); };
    EXPECT_NEAR(ratio(exact.wins, exact), ratio(simulated.wins, simulated), 0.002);
    EXPECT_NEAR(ratio(exact.ties, exact), ratio(simulated.ties, simulated), 0.002);
}

TEST(ExactEnumeration, SmallCasesUseExactPath)
{
    const Deck player = Deck::parseHand("ac ks");
    const Deck board = Deck::parseHand("td 9c 9s th 2h");
    const GameStatistics exact = exactGameStatistics(player, board, 2);
    EXPECT_EQ(calculateProbability("ac ks", "td 9c 9s th 2h", 1'000'000, 2),
              static_cast<double>(exact.wins + exact.ties) / static_cast<double>(exact.totalGames()));
}

TEST(PreflopTable, ClassIndexGroupsAll1326StartingHands)
{
    std::array<int, 169> combos{};
    const std::uint64_t full = Deck::createFullDeck().getMask();
    forEachCombination(full, 2, 0, [&](std::uint64_t hole)
                       { ++combos[preflopClassIndex(Deck::from_mask(hole))]; });
    for (std::size_t row = 0; row < 13; ++row)
    {
        for (std::size_t col = 0; col < 13; ++col)
        {
            const int expected = row == col ? 6 : (row > col ? 4 : 12); 
            EXPECT_EQ(combos[row * 13 + col], expected) << "row " << row << " col " << col;
        }
    }
}

TEST(PreflopTable, EntriesMatchFreshSimulation)
{
    const auto check = [](std::string_view hole, std::size_t players)
    {
        const Deck cards = Deck::parseHand(hole);
        const GameStatistics table = preflopStatistics(cards, players);
        ASSERT_GT(table.totalGames(), 0u) << "preflop table not generated";
        omp::XoroShiro128Plus rng{2024};
        const GameStatistics fresh = computeRandomGameStatistics(rng, cards, Deck::emptyDeck(), 1'000'000, players);
        const auto notLosing = [](const GameStatistics &s)
        { return s.notLosing(); };
        EXPECT_NEAR(notLosing(table), notLosing(fresh), 0.003) << hole << " vs " << players - 1 << " opponents";
        EXPECT_NEAR(table.equity(), fresh.equity(), 0.003) << hole << " vs " << players - 1 << " opponents";
    };
    check("as ah", 2);
    check("7c 2d", 6);
    check("js ts", 10);
    check("kh qd", 4);
}

TEST(PreflopTable, ApiPathUsesTable)
{
    const GameStatistics table = preflopStatistics(Deck::parseHand("as ah"), 2);
    ASSERT_GT(table.totalGames(), 0u);
    EXPECT_EQ(calculateProbability("as ah", "", 1'000'000, 2),
              static_cast<double>(table.wins + table.ties) / static_cast<double>(table.totalGames()));
}

TEST(Equity, BoardPlaysSplitsPotEvenly)
{
    const GameStatistics stats = computeRandomGameStatistics(Deck::parseHand("2c 7d"), Deck::parseHand("ts js qs ks as"), 1'000'000, 3, threadPool);
    EXPECT_EQ(stats.ties, stats.totalGames());
    EXPECT_DOUBLE_EQ(stats.notLosing(), 1.0);
    EXPECT_DOUBLE_EQ(stats.equity(), 1.0 / 3.0);
}

TEST(Equity, SimulationMatchesExactMultiway)
{
    const Deck player = Deck::parseHand("jh 6h");
    const Deck board = Deck::parseHand("qs 8h th 2h");
    const GameStatistics exact = exactGameStatistics(player, board, 3);
    omp::XoroShiro128Plus rng{11};
    const GameStatistics simulated = computeRandomGameStatistics(rng, player, board, 2'000'000, 3);
    EXPECT_NEAR(exact.equity(), simulated.equity(), 0.002);
    EXPECT_LE(exact.equity(), exact.notLosing());
}

TEST(ExactEnumeration, MultiwayGoldenCounts)
{
    const auto expect = [](std::string_view board, std::size_t players, std::size_t wins, std::size_t losses, std::size_t ties, std::uint64_t potShares)
    {
        for (const GameStatistics &stats : {exactGameStatistics(Deck::parseHand("jh 6h"), Deck::parseHand(board), players),
                                            exactGameStatistics(Deck::parseHand("jh 6h"), Deck::parseHand(board), players, threadPool)})
        {
            EXPECT_EQ(stats.wins, wins) << board << " " << players;
            EXPECT_EQ(stats.losses, losses) << board << " " << players;
            EXPECT_EQ(stats.ties, ties) << board << " " << players;
            EXPECT_EQ(stats.potShares, potShares) << board << " " << players;
        }
    };
    expect("qs 8h th 2h", 3, 37734288, 3388332, 0, 95090405760);
    expect("qs 8h th 2h 3d", 4, 693439200, 39616200, 0, 1747466784000);
    expect("7h 7c 7d 2s 2c", 4, 0, 212786100, 520269300, 327769659000);
}

TEST(ExactEnumeration, ParallelMatchesSequential)
{
    const Deck player = Deck::parseHand("ah kd");
    const Deck board = Deck::parseHand("kc 7s 2h");
    const GameStatistics sequential = exactGameStatistics(player, board, 2);
    const GameStatistics parallel = exactGameStatistics(player, board, 2, threadPool);
    EXPECT_EQ(sequential.totalGames(), 1081u * 990u); 
    EXPECT_EQ(parallel.wins, sequential.wins);
    EXPECT_EQ(parallel.losses, sequential.losses);
    EXPECT_EQ(parallel.ties, sequential.ties);
    EXPECT_EQ(parallel.potShares, sequential.potShares);
}
