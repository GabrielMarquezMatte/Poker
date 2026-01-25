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
    EXPECT_GE(probability, 0.18);
    EXPECT_LE(probability, 0.22);
}

TEST(ExecutionTests, JHighFlushVsTheField)
{
    double probability = calculateProbability("jh 6h", "qs 8d ts td 9c", 500'000, 8);
    EXPECT_GE(probability, 0.74);
    EXPECT_LE(probability, 0.75);
}