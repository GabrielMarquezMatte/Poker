#include "../include/cfr/card_abstraction.hpp"
#include "../include/deck.hpp"
#include <gtest/gtest.h>

// Needs data/abstraction.bin from Poker_Abstraction; skipped otherwise.
static const CardAbstraction *generated()
{
    static const CardAbstraction *abstraction = []() -> const CardAbstraction *
    {
        static CardAbstraction a;
        return a.load(POKER_DATA_DIR "/abstraction.bin") ? &a : nullptr;
    }();
    return abstraction;
}

static std::uint8_t bucket(std::string_view hole, std::string_view board)
{
    return generated()->bucket(Deck::parseHand(hole).getMask(), Deck::parseHand(board).getMask());
}

TEST(CardAbstraction, BucketsOrderedByStrength)
{
    if (generated() == nullptr)
    {
        GTEST_SKIP() << "run Poker_Abstraction " POKER_DATA_DIR "/abstraction.bin first";
    }
    const auto maxBucket = *std::max_element(generated()->buckets[2].begin(), generated()->buckets[2].end());
    // River: nuts in the top bucket; top pair above playing the board.
    EXPECT_EQ(bucket("As Ks", "Qs Js Ts 2h 3d"), maxBucket);
    EXPECT_GT(bucket("Ac 2d", "As Kh Qd 9s 7c"), bucket("3c 2d", "As Kh Qd 9s 7c"));
    // Turn and flop: overpair beats air, set beats overpair.
    EXPECT_GT(bucket("As Ah", "2c 7d Jh 9s"), bucket("4s 3h", "2c 7d Jh 9s"));
    EXPECT_GT(bucket("As Ah", "2c 7d Jh"), bucket("4s 3h", "2c 7d Jh"));
    EXPECT_GT(bucket("Js Jc", "2c 7d Jh"), bucket("As Ah", "2c 7d Jh"));
}

TEST(CardAbstraction, IsomorphicHandsShareBucket)
{
    if (generated() == nullptr)
    {
        GTEST_SKIP();
    }
    EXPECT_EQ(bucket("As Ks", "Qs 7h 2d"), bucket("Ah Kh", "Qh 7s 2c"));
    EXPECT_EQ(bucket("9c 8c", "7c 6d 2h Kd"), bucket("9d 8d", "7d 6s 2c Ks"));
}
