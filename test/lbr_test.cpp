#include "../include/cfr/lbr.hpp"
#include <memory>
#include <gtest/gtest.h>

// Needs data/preflop_equity.bin (Poker_PushFold creates it); skipped otherwise.
static const PreflopEquity *lbrEquity()
{
    static const auto equity = []
    {
        auto e = std::make_unique<PreflopEquity>();
        return e->load(POKER_DATA_DIR "/preflop_equity.bin") ? std::move(e) : nullptr;
    }();
    return equity.get();
}

// In push/fold nothing is bet after the all-in, so LBR's call-down assumption is exact and LBR is
// the exact best response: its winnings must match pushFoldExploitability within sampling error.
TEST(Lbr, MatchesExactBestResponseInPushFold)
{
    if (lbrEquity() == nullptr)
    {
        GTEST_SKIP();
    }
    const Mccfr<PushFold> untrained(1024); // every infoset uniform: push and call half the time
    const auto exact = pushFoldExploitability(*lbrEquity(), pushFoldStrategy(untrained));
    const auto result = LocalBestResponse<PushFold20bb>(untrained, *lbrEquity()).evaluate(200'000, 8, 1);
    const double toMbb = 1000.0 / PushFold20bb::bigBlind;
    EXPECT_NEAR(result.mbbPerHand[0], exact.smallBlindBestResponse * toMbb, 4 * result.standardError[0]);
    EXPECT_NEAR(result.mbbPerHand[1], exact.bigBlindBestResponse * toMbb, 4 * result.standardError[1]);
}

TEST(Lbr, BarelyExploitsTrainedPushFold)
{
    if (lbrEquity() == nullptr)
    {
        GTEST_SKIP();
    }
    Mccfr<PushFold> solver(1024);
    CfrRng rng{42};
    trainLinear(solver, 100, 20'000, rng);
    const auto exact = pushFoldExploitability(*lbrEquity(), pushFoldStrategy(solver));
    const auto result = LocalBestResponse<PushFold20bb>(solver, *lbrEquity()).evaluate(200'000, 8, 2);
    EXPECT_NEAR(result.average(), exact.mbbPerHand(), 4 * result.averageError());
}

TEST(Lbr, SelfPlayWithNoDeviationStreetsBreaksEven)
{
    if (lbrEquity() == nullptr)
    {
        GTEST_SKIP();
    }
    const Mccfr<PushFold> untrained(1024);
    const auto result = LocalBestResponse<PushFold20bb>(untrained, *lbrEquity(), 100, 0).evaluate(200'000, 8, 3);
    EXPECT_NEAR(result.average(), 0.0, 4 * result.averageError());
}
