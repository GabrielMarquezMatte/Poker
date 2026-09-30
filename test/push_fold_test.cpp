#include "../include/cfr/push_fold.hpp"
#include <memory>
#include <gtest/gtest.h>

TEST(PushFold, HoleIndexRoundTrips)
{
    for (std::size_t i = 0; i < holeCombos; ++i)
    {
        ASSERT_EQ(std::popcount(holes[i]), 2);
        ASSERT_EQ(holeIndex(holes[i]), i);
    }
}

// Needs data/preflop_equity.bin (Poker_PushFold creates it); skipped otherwise.
static const PreflopEquity *exactEquity()
{
    static const auto equity = []
    {
        auto e = std::make_unique<PreflopEquity>();
        return e->load(POKER_DATA_DIR "/preflop_equity.bin") ? std::move(e) : nullptr;
    }();
    return equity.get();
}

static std::size_t hole(std::string_view s) { return holeIndex(Deck::parseHand(s).getMask()); }

// Plain nested-loop enumeration, independent of the isomorphism tables.
static double bruteForceEquity(std::string_view hero, std::string_view villain)
{
    const std::uint64_t h = Deck::parseHand(hero).getMask(), v = Deck::parseHand(villain).getMask();
    std::array<std::uint64_t, 48> rest{};
    std::size_t n = 0;
    for (std::size_t c = 0; c < 52; ++c)
    {
        if (((h | v) >> c & 1) == 0)
        {
            rest[n++] = 1ull << c;
        }
    }
    double points = 0.0, boards = 0.0;
    for (std::size_t a = 0; a < 48; ++a)
    {
        for (std::size_t b = a + 1; b < 48; ++b)
        {
            for (std::size_t c = b + 1; c < 48; ++c)
            {
                for (std::size_t d = c + 1; d < 48; ++d)
                {
                    for (std::size_t e = d + 1; e < 48; ++e)
                    {
                        const std::uint64_t board = rest[a] | rest[b] | rest[c] | rest[d] | rest[e];
                        const auto x = Hand::classify(Deck::from_mask(h | board));
                        const auto y = Hand::classify(Deck::from_mask(v | board));
                        points += x > y ? 1.0 : (x == y ? 0.5 : 0.0);
                        boards += 1.0;
                    }
                }
            }
        }
    }
    return points / boards;
}

TEST(PushFold, ExactEquitiesMatchBruteForce)
{
    if (exactEquity() == nullptr)
    {
        GTEST_SKIP() << "run Poker_PushFold " POKER_DATA_DIR "/preflop_equity.bin first";
    }
    const PreflopEquity &e = *exactEquity();
    for (const auto &[hero, villain] : {std::pair{"Ad Ac", "Kh Ks"}, std::pair{"Ks Qh", "Jh Tc"}, std::pair{"7h 6h", "Ah Kd"}, std::pair{"5c 5d", "5h 4h"}})
    {
        EXPECT_NEAR(e(hole(hero), hole(villain)), bruteForceEquity(hero, villain), 1e-6) << hero << " vs " << villain;
    }
    EXPECT_NEAR(e(hole("As Ah"), hole("Kd Kc")), 0.82, 0.01);
    EXPECT_NEAR(e(hole("As Ah"), hole("Ad Ac")), 0.5, 0.01);
    for (std::size_t i = 0; i < holeCombos; i += 37)
    {
        for (std::size_t j = 0; j < holeCombos; j += 41)
        {
            if ((holes[i] & holes[j]) == 0)
            {
                ASSERT_NEAR(e(i, j) + e(j, i), 1.0, 1e-6);
            }
        }
    }
}

TEST(PushFold, NaiveStrategiesAreExploitable)
{
    if (exactEquity() == nullptr)
    {
        GTEST_SKIP();
    }
    PushFoldStrategy always;
    always.push.fill(1.0);
    always.call.fill(1.0);
    EXPECT_GT(pushFoldExploitability(*exactEquity(), always).mbbPerHand(), 1000.0);
}

TEST(PushFold, SolverConvergesToLowExploitability)
{
    if (exactEquity() == nullptr)
    {
        GTEST_SKIP();
    }
    Mccfr<PushFold> solver(1024);
    CfrRng rng{42};
    trainLinear(solver, 100, 20'000, rng);
    const auto e = pushFoldExploitability(*exactEquity(), pushFoldStrategy(solver));
    EXPECT_GE(e.mbbPerHand(), 0.0);
    EXPECT_LT(e.mbbPerHand(), 10.0);
}

TEST(PushFold, SolverWithPruningConvergesToLowExploitability)
{
    if (exactEquity() == nullptr)
    {
        GTEST_SKIP();
    }
    Mccfr<PushFold> solver(1024);
    CfrRng rng{42};
    for (std::uint64_t t = 1; t <= 100; ++t)
    {
        if (t == 10)
        {
            solver.enablePruning(-2000.0f);
        }
        solver.train(20'000, rng);
        solver.discount(static_cast<float>(t) / static_cast<float>(t + 1));
    }
    EXPECT_LT(pushFoldExploitability(*exactEquity(), pushFoldStrategy(solver)).mbbPerHand(), 10.0);
}
