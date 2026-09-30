#include "../include/cfr/hunl.hpp"
#include <cmath>
#include <memory>
#include <gtest/gtest.h>

using Nl = Hunl<Hunl100bbConfig>;
static_assert(CfrGame<Nl>);

static std::uint64_t cards(std::string_view s) { return Deck::parseHand(s).getMask(); }

// Fixed runout 4s 5s 6h | Tc | Jd, disjoint from every hole used below.
template <typename G = Nl>
static typename G::State dealt(std::string_view sb, std::string_view bb)
{
    return G::deal(G::initial(), cards(sb), cards(bb), cards("4s 5s 6h"), cards("Tc"), cards("Jd"));
}

// Index of the action whose target investment is `to`.
template <typename G>
static std::size_t actionTo(const typename G::State &s, std::uint32_t to)
{
    const auto legal = G::legalActions(s);
    for (std::size_t a = 0; a < legal.size; ++a)
    {
        if (legal.to[a] == to)
        {
            return a;
        }
    }
    ADD_FAILURE() << "no action to " << to;
    return 0;
}

TEST(Hunl, SmallBlindOpensWithFoldCallRaisesAllIn)
{
    const auto legal = Nl::legalActions(dealt("As Ad", "7c 2d"));
    // Pot after call is 4: raise to 2 + 0.5/1/2 * 4.
    ASSERT_EQ(legal.size, 6u);
    EXPECT_EQ(legal.to[0], Nl::fold);
    EXPECT_EQ(legal.to[1], 2u);
    EXPECT_EQ(legal.to[2], 4u);
    EXPECT_EQ(legal.to[3], 6u);
    EXPECT_EQ(legal.to[4], 10u);
    EXPECT_EQ(legal.to[5], 200u);
}

TEST(Hunl, DropsRaisesOverStackAndCapsRaisesPerStreet)
{
    auto s = dealt("As Ad", "7c 2d");
    s = Nl::apply(s, actionTo<Nl>(s, 10));
    s = Nl::apply(s, actionTo<Nl>(s, 50)); // 2 pot: 10 + 2 * 20
    const auto legal = Nl::legalActions(s);
    // Pot after call 100: 0.5 pot -> 100, 1 pot -> 150, 2 pot -> 250 (over stack, dropped), all-in.
    ASSERT_EQ(legal.size, 5u);
    EXPECT_EQ(legal.to[2], 100u);
    EXPECT_EQ(legal.to[3], 150u);
    EXPECT_EQ(legal.to[4], 200u);
    s = Nl::apply(s, actionTo<Nl>(s, 150)); // third raise hits the cap
    EXPECT_EQ(Nl::numActions(s), 2u);
}

TEST(Hunl, SmallBlindFoldLosesBlind)
{
    const auto s = Nl::apply(dealt("As Ad", "7c 2d"), 0);
    ASSERT_TRUE(Nl::isTerminal(s));
    EXPECT_EQ(Nl::utility(s, 0), -1.0);
    EXPECT_EQ(Nl::utility(s, 1), 1.0);
}

TEST(Hunl, LimpCheckDealsFlopWithBigBlindFirst)
{
    auto s = Nl::apply(dealt("As Ad", "7c 2d"), actionTo<Nl>(dealt("As Ad", "7c 2d"), 2));
    EXPECT_EQ(Nl::currentPlayer(s), 1u);
    EXPECT_FALSE(Nl::isChance(s));
    s = Nl::apply(s, actionTo<Nl>(s, 2)); // check
    ASSERT_TRUE(Nl::isChance(s));
    CfrRng rng{1};
    s = Nl::sampleChance(s, rng);
    EXPECT_EQ(s.board, cards("4s 5s 6h"));
    EXPECT_EQ(Nl::currentPlayer(s), 1u);
    EXPECT_FALSE(Nl::isChance(s));
    EXPECT_FALSE(Nl::isTerminal(s));
}

TEST(Hunl, AllInCallRunsOutToShowdown)
{
    auto s = dealt("As Ad", "7c 2d");
    s = Nl::apply(s, actionTo<Nl>(s, 200));
    s = Nl::apply(s, actionTo<Nl>(s, 200));
    CfrRng rng{7};
    while (!Nl::isTerminal(s))
    {
        ASSERT_TRUE(Nl::isChance(s));
        s = Nl::sampleChance(s, rng);
    }
    EXPECT_EQ(s.board, cards("4s 5s 6h Tc Jd"));
    EXPECT_EQ(Nl::utility(s, 0), 200.0); // aces hold
    EXPECT_EQ(Nl::utility(s, 1), -200.0);
}

TEST(Hunl, RandomPlayoutsAreZeroSumAndBounded)
{
    CfrRng rng{123};
    for (int game = 0; game < 20'000; ++game)
    {
        auto s = Nl::initial();
        int steps = 0;
        while (!Nl::isTerminal(s))
        {
            ASSERT_LT(++steps, 100);
            if (Nl::isChance(s))
            {
                s = Nl::sampleChance(s, rng);
                continue;
            }
            const std::size_t n = Nl::numActions(s);
            ASSERT_GE(n, 1u);
            ASSERT_LE(n, Nl::maxActions);
            s = Nl::apply(s, rng() % n);
        }
        ASSERT_EQ(std::popcount(s.hole[0] | s.hole[1] | s.runout[0] | s.runout[1] | s.runout[2]), 9);
        const double u0 = Nl::utility(s, 0);
        EXPECT_EQ(u0 + Nl::utility(s, 1), 0.0);
        EXPECT_LE(std::abs(u0), 200.0);
    }
}

TEST(Hunl, TrainsOnFullTree)
{
    Mccfr<Nl> solver(1 << 20);
    CfrRng rng{5};
    solver.train(2'000, rng);
    EXPECT_GT(solver.numInfosets(), 1000u);
}

// 20bb push/fold: small blind shoves or folds, big blind calls or folds.
struct PushFold20bb : Hunl100bbConfig
{
    static constexpr std::uint32_t stack = 40;
    static constexpr std::array<std::array<double, 0>, 4> raiseFractions{};
    static constexpr std::array<std::uint8_t, 4> maxRaises{1, 1, 1, 1};
    static constexpr bool allowLimp = false;
};
using Pf = Hunl<PushFold20bb>;

static const Mccfr<Pf> &trainedPushFold()
{
    static const auto solver = []
    {
        auto s = std::make_unique<Mccfr<Pf>>(1024);
        CfrRng rng{42};
        trainLinear(*s, 100, 20'000, rng);
        return s;
    }();
    return *solver;
}

// Opponent cards never enter the actor's infoset key.
static double pushProb(std::string_view hole)
{
    const auto s = dealt<Pf>(hole, "9h 8h");
    return trainedPushFold().averageStrategy(Pf::infosetKey(s), 2)[1];
}

static double callProb(std::string_view hole)
{
    auto s = dealt<Pf>("9h 8h", hole);
    s = Pf::apply(s, 1);
    return trainedPushFold().averageStrategy(Pf::infosetKey(s), 2)[1];
}

TEST(HunlPushFold, SmallBlindShovesPremiumsFoldsTrash)
{
    ASSERT_EQ(Pf::numActions(Pf::initial()), 2u);
    EXPECT_GT(pushProb("As Ad"), 0.95);
    EXPECT_GT(pushProb("Kh Kd"), 0.95);
    EXPECT_GT(pushProb("As Ks"), 0.95);
    EXPECT_LT(pushProb("7c 2d"), 0.05);
    EXPECT_LT(pushProb("3c 2d"), 0.05);
}

TEST(HunlPushFold, BigBlindCallsPremiumsFoldsTrash)
{
    EXPECT_GT(callProb("As Ad"), 0.95);
    EXPECT_GT(callProb("Kh Kd"), 0.95);
    EXPECT_LT(callProb("7c 2d"), 0.05);
    EXPECT_LT(callProb("3c 2d"), 0.05);
}
