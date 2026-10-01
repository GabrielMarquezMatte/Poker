#include "../include/cfr/mccfr.hpp"
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

// Kuhn poker: 3 cards (J=0, Q=1, K=2), ante 1, one bet of 1. Action 0 = pass (check/fold), 1 = bet (bet/call).
struct Kuhn
{
    static constexpr std::size_t numPlayers = 2;
    static constexpr std::size_t maxActions = 2;
    struct State
    {
        std::array<std::uint8_t, 2> cards{};
        std::uint8_t history = 0; // bit i = action i
        std::uint8_t length = 0;
        bool dealt = false;
    };
    static State initial() { return {}; }
    static bool isChance(const State &s) { return !s.dealt; }
    static State sampleChance(const State &s, CfrRng &rng)
    {
        const std::uint64_t r = rng() % 6;
        State next = s;
        next.cards[0] = static_cast<std::uint8_t>(r / 2);
        next.cards[1] = static_cast<std::uint8_t>((next.cards[0] + 1 + r % 2) % 3);
        next.dealt = true;
        return next;
    }
    static bool isTerminal(const State &s) { return s.length == 3 || (s.length == 2 && s.history != 0b10); }
    static double utility(const State &s, std::size_t player)
    {
        const double showdown = s.cards[0] > s.cards[1] ? 1.0 : -1.0;
        double u = 0.0;
        if (s.length == 3)
        {
            u = (s.history & 0b100) ? 2.0 * showdown : -1.0; // pb-call or pb-fold
        }
        else if (s.history == 0b00)
        {
            u = showdown;
        }
        else if (s.history == 0b01)
        {
            u = 1.0; // bet, fold
        }
        else
        {
            u = 2.0 * showdown; // bet, call
        }
        return player == 0 ? u : -u;
    }
    static std::size_t currentPlayer(const State &s) { return s.length % 2; }
    static std::size_t numActions(const State &) { return 2; }
    static State apply(const State &s, std::size_t action)
    {
        State next = s;
        next.history = static_cast<std::uint8_t>(next.history | (action << next.length));
        ++next.length;
        return next;
    }
    static std::uint64_t infosetKey(const State &s)
    {
        return s.cards[currentPlayer(s)] | (std::uint64_t{s.length} << 2) | (std::uint64_t{s.history} << 4);
    }
};
static_assert(CfrGame<Kuhn>);

static const Mccfr<Kuhn> &trainedKuhn()
{
    static const auto solver = []
    {
        auto s = std::make_unique<Mccfr<Kuhn>>(64);
        CfrRng rng{42};
        trainLinear(*s, 100, 3'000, rng);
        return s;
    }();
    return *solver;
}

// Probability of action 1 (bet/call) holding `card` after `history` of `length` actions.
static double betProb(std::uint8_t card, std::uint8_t history, std::uint8_t length)
{
    const Kuhn::State s{{card, card}, history, length, true};
    return trainedKuhn().averageStrategy(Kuhn::infosetKey(s), 2)[1];
}

static double expectedValue(const Mccfr<Kuhn> &solver, const Kuhn::State &s)
{
    if (Kuhn::isTerminal(s))
    {
        return Kuhn::utility(s, 0);
    }
    const auto sigma = solver.averageStrategy(Kuhn::infosetKey(s), 2);
    return sigma[0] * expectedValue(solver, Kuhn::apply(s, 0)) + sigma[1] * expectedValue(solver, Kuhn::apply(s, 1));
}

// Player 0's value under both average strategies, exact over the 6 deals.
static double gameValue(const Mccfr<Kuhn> &solver)
{
    double total = 0.0;
    for (std::uint8_t c0 = 0; c0 < 3; ++c0)
    {
        for (std::uint8_t c1 = 0; c1 < 3; ++c1)
        {
            if (c0 != c1)
            {
                total += expectedValue(solver, Kuhn::State{{c0, c1}, 0, 0, true});
            }
        }
    }
    return total / 6.0;
}

TEST(MccfrKuhn, VisitsAllTwelveInfosets)
{
    EXPECT_EQ(trainedKuhn().numInfosets(), 12u);
}

TEST(MccfrKuhn, ConvergesToGameValue)
{
    EXPECT_NEAR(gameValue(trainedKuhn()), -1.0 / 18.0, 5e-3);
}

TEST(MccfrKuhn, ConcurrentTrainingConverges)
{
    Mccfr<Kuhn> solver(64);
    for (std::uint64_t t = 1; t <= 100; ++t)
    {
        std::vector<std::jthread> threads;
        for (std::uint64_t i = 0; i < 4; ++i)
        {
            threads.emplace_back([&solver, seed = t * 4 + i]
                                 {
                CfrRng rng{seed};
                solver.train(750, rng); });
        }
        threads.clear();
        solver.discount(static_cast<float>(t) / static_cast<float>(t + 1));
    }
    EXPECT_EQ(solver.iterations(), 300'000u);
    EXPECT_NEAR(gameValue(solver), -1.0 / 18.0, 5e-3);
}

TEST(MccfrKuhn, SaveLoadRoundTrips)
{
    const std::string path = (std::filesystem::temp_directory_path() / "mccfr_kuhn_test.bin").string();
    ASSERT_TRUE(trainedKuhn().save(path));
    Mccfr<Kuhn> loaded(64);
    ASSERT_TRUE(loaded.load(path));
    std::filesystem::remove(path);
    EXPECT_EQ(loaded.numInfosets(), 12u);
    EXPECT_EQ(loaded.iterations(), trainedKuhn().iterations());
    EXPECT_EQ(gameValue(loaded), gameValue(trainedKuhn()));
}

// Kuhn keeping the average strategy for the opening decision only.
struct KuhnOpeningAverage : Kuhn
{
    static bool averaged(const State &s) { return s.length == 0; }
    static std::vector<std::uint64_t> averagedKeys() { return {0, 1, 2}; } // infosetKey of each card, no history
};
static_assert(PartlyAveraged<KuhnOpeningAverage> && !PartlyAveraged<Kuhn>);

// Averages do not feed back into regrets: the kept ones match a fully averaged run of the same seed,
// and only they are stored.
TEST(MccfrKuhn, PartlyAveragedKeepsOnlyThoseAverages)
{
    Mccfr<KuhnOpeningAverage> solver(64);
    CfrRng rng{42};
    trainLinear(solver, 100, 3'000, rng);
    for (std::uint64_t card = 0; card < 3; ++card)
    {
        EXPECT_EQ(solver.averageStrategy(card, 2), trainedKuhn().averageStrategy(card, 2));
    }
    const std::string path = (std::filesystem::temp_directory_path() / "mccfr_kuhn_partly_test.bin").string();
    ASSERT_TRUE(solver.save(path));
    EXPECT_EQ(std::filesystem::file_size(path), 5 * 8 + (12 + 3) * (8 + 2 * 4)); // header, 12 regrets, 3 averages
    Mccfr<KuhnOpeningAverage> loaded(64);
    ASSERT_TRUE(loaded.load(path));
    std::filesystem::remove(path);
    for (std::uint64_t key = 0; key < 64; ++key)
    {
        EXPECT_EQ(loaded.averageStrategy(key, 2), solver.averageStrategy(key, 2));
    }
}

// Files from before the averages had their own table hold one for every infoset.
TEST(MccfrKuhn, LoadsTheOldFormatDroppingAveragesNotKept)
{
    const std::string path = (std::filesystem::temp_directory_path() / "mccfr_kuhn_old_test.bin").string();
    const std::uint64_t opening = 1, later = 1 | (1 << 2); // Q to open, Q after one action
    {
        std::ofstream out(path, std::ios::binary);
        const std::uint64_t header[4] = {2, 2, 7, 3}; // maxActions, infosets, iterations, discounts
        out.write(reinterpret_cast<const char *>(header), sizeof(header));
        for (std::uint64_t key : {opening, later})
        {
            const std::uint64_t stored = omp::splitmix64(key); // as Mccfr stores keys (it advances its argument)
            const float regrets[2] = {1.0f, 3.0f}, average[2] = {1.0f, 1.0f};
            out.write(reinterpret_cast<const char *>(&stored), sizeof(stored));
            out.write(reinterpret_cast<const char *>(regrets), sizeof(regrets));
            out.write(reinterpret_cast<const char *>(average), sizeof(average));
        }
    }
    Mccfr<Kuhn> all(64);
    Mccfr<KuhnOpeningAverage> partly(64);
    ASSERT_TRUE(all.load(path));
    ASSERT_TRUE(partly.load(path));
    std::filesystem::remove(path);
    using Strategy = Mccfr<Kuhn>::Strategy;
    EXPECT_EQ(all.averageStrategy(opening, 2), (Strategy{0.5, 0.5}));
    EXPECT_EQ(all.averageStrategy(later, 2), (Strategy{0.5, 0.5}));
    EXPECT_EQ(partly.averageStrategy(opening, 2), (Strategy{0.5, 0.5}));
    EXPECT_EQ(partly.averageStrategy(later, 2), (Strategy{0.25, 0.75})); // no average kept: the current strategy
    EXPECT_EQ(partly.iterations(), 7u);
    EXPECT_EQ(partly.discounts(), 3u);
}

TEST(MccfrKuhn, MatchesKnownEquilibriumFamily)
{
    constexpr double tol = 0.05;
    constexpr std::uint8_t J = 0, Q = 1, K = 2;
    constexpr std::uint8_t pass = 0b0, bet = 0b1, passBet = 0b10;
    // Player 0 opening: bet J with alpha in [0, 1/3], never Q, K with 3*alpha.
    const double alpha = betProb(J, 0, 0);
    EXPECT_LE(alpha, 1.0 / 3.0 + tol);
    EXPECT_NEAR(betProb(Q, 0, 0), 0.0, tol);
    EXPECT_NEAR(betProb(K, 0, 0), 3.0 * alpha, tol);
    // Player 0 facing check-bet: fold J, call Q with alpha + 1/3, call K.
    EXPECT_NEAR(betProb(J, passBet, 2), 0.0, tol);
    EXPECT_NEAR(betProb(Q, passBet, 2), alpha + 1.0 / 3.0, tol);
    EXPECT_NEAR(betProb(K, passBet, 2), 1.0, tol);
    // Player 1 after check: bet J 1/3, check Q, bet K.
    EXPECT_NEAR(betProb(J, pass, 1), 1.0 / 3.0, tol);
    EXPECT_NEAR(betProb(Q, pass, 1), 0.0, tol);
    EXPECT_NEAR(betProb(K, pass, 1), 1.0, tol);
    // Player 1 facing bet: fold J, call Q 1/3, call K.
    EXPECT_NEAR(betProb(J, bet, 1), 0.0, tol);
    EXPECT_NEAR(betProb(Q, bet, 1), 1.0 / 3.0, tol);
    EXPECT_NEAR(betProb(K, bet, 1), 1.0, tol);
}
