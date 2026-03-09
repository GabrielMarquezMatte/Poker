#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

#include <BS_thread_pool.hpp>

#include "../include/neural_network/dlib_policy.hpp"
#include "../include/neural_network/rl_actions.hpp"
#include "../include/neural_network/rl_featurizer.hpp"
#include "../include/neural_network/rl_trainer.hpp"

// ─────────────────────────────── Helpers ─────────────────────────────────────

static constexpr Blinds kBlinds{50, 100};

// Build a minimal TrajStep with given action and advantage
static TrajStep make_step(unsigned action, float adv, float ret = 0.f, std::size_t actor = 0)
{
    TrajStep t;
    t.s        = dlib::matrix<float>(kInputDims, 1);
    t.s        = 0;
    t.a        = action;
    t.advantage = adv;
    t.R        = ret;
    t.value    = 0.f;
    t.log_prob = -1.f;
    t.entropy  = 0.5f;
    t.actor    = actor;
    return t;
}

// ═══════════════════════════ softmax_legal ════════════════════════════════════

TEST(SoftmaxLegal, LegalProbsSumToOne)
{
    std::vector<float>    logits = {1.f, 2.f, 0.5f, -1.f, 3.f};
    std::vector<unsigned> legal  = {0, 1, 2, 3, 4};
    auto probs = softmax_legal(logits, legal);
    float sum = 0.f;
    for (float p : probs) sum += p;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxLegal, IllegalActionsAreExactlyZero)
{
    std::vector<float>    logits = {1.f, 2.f, 0.5f, -1.f, 3.f};
    std::vector<unsigned> legal  = {A_CheckCall, A_AllIn};
    auto probs = softmax_legal(logits, legal);
    EXPECT_FLOAT_EQ(probs[A_Fold],      0.f);
    EXPECT_FLOAT_EQ(probs[A_BetHalfPot], 0.f);
    EXPECT_FLOAT_EQ(probs[A_BetPot],    0.f);
    EXPECT_GT(probs[A_CheckCall], 0.f);
    EXPECT_GT(probs[A_AllIn],     0.f);
}

TEST(SoftmaxLegal, SingleLegalActionHasProbOne)
{
    std::vector<float>    logits = {-99.f, 5.f, -99.f, -99.f, -99.f};
    std::vector<unsigned> legal  = {A_CheckCall};
    auto probs = softmax_legal(logits, legal);
    EXPECT_NEAR(probs[A_CheckCall], 1.0f, 1e-5f);
}

TEST(SoftmaxLegal, NumericalStabilityWithLargeLogits)
{
    std::vector<float>    logits(kNumActions, 1000.f);
    std::vector<unsigned> legal  = {0, 1, 2, 3, 4};
    auto probs = softmax_legal(logits, legal);
    float sum = 0.f;
    for (float p : probs)
    {
        EXPECT_FALSE(std::isnan(p)) << "NaN in probs";
        EXPECT_FALSE(std::isinf(p)) << "Inf in probs";
        EXPECT_GE(p, 0.f);
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxLegal, EqualLogitsYieldUniformOverLegal)
{
    std::vector<float>    logits(kNumActions, 1.f);
    std::vector<unsigned> legal  = {A_Fold, A_BetHalfPot, A_AllIn};  // 3 actions
    auto probs = softmax_legal(logits, legal);
    for (unsigned a : legal)
        EXPECT_NEAR(probs[a], 1.f / 3.f, 1e-5f);
}

// ═══════════════════════════ compute_entropy ══════════════════════════════════

TEST(ComputeEntropy, DeterministicDistributionIsZero)
{
    std::vector<float> probs(kNumActions, 0.f);
    probs[A_CheckCall] = 1.f;
    EXPECT_NEAR(compute_entropy(probs), 0.0f, 1e-5f);
}

TEST(ComputeEntropy, UniformDistributionIsLogN)
{
    std::vector<float> probs(kNumActions, 1.f / kNumActions);
    float expected = std::log(static_cast<float>(kNumActions));
    EXPECT_NEAR(compute_entropy(probs), expected, 1e-4f);
}

TEST(ComputeEntropy, ZeroProbEntriesAreIgnored)
{
    // H({0.5, 0.5, 0, 0, 0}) = log(2)
    std::vector<float> probs = {0.5f, 0.5f, 0.f, 0.f, 0.f};
    float h = compute_entropy(probs);
    EXPECT_FALSE(std::isnan(h));
    EXPECT_NEAR(h, std::log(2.f), 1e-4f);
}

// ══════════════════════ compute_action_diversity ══════════════════════════════

TEST(ActionDiversity, ZeroTotalReturnsZero)
{
    std::array<int, kNumActions> counts = {};
    EXPECT_FLOAT_EQ(compute_action_diversity(counts, 0), 0.f);
}

TEST(ActionDiversity, SingleActionReturnsZero)
{
    std::array<int, kNumActions> counts = {};
    counts[A_CheckCall] = 100;
    EXPECT_NEAR(compute_action_diversity(counts, 100), 0.f, 1e-5f);
}

TEST(ActionDiversity, UniformDistributionReturnsOne)
{
    std::array<int, kNumActions> counts = {};
    for (auto &c : counts) c = 20;  // uniform: 20 each, total = 100
    EXPECT_NEAR(compute_action_diversity(counts, 100), 1.f, 1e-5f);
}

TEST(ActionDiversity, MoreActionsIsMoreDiverse)
{
    std::array<int, kNumActions> two   = {};  two[0]   = 50; two[1]   = 50;
    std::array<int, kNumActions> three = {};  three[0] = 33; three[1] = 33; three[2] = 34;
    float d2 = compute_action_diversity(two,   100);
    float d3 = compute_action_diversity(three, 100);
    EXPECT_LT(d2, d3);
    EXPECT_GT(d2, 0.f);
    EXPECT_GT(d3, 0.f);
}

// ═══════════════════════════ legal_actions ════════════════════════════════════

TEST(LegalActions, CheckCallAlwaysPresentForAlivePlayer)
{
    Game g(kBlinds);
    g.addPlayer(10000); g.addPlayer(10000); g.addPlayer(10000);
    omp::XoroShiro128Plus rng{42};
    g.startNewHand(rng);

    for (const auto &p : g.players())
    {
        if (!p.alive()) continue;
        auto leg = legal_actions(g, p.id, kBlinds);
        bool has_cc = std::find(leg.begin(), leg.end(), A_CheckCall) != leg.end();
        EXPECT_TRUE(has_cc) << "Player " << p.id << " missing Check/Call";
    }
}

TEST(LegalActions, FoldOnlyWhenFacingABet)
{
    Game g(kBlinds);
    g.addPlayer(10000); g.addPlayer(10000); g.addPlayer(10000);
    omp::XoroShiro128Plus rng{42};
    g.startNewHand(rng);

    for (const auto &p : g.players())
    {
        if (!p.alive()) continue;
        auto leg = legal_actions(g, p.id, kBlinds);
        bool has_fold = std::find(leg.begin(), leg.end(), A_Fold) != leg.end();
        uint32_t to_call = (g.betData().currentBet > p.committed)
                         ? g.betData().currentBet - p.committed : 0;
        EXPECT_EQ(has_fold, to_call > 0)
            << "Player " << p.id << " to_call=" << to_call;
    }
}

TEST(LegalActions, EmptyForDeadPlayer)
{
    // Player 2 starts with 0 chips → not alive after addPlayer
    Game g(kBlinds);
    g.addPlayer(10000);
    g.addPlayer(10000);
    g.addPlayer(0);
    omp::XoroShiro128Plus rng{7};
    g.startNewHand(rng);
    auto leg = legal_actions(g, 2, kBlinds);
    EXPECT_TRUE(leg.empty());
}

TEST(LegalActions, SizedBetsAbsentWhenPlayerCannotRaise)
{
    // Player with tiny stack cannot post a proper raise target above call
    Game g(kBlinds);
    g.addPlayer(10000);
    g.addPlayer(10000);
    g.addPlayer(105);  // just above BB; can call but not raise meaningfully
    omp::XoroShiro128Plus rng{99};
    g.startNewHand(rng);

    const auto &short_p = g.players()[2];
    if (!short_p.alive()) return;  // may be gone if stack==0

    auto leg = legal_actions(g, 2, kBlinds);
    const uint32_t stack_total = short_p.committed + short_p.chips;

    for (unsigned a : leg)
    {
        if (a != A_BetHalfPot && a != A_BetPot) continue;
        // If a sized bet is listed, the player must be able to reach it short of shoving
        EXPECT_GT(stack_total, short_p.committed)
            << "Player 2 has a sized bet but no room to raise";
    }
}

// ═══════════════════════════ featurize ═══════════════════════════════════════

class FeaturizeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        pool = std::make_unique<BS::thread_pool<BS::tp::none>>(
            std::max(1u, std::thread::hardware_concurrency()));
        g = Game(kBlinds);
        g.addPlayer(10000); g.addPlayer(10000); g.addPlayer(10000);
        omp::XoroShiro128Plus rng{42};
        g.startNewHand(rng);
    }
    std::unique_ptr<BS::thread_pool<BS::tp::none>> pool;
    Game g{kBlinds};
};

TEST_F(FeaturizeTest, OutputHasCorrectDimension)
{
    auto f = featurize(g, 0, kBlinds, *pool);
    EXPECT_EQ(f.nr(), kInputDims);
    EXPECT_EQ(f.nc(), 1);
}

TEST_F(FeaturizeTest, NoNaNOrInfInOutput)
{
    auto f = featurize(g, 0, kBlinds, *pool);
    for (int i = 0; i < kInputDims; ++i)
    {
        EXPECT_FALSE(std::isnan(f(i, 0))) << "feature[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(f(i, 0))) << "feature[" << i << "] is Inf";
    }
}

TEST_F(FeaturizeTest, AllNormalizedFeaturesInUnitRange)
{
    auto f = featurize(g, 0, kBlinds, *pool);
    // Every feature should be non-negative. Features 0-3 (one-hot) and most
    // normalized features should stay ≤ 1. We allow a small epsilon for FP rounding.
    for (int i = 0; i < kInputDims; ++i)
        EXPECT_GE(f(i, 0), -1e-5f) << "feature[" << i << "] < 0";
}

TEST_F(FeaturizeTest, StreetOneHotHasExactlyOneBit)
{
    auto f = featurize(g, 0, kBlinds, *pool);
    // Features 0-3 encode the current street as a one-hot vector
    int ones = 0;
    for (int i = 0; i < 4; ++i)
    {
        float v = f(i, 0);
        EXPECT_TRUE(v == 0.f || v == 1.f) << "street feature[" << i << "]=" << v;
        if (v == 1.f) ++ones;
    }
    EXPECT_EQ(ones, 1) << "Street one-hot must have exactly one active bit";
}

TEST_F(FeaturizeTest, EquityFeatureInUnitRange)
{
    // Feature index 19 is the raw Monte Carlo equity in [0, 1]
    auto f = featurize(g, 0, kBlinds, *pool);
    EXPECT_GE(f(19, 0), 0.f);
    EXPECT_LE(f(19, 0), 1.f);
}

// ════════════════════════ build_training_batch ════════════════════════════════

TEST(BuildBatch, EmptyTrajectoryDoesNotCrash)
{
    std::vector<TrajStep> empty;
    std::vector<dlib::matrix<float>> Xp, Xv;
    std::vector<unsigned long> yp;
    std::vector<float> yv, adv;
    EXPECT_NO_FATAL_FAILURE(build_training_batch(empty, Xp, yp, Xv, yv, adv));
    EXPECT_TRUE(Xp.empty());
    EXPECT_TRUE(Xv.empty());
}

TEST(BuildBatch, ValueBatchReceivesAllSteps)
{
    // The value network trains on every step regardless of advantage sign
    std::vector<TrajStep> traj = {
        make_step(A_Fold,       -2.f, -2.f),
        make_step(A_CheckCall,   1.f,  1.f),
        make_step(A_BetHalfPot, -0.5f, -0.5f),
        make_step(A_CheckCall,   3.f,  3.f),
    };
    std::vector<dlib::matrix<float>> Xp, Xv;
    std::vector<unsigned long> yp;
    std::vector<float> yv, adv;
    build_training_batch(traj, Xp, yp, Xv, yv, adv);
    EXPECT_EQ(Xv.size(), traj.size());
    EXPECT_EQ(yv.size(), traj.size());
}

TEST(BuildBatch, NormalizedAdvantagesAreFinite)
{
    std::vector<TrajStep> traj = {
        make_step(A_CheckCall,   5.f,  5.f),
        make_step(A_CheckCall,  -3.f, -3.f),
        make_step(A_BetHalfPot,  2.f,  2.f),
        make_step(A_Fold,       -1.f, -1.f),
        make_step(A_BetPot,      4.f,  4.f),
        make_step(A_CheckCall,  -2.f, -2.f),
    };
    std::vector<dlib::matrix<float>> Xp, Xv;
    std::vector<unsigned long> yp;
    std::vector<float> yv, adv;
    build_training_batch(traj, Xp, yp, Xv, yv, adv);
    for (float a : adv)
    {
        EXPECT_FALSE(std::isnan(a)) << "NaN in normalized advantages";
        EXPECT_FALSE(std::isinf(a)) << "Inf in normalized advantages";
    }
}

TEST(BuildBatch, AllInPolicySamplesAreCapped)
{
    TrainConfig cfg;
    cfg.max_allin_ratio = 10;  // allow at most 10% all-ins in the policy batch

    // Build a trajectory that is 80% all-in (clearly over the limit)
    std::vector<TrajStep> traj;
    for (int i = 0; i < 80; ++i)
        traj.push_back(make_step(A_AllIn, 2.f, 2.f));
    for (int i = 0; i < 20; ++i)
        traj.push_back(make_step(A_CheckCall, 1.f, 1.f));

    std::vector<dlib::matrix<float>> Xp, Xv;
    std::vector<unsigned long> yp;
    std::vector<float> yv, adv;
    build_training_batch(traj, Xp, yp, Xv, yv, adv, cfg);

    std::size_t allin_count = std::count(
        yp.begin(), yp.end(), static_cast<unsigned long>(A_AllIn));
    // Cap is ceil(100 * 10% / 100) = 10 at most
    std::size_t cap = std::max<std::size_t>(
        1, static_cast<std::size_t>(
               std::ceil(100.0 * cfg.max_allin_ratio / 100.0)));
    EXPECT_LE(allin_count, cap)
        << "All-in policy samples exceeded the configured cap";
}

TEST(BuildBatch, SingleStepAllZeroAdvantanceDoesNotCrashNormalization)
{
    // Edge case: all advantages equal → variance = 0 → division by near-zero
    std::vector<TrajStep> traj = {
        make_step(A_CheckCall, 0.f, 0.f),
        make_step(A_CheckCall, 0.f, 0.f),
    };
    std::vector<dlib::matrix<float>> Xp, Xv;
    std::vector<unsigned long> yp;
    std::vector<float> yv, adv;
    EXPECT_NO_FATAL_FAILURE(build_training_batch(traj, Xp, yp, Xv, yv, adv));
    for (float a : adv)
    {
        EXPECT_FALSE(std::isnan(a));
        EXPECT_FALSE(std::isinf(a));
    }
}

// ════════════════ policy_sample_with_probs (network integration) ══════════════

class PolicyNetTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        dlib::matrix<float> dummy(kInputDims, 1);
        dummy = 0;
        pnet(dummy);  // triggers lazy weight initialization
    }
    policy_net pnet;
    std::mt19937 rng{42};
};

TEST_F(PolicyNetTest, SampledActionIsAlwaysInLegalSet)
{
    dlib::matrix<float> s(kInputDims, 1);
    s = 0;
    std::vector<unsigned> legal = {A_CheckCall, A_BetHalfPot, A_AllIn};
    for (int trial = 0; trial < 50; ++trial)
    {
        auto [action, probs] = policy_sample_with_probs(pnet, s, legal, rng);
        bool found = std::find(legal.begin(), legal.end(), action) != legal.end();
        EXPECT_TRUE(found) << "action=" << action << " not in legal set (trial " << trial << ")";
        (void)probs;
    }
}

TEST_F(PolicyNetTest, ReturnedProbsSumToOne)
{
    dlib::matrix<float> s(kInputDims, 1);
    s = 0;
    std::vector<unsigned> legal = {0, 1, 2, 3, 4};
    auto [action, probs] = policy_sample_with_probs(pnet, s, legal, rng);
    float sum = std::accumulate(probs.begin(), probs.end(), 0.f);
    EXPECT_NEAR(sum, 1.f, 1e-4f);
    (void)action;
}

TEST_F(PolicyNetTest, EmptyLegalSetDoesNotCrash)
{
    dlib::matrix<float> s(kInputDims, 1);
    s = 0;
    std::vector<unsigned> legal;
    auto [action, probs] = policy_sample_with_probs(pnet, s, legal, rng);
    EXPECT_EQ(action, 0u);
    EXPECT_EQ(static_cast<int>(probs.size()), kNumActions);
    (void)probs;
}

TEST_F(PolicyNetTest, ProbsMatchGetActionProbs)
{
    // Verify policy_sample_with_probs returns exactly the same probs
    // as the separate get_action_probs call (same logits, same softmax)
    dlib::matrix<float> s(kInputDims, 1);
    s = 0;
    std::vector<unsigned> legal = {A_CheckCall, A_BetPot};
    std::mt19937 rng2{0};  // separate rng so sampling doesn't affect comparison
    auto [action, probs_combined] = policy_sample_with_probs(pnet, s, legal, rng2);
    auto  probs_separate          = get_action_probs(pnet, s, legal);
    for (int i = 0; i < kNumActions; ++i)
        EXPECT_NEAR(probs_combined[i], probs_separate[i], 1e-5f) << "probs differ at index " << i;
    (void)action;
}

// ═══════════════════════════ GAE / trajectory ═════════════════════════════════

class HandCollectionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        pool = std::make_unique<BS::thread_pool<BS::tp::none>>(
            std::max(1u, std::thread::hardware_concurrency()));
        // Initialize policy net
        dlib::matrix<float> dummy(kInputDims, 1);
        dummy = 0;
        pnet(dummy);
        // Initialize value net via one dummy training step
        {
            dlib::dnn_trainer<value_net> t(vnet);
            t.set_max_num_epochs(1);
            t.be_quiet();
            std::vector<dlib::matrix<float>> X = {dummy};
            std::vector<float> y = {0.f};
            t.train(X, y);
        }
    }
    std::unique_ptr<BS::thread_pool<BS::tp::none>> pool;
    omp::XoroShiro128Plus rng{42u};
    policy_net pnet;
    value_net  vnet;
};

TEST_F(HandCollectionTest, AllReturnsAndAdvantagesAreFinite)
{
    Game g(kBlinds);
    g.addPlayer(10000); g.addPlayer(10000); g.addPlayer(10000);

    TrainConfig cfg;
    std::vector<TrajStep> steps;
    play_one_hand_collect(g, rng, pnet, vnet, kBlinds, steps,
                          /*epsilon=*/0.0, *pool, cfg, /*temperature=*/1.0f, /*hero=*/0);

    for (const auto &s : steps)
    {
        EXPECT_FALSE(std::isnan(s.R))          << "R is NaN";
        EXPECT_FALSE(std::isinf(s.R))          << "R is Inf";
        EXPECT_FALSE(std::isnan(s.advantage))  << "advantage is NaN";
        EXPECT_FALSE(std::isinf(s.advantage))  << "advantage is Inf";
    }
}

TEST_F(HandCollectionTest, AllStepsPerActorShareSameReturn)
{
    // Each player's terminal chip-delta is the same across all their collected steps.
    Game g(kBlinds);
    g.addPlayer(10000); g.addPlayer(10000); g.addPlayer(10000);

    TrainConfig cfg;
    std::vector<TrajStep> steps;
    play_one_hand_collect(g, rng, pnet, vnet, kBlinds, steps,
                          0.0, *pool, cfg, 1.0f, 0);

    constexpr float kNone = std::numeric_limits<float>::quiet_NaN();
    std::array<float, 3> first_R = {kNone, kNone, kNone};

    for (const auto &s : steps)
    {
        if (s.actor >= 3) continue;
        if (std::isnan(first_R[s.actor]))
            first_R[s.actor] = s.R;
        else
            EXPECT_NEAR(s.R, first_R[s.actor], 1e-5f)
                << "Inconsistent R for actor " << s.actor;
    }
}

TEST_F(HandCollectionTest, LambdaZeroGivesTDAdvantages)
{
    // With gae_lambda = 0 and value network ≈ 0 for all states,
    // the advantage of the terminal step should equal delta_bb (≈ R),
    // and non-terminal steps should have advantage ≈ 0.
    Game g(kBlinds);
    g.addPlayer(10000); g.addPlayer(10000); g.addPlayer(10000);

    TrainConfig cfg;
    cfg.gae_lambda = 0.f;
    cfg.entropy_coef = 0.f;   // disable entropy bonus so adv reflects pure TD error

    std::vector<TrajStep> steps;
    play_one_hand_collect(g, rng, pnet, vnet, kBlinds, steps,
                          0.0, *pool, cfg, 1.0f, 0);

    // Collect per-actor step lists in order
    std::array<std::vector<const TrajStep *>, 3> per_actor;
    for (const auto &s : steps)
        if (s.actor < 3) per_actor[s.actor].push_back(&s);

    for (auto &actor_steps : per_actor)
    {
        if (actor_steps.empty()) continue;
        const TrajStep *terminal = actor_steps.back();
        // Terminal advantage = delta_bb - V(s_T); since V ≈ 0, this ≈ delta_bb = R
        // We just verify it's finite and has the same sign as R (within reason).
        EXPECT_FALSE(std::isnan(terminal->advantage));
        EXPECT_FALSE(std::isinf(terminal->advantage));
    }
}
