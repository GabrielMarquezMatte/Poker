#include <benchmark/benchmark.h>

#include <array>
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

static Game make_started_game(std::size_t n = 3, uint32_t chips = 10000)
{
    Game g(kBlinds);
    for (std::size_t i = 0; i < n; ++i)
        g.addPlayer(chips);
    omp::XoroShiro128Plus rng{42};
    g.startNewHand(rng);
    return g;
}

static policy_net make_policy_net()
{
    policy_net net;
    dlib::matrix<float> dummy(kInputDims, 1);
    dummy = 0;
    net(dummy);
    return net;
}

static value_net make_value_net()
{
    value_net net;
    dlib::matrix<float> dummy(kInputDims, 1);
    dummy = 0;
    {
        dlib::dnn_trainer<value_net> t(net);
        t.set_max_num_epochs(1);
        t.be_quiet();
        std::vector<dlib::matrix<float>> X = {dummy};
        std::vector<float> y = {0.f};
        t.train(X, y);
    }
    return net;
}

// ══════════════════════════ Featurizer ═══════════════════════════════════════

// Measures the full featurize() call including equity Monte Carlo.
// Run with 1, 4, and 8 threads to show thread-pool scaling.
static void BM_Featurize(benchmark::State &st)
{
    auto g = make_started_game();
    BS::thread_pool<BS::tp::none> pool(static_cast<std::size_t>(st.range(0)));
    for (auto _ : st)
    {
        auto feat = featurize(g, 0, kBlinds, pool);
        benchmark::DoNotOptimize(feat);
    }
    st.SetLabel("threads=" + std::to_string(st.range(0)));
}
BENCHMARK(BM_Featurize)->Arg(1)->Arg(4)->Arg(8)->Unit(benchmark::kMicrosecond);

// ═══════════════════════════ Softmax ════════════════════════════════════════

// Sub-microsecond: shows cost of action masking + softmax
static void BM_SoftmaxLegal_AllActions(benchmark::State &st)
{
    std::vector<float>    logits = {1.2f, 0.8f, -0.5f, 2.1f, 0.3f};
    std::vector<unsigned> legal  = {0, 1, 2, 3, 4};
    for (auto _ : st)
    {
        auto probs = softmax_legal(logits, legal);
        benchmark::DoNotOptimize(probs);
    }
}
BENCHMARK(BM_SoftmaxLegal_AllActions);

static void BM_SoftmaxLegal_PartialLegal(benchmark::State &st)
{
    std::vector<float>    logits = {1.2f, 0.8f, -0.5f, 2.1f, 0.3f};
    std::vector<unsigned> legal  = {A_CheckCall, A_AllIn};
    for (auto _ : st)
    {
        auto probs = softmax_legal(logits, legal);
        benchmark::DoNotOptimize(probs);
    }
}
BENCHMARK(BM_SoftmaxLegal_PartialLegal);

// ═══════════════════════════ Legal actions ═══════════════════════════════════

static void BM_LegalActions(benchmark::State &st)
{
    auto g = make_started_game();
    for (auto _ : st)
    {
        auto leg = legal_actions(g, 0, kBlinds);
        benchmark::DoNotOptimize(leg);
    }
}
BENCHMARK(BM_LegalActions);

// ══════════════════════════ Network inference ═════════════════════════════════

// Policy forward pass (32 → 512 → 256 → 128 → 5)
static void BM_PolicyForwardPass(benchmark::State &st)
{
    auto net = make_policy_net();
    dlib::matrix<float> s(kInputDims, 1);
    s = 0;
    for (auto _ : st)
    {
        auto logits = get_action_logits(net, s);
        benchmark::DoNotOptimize(logits);
    }
}
BENCHMARK(BM_PolicyForwardPass)->Unit(benchmark::kMicrosecond);

// Value forward pass (32 → 128 → 64 → 1)
static void BM_ValueForwardPass(benchmark::State &st)
{
    auto net = make_value_net();
    dlib::matrix<float> s(kInputDims, 1);
    s = 0;
    for (auto _ : st)
    {
        float v = predict_value(net, s);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_ValueForwardPass)->Unit(benchmark::kMicrosecond);

// ════════════ policy_sample_with_probs vs separate calls ══════════════════════
//
// These two benchmarks are intentionally side-by-side to demonstrate the
// speedup from eliminating the redundant policy forward pass.

// Combined: single forward pass → action + probs
static void BM_PolicySampleWithProbs(benchmark::State &st)
{
    auto net = make_policy_net();
    dlib::matrix<float> s(kInputDims, 1);
    s = 0;
    std::vector<unsigned> legal = {0, 1, 2, 3, 4};
    std::mt19937 rng{42};
    for (auto _ : st)
    {
        auto [action, probs] = policy_sample_with_probs(net, s, legal, rng);
        benchmark::DoNotOptimize(action);
        benchmark::DoNotOptimize(probs);
    }
}
BENCHMARK(BM_PolicySampleWithProbs)->Unit(benchmark::kMicrosecond);

// Separate: two forward passes (old approach, kept for comparison)
static void BM_PolicySampleThenGetProbs(benchmark::State &st)
{
    auto net = make_policy_net();
    dlib::matrix<float> s(kInputDims, 1);
    s = 0;
    std::vector<unsigned> legal = {0, 1, 2, 3, 4};
    std::mt19937 rng{42};
    for (auto _ : st)
    {
        auto action = policy_sample(net, s, legal, rng);
        auto probs  = get_action_probs(net, s, legal);
        benchmark::DoNotOptimize(action);
        benchmark::DoNotOptimize(probs);
    }
}
BENCHMARK(BM_PolicySampleThenGetProbs)->Unit(benchmark::kMicrosecond);

// ════════════════════════ Action diversity ════════════════════════════════════

static void BM_ComputeActionDiversity(benchmark::State &st)
{
    std::array<int, kNumActions> counts = {30, 40, 10, 12, 8};
    for (auto _ : st)
    {
        float d = compute_action_diversity(counts, 100);
        benchmark::DoNotOptimize(d);
    }
}
BENCHMARK(BM_ComputeActionDiversity);

// ═════════════════════════ build_training_batch ═══════════════════════════════

static void BM_BuildTrainingBatch(benchmark::State &st)
{
    const int n = static_cast<int>(st.range(0));

    // Pre-build a trajectory of n steps (mix of actions and advantage signs)
    std::vector<TrajStep> traj;
    traj.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        TrajStep t;
        t.s        = dlib::matrix<float>(kInputDims, 1);
        t.s        = 0;
        t.a        = static_cast<unsigned>(i % kNumActions);
        t.advantage = (i % 3 == 0) ? -1.f : 1.f;
        t.R        = t.advantage;
        t.value    = 0.f;
        t.log_prob = -1.f;
        t.entropy  = 0.5f;
        t.actor    = 0;
        traj.push_back(t);
    }

    for (auto _ : st)
    {
        std::vector<dlib::matrix<float>> Xp, Xv;
        std::vector<unsigned long> yp;
        std::vector<float> yv, adv;
        build_training_batch(traj, Xp, yp, Xv, yv, adv);
        benchmark::DoNotOptimize(Xp);
        benchmark::DoNotOptimize(Xv);
    }
    st.SetItemsProcessed(st.iterations() * n);
}
BENCHMARK(BM_BuildTrainingBatch)->Arg(256)->Arg(1024)->Arg(4096)->Unit(benchmark::kMicrosecond);

// ═════════════════════════ Full hand collection ═══════════════════════════════

// End-to-end: one hand played with policy inference and trajectory collection.
// This is the innermost loop of train_epoch.
static void BM_PlayOneHand(benchmark::State &st)
{
    auto pool  = std::make_unique<BS::thread_pool<BS::tp::none>>(
        std::max(1u, std::thread::hardware_concurrency()));
    auto pnet  = make_policy_net();
    auto vnet  = make_value_net();
    omp::XoroShiro128Plus rng{42};
    TrainConfig cfg;

    for (auto _ : st)
    {
        Game g(kBlinds);
        g.addPlayer(10000); g.addPlayer(10000); g.addPlayer(10000);
        std::vector<TrajStep> steps;
        play_one_hand_collect(g, rng, pnet, vnet, kBlinds, steps,
                              /*epsilon=*/0.0, *pool, cfg, /*temperature=*/1.0f, /*hero=*/0);
        benchmark::DoNotOptimize(steps);
    }
}
BENCHMARK(BM_PlayOneHand)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
