#ifndef __POKER_RL_TRAINER_HPP__
#define __POKER_RL_TRAINER_HPP__
#include <random>
#include <dlib/dnn.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <BS_thread_pool.hpp>
#include "dlib_policy.hpp"
#include "rl_featurizer.hpp"
#include "rl_actions.hpp"

// Training configuration
struct TrainConfig
{
    float entropy_coef = 0.05f;        // Entropy bonus coefficient (higher = more exploration)
    float action_diversity_coef = 0.1f; // Penalty for action distribution collapse
    float min_action_prob = 0.05f;     // Minimum probability for any action (soft constraint)
    bool self_play = true;             // Train all players with the policy (not just player 0)
    float allin_penalty = 0.3f;        // Penalty for excessive all-in usage
    int max_allin_ratio = 30;          // Max percentage of all-ins before penalty applies
    float gae_lambda = 0.95f;          // GAE λ: 0=TD(0), 1=Monte Carlo
};

// Diversity thresholds for adaptive epsilon and learning-rate control
static constexpr float kHighDiversityThreshold = 0.5f;
static constexpr float kLowDiversityThreshold  = 0.3f;

struct TrajStep
{
    dlib::matrix<float> s;
    unsigned a;
    std::size_t actor = 0;
    float R = 0.f;          // Total return
    float value = 0.f;      // Value estimate at this state
    float advantage = 0.f;  // Advantage = R - V(s)
    float log_prob = 0.f;
    float entropy = 0.f;    // Policy entropy for this step
};

// Statistics for monitoring training
struct EpochStats
{
    float mean_reward = 0.f;
    float std_reward = 0.f;
    float mean_advantage = 0.f;
    float mean_entropy = 0.f;
    float policy_loss = 0.f;
    float value_loss = 0.f;
    std::array<int, kNumActions> action_counts = {};
    int total_steps = 0;
    float win_rate = 0.f;
    float action_diversity = 0.f;  // Entropy of action distribution (0-1, higher = more diverse)
};

// Compute action distribution entropy (measure of diversity)
inline float compute_action_diversity(const std::span<const int, kNumActions> counts, int total)
{
    if (total == 0) return 0.f;
    float entropy = 0.f;
    for (int i = 0; i < kNumActions; ++i)
    {
        if (counts[i] > 0)
        {
            float p = static_cast<float>(counts[i]) / static_cast<float>(total);
            entropy -= p * std::log(p + 1e-8f);
        }
    }
    // Normalize by max entropy (uniform distribution)
    float max_entropy = std::log(static_cast<float>(kNumActions));
    return entropy / max_entropy;
}

// Play one hand with self-play (all players use policy) or mixed (player 0 = policy, others = random)
template <class TRng>
float play_one_hand_collect(Game &g, TRng &rng, policy_net &net, value_net &vnet,
                            const Blinds &blinds,
                            std::vector<TrajStep> &out_steps,
                            double epsilon,
                            BS::thread_pool<BS::tp::none> &pool,
                            const TrainConfig &config,
                            float temperature = 1.0f,
                            std::size_t hero_id = 0)  // Only collect for this player if not self_play
{
    g.startNewHand(rng);
    const auto n = g.players().size();

    std::vector<std::uint32_t> chips_before(n);
    for (auto const &p : g.players())
        chips_before[p.id] = p.chips;

    std::vector<std::vector<TrajStep>> per_actor(n);

    static thread_local omp::XoroShiro128Plus exrng{std::random_device{}()};
    static thread_local std::uniform_real_distribution<double> U(0.0, 1.0);

    while (g.state() != GameState::Finished)
    {
        if (!g.hasCurrentActor())
        {
            g.applyAction(rng, ActionStruct{ActionType::Check, 0});
            continue;
        }

        auto cur = g.currentPlayer().id;
        auto leg = legal_actions(g, cur, blinds);
        if (leg.empty())
        {
            g.applyAction(rng, ActionStruct{ActionType::Fold, 0});
            continue;
        }

        unsigned aidx;
        float log_prob = 0.f;
        float entropy = 0.f;
        float value_est = 0.f;
        bool use_policy = config.self_play || (cur == hero_id);
        if (!use_policy)
        {
            omp::FastUniformIntDistribution<size_t> D(0, leg.size() - 1);
            aidx = leg[D(exrng)];
            g.applyAction(rng, to_engine_action(aidx, g, cur, blinds));
            continue;
        }
        auto s = featurize(g, cur, blinds, pool);
        value_est = predict_value(vnet, s);

        if (U(exrng) < epsilon)
        {
            omp::FastUniformIntDistribution<size_t> D(0, leg.size() - 1);
            aidx = leg[D(exrng)];
            log_prob = -std::log(static_cast<float>(leg.size()));
            entropy = std::log(static_cast<float>(leg.size()));
        }
        else
        {
            auto [sampled, probs] = policy_sample_with_probs(net, s, leg, exrng, temperature);
            aidx = sampled;
            log_prob = std::log(std::max(1e-8f, probs[aidx]));
            entropy = compute_entropy(probs);
        }
        TrajStep step;
        step.s = s;
        step.a = aidx;
        step.actor = cur;
        step.value = value_est;
        step.log_prob = log_prob;
        step.entropy = entropy;
        per_actor[cur].push_back(step);
        g.applyAction(rng, to_engine_action(aidx, g, cur, blinds));
    }
    const float bb = static_cast<float>(std::max<std::uint32_t>(1, blinds.bigBlind));
    // Compute returns and advantages using GAE (TD-λ backwards pass).
    // Reward is terminal-only (delta_bb at hand end); no discounting (γ = 1).
    //   δ_t = r_t + V(s_{t+1}) − V(s_t)   (r_t = 0 for t < T, delta_bb for last step)
    //   A_t = δ_t + λ · A_{t+1}            (backwards induction)
    for (std::size_t i = 0; i < n; ++i)
    {
        float delta_bb = (static_cast<float>(g.players()[i].chips) - static_cast<float>(chips_before[i])) / bb;
        auto &steps = per_actor[i];
        float gae = 0.f;
        for (int t = static_cast<int>(steps.size()) - 1; t >= 0; --t)
        {
            steps[t].R = delta_bb;
            bool is_terminal = (t == static_cast<int>(steps.size()) - 1);
            float next_val = is_terminal ? 0.f : steps[t + 1].value;
            float reward_t = is_terminal ? delta_bb : 0.f;
            float td_error = reward_t + next_val - steps[t].value;
            gae = td_error + config.gae_lambda * gae;
            steps[t].advantage = gae;
        }
    }
    for (auto &vec : per_actor)
    {
        out_steps.insert(out_steps.end(),
                         std::make_move_iterator(vec.begin()),
                         std::make_move_iterator(vec.end()));
    }
    return (static_cast<float>(g.players()[0].chips) - static_cast<float>(chips_before[0])) / bb;
}

// Build training batch with improved handling to prevent action collapse
inline void build_training_batch(
    const std::span<const TrajStep> traj,
    std::vector<dlib::matrix<float>> &X_policy,
    std::vector<unsigned long> &y_policy,
    std::vector<dlib::matrix<float>> &X_value,
    std::vector<float> &y_value,
    std::vector<float> &advantages_out,
    const TrainConfig &config = TrainConfig{})
{
    if (traj.empty())
        return;

    // Advantage-adjustment thresholds
    constexpr float    kMinShoveAdvantage = 0.25f; // Minimum adj_adv to keep a shove label
    constexpr float    kAdvRepeat1        = 0.5f;  // Add one extra copy above this advantage
    constexpr float    kAdvRepeat2        = 1.0f;  // Add two extra copies above this advantage
    constexpr unsigned kNegAdvKeepEvery   = 20u;   // Keep 1-in-N non-positive samples as stabilizer
    constexpr float    kRareActionFreq    = 0.1f;  // Action is "rare" below this frequency

    // Phase 1: Count actions to detect distribution collapse.
    std::array<int, kNumActions> action_counts = {};
    for (const auto &t : traj)
        action_counts[t.a]++;

    int total_actions = static_cast<int>(traj.size());
    const float allin_ratio = (total_actions > 0)
        ? (100.f * static_cast<float>(action_counts[A_AllIn]) / static_cast<float>(total_actions))
        : 0.f;

    // Phase 2: Compute adjusted advantages (GAE + entropy bonus ± all-in penalty).
    std::vector<float> adj_adv(traj.size());
    for (size_t i = 0; i < traj.size(); ++i)
    {
        adj_adv[i] = traj[i].advantage;
        // Encourage higher-entropy decision points.
        adj_adv[i] += config.entropy_coef * traj[i].entropy;
        if (traj[i].a != A_AllIn)
            continue;
        // Penalize all-in labels when action distribution is already all-in heavy.
        if (allin_ratio > static_cast<float>(config.max_allin_ratio))
            adj_adv[i] -= config.allin_penalty * (allin_ratio - config.max_allin_ratio) / 100.f;
    }

    // Phase 3: Normalize adjusted advantages (zero mean, unit variance).
    float mu = std::accumulate(adj_adv.begin(), adj_adv.end(), 0.f) / static_cast<float>(adj_adv.size());

    float var = 0.f;
    for (float v : adj_adv) { float d = v - mu; var += d * d; }
    var /= static_cast<float>(std::max<size_t>(1, adj_adv.size()));
    float sd = std::sqrt(std::max(1e-6f, var));

    for (float &v : adj_adv) v = (v - mu) / sd;

    // Phase 4: Build policy and value batches.
    X_policy.reserve(X_policy.size() + traj.size() * 3);
    y_policy.reserve(y_policy.size() + traj.size() * 3);
    X_value.reserve(X_value.size() + traj.size());
    y_value.reserve(y_value.size() + traj.size());
    advantages_out.reserve(advantages_out.size() + traj.size() * 3);

    // Cap explicit all-in supervision to avoid label collapse.
    const std::size_t max_allin_policy_samples =
        std::max<std::size_t>(
            1,
            static_cast<std::size_t>(std::ceil(
                static_cast<double>(traj.size()) *
                static_cast<double>(std::max(1, config.max_allin_ratio)) / 100.0)));
    std::size_t kept_allin_policy = 0;

    for (size_t i = 0; i < traj.size(); ++i)
    {
        const auto &t = traj[i];
        const float a = adj_adv[i];

        // Value network still learns from all states.
        X_value.push_back(t.s);
        y_value.push_back(t.R);

        // Policy network focuses on favorable actions; keep sparse negatives as a stabilizer.
        if (a <= 0.f && (i % kNegAdvKeepEvery) != 0u)
            continue;

        if (t.a == A_AllIn)
        {
            // Require clearly positive advantage for shove labels.
            if (a <= kMinShoveAdvantage || kept_allin_policy >= max_allin_policy_samples)
                continue;
            ++kept_allin_policy;
        }

        // Higher-advantage samples repeat to increase their gradient contribution.
        // Shove labels are never repeated to avoid over-amplifying all-in gradient.
        int repeats = 1;
        if (t.a != A_AllIn)
        {
            if (a > kAdvRepeat1) ++repeats;
            if (a > kAdvRepeat2) ++repeats;
        }

        for (int r = 0; r < repeats; ++r)
        {
            X_policy.push_back(t.s);
            y_policy.push_back(t.a);
            advantages_out.push_back(a / static_cast<float>(r + 1));
        }

        // Reinforce rare, successful non-all-in actions to maintain diversity.
        // (Merged from former Phase 5 — safe here because a > kAdvRepeat1 implies a > 0,
        //  so Phase 4's negative-advantage filter cannot have triggered for this step.)
        if (t.a != A_AllIn)
        {
            const float action_freq = static_cast<float>(action_counts[t.a]) / static_cast<float>(total_actions);
            if (action_freq < kRareActionFreq && t.R > 0.f && a > kAdvRepeat1)
            {
                X_policy.push_back(t.s);
                y_policy.push_back(t.a);
                advantages_out.push_back(std::max(kAdvRepeat1, a));
            }
        }
    }
}

template <class TRng>
EpochStats train_epoch(policy_net &net, value_net &vnet,
                       dlib::dnn_trainer<policy_net> &policy_trainer,
                       dlib::dnn_trainer<value_net> &value_trainer,
                       Game &g, TRng &rng,
                       const Blinds &blinds, int hands_per_epoch,
                       double &epsilon, float temperature,
                       std::uint32_t starting_chips,
                       BS::thread_pool<BS::tp::none> &pool,
                       const TrainConfig &config = TrainConfig{})
{
    EpochStats stats;
    std::vector<TrajStep> traj;
    traj.reserve(8192);

    std::vector<float> hand_rewards;
    hand_rewards.reserve(hands_per_epoch);
    int wins = 0;

    // Collect trajectories
    for (int h = 0; h < hands_per_epoch; ++h)
    {
        if (std::any_of(g.players().begin(), g.players().end(),
                        [&](const auto &p) { return p.chips < blinds.bigBlind * 3; }))
            g.resetPlayerChips(starting_chips);

        float reward = play_one_hand_collect(g, rng, net, vnet, blinds, traj, epsilon, pool,
                                             config, temperature, 0);
        hand_rewards.push_back(reward);
        if (reward > 0)
            wins++;
    }

    stats.win_rate = static_cast<float>(wins) / static_cast<float>(hands_per_epoch);

    // Compute action distribution and entropy
    float total_entropy = 0.f;
    for (auto const &t : traj)
    {
        stats.action_counts[t.a]++;
        total_entropy += t.entropy;
    }
    stats.total_steps = static_cast<int>(traj.size());
    stats.mean_entropy = stats.total_steps > 0 ? total_entropy / static_cast<float>(stats.total_steps) : 0.f;
    stats.action_diversity = compute_action_diversity(stats.action_counts, stats.total_steps);

    // Build training batches
    std::vector<dlib::matrix<float>> X_policy, X_value;
    std::vector<unsigned long> y_policy;
    std::vector<float> y_value, advantages;
    build_training_batch(traj, X_policy, y_policy, X_value, y_value, advantages, config);

    if (X_policy.empty() || X_value.empty())
    {
        return stats;
    }

    // Compute statistics
    stats.mean_reward = std::accumulate(hand_rewards.begin(), hand_rewards.end(), 0.f) /
                        static_cast<float>(hand_rewards.size());

    float var = 0.f;
    for (float r : hand_rewards)
    {
        float d = r - stats.mean_reward;
        var += d * d;
    }
    stats.std_reward = std::sqrt(var / static_cast<float>(hand_rewards.size()));

    if (!advantages.empty())
    {
        stats.mean_advantage = std::accumulate(advantages.begin(), advantages.end(), 0.f) /
                               static_cast<float>(advantages.size());
    }

    // Train policy network with adjusted learning rate based on action diversity
    // Lower LR if actions are collapsing to prevent further collapse
    float policy_lr = 1e-4f;
    if (stats.action_diversity < kLowDiversityThreshold)
        policy_lr *= 0.5f;

    policy_trainer.set_learning_rate(policy_lr);
    policy_trainer.set_mini_batch_size(std::min<size_t>(256, X_policy.size()));
    policy_trainer.train(X_policy, y_policy);

    // Train value network
    value_trainer.set_mini_batch_size(std::min<size_t>(256, X_value.size()));
    value_trainer.train(X_value, y_value);

    // Adaptive epsilon decay based on action diversity
    // Decay slower if actions are collapsing (need more exploration)
    if (stats.action_diversity > kHighDiversityThreshold)
    {
        epsilon = std::max(0.05, epsilon * 0.997);  // Normal decay
    }
    else if (stats.action_diversity > kLowDiversityThreshold)
    {
        epsilon = std::max(0.10, epsilon * 0.999);  // Slower decay
    }
    else
    {
        epsilon = std::min(0.30, epsilon * 1.01);   // INCREASE epsilon if collapsing
    }

    return stats;
}

// Print training statistics
inline void print_epoch_stats(int epoch, const EpochStats &stats, double epsilon, float temperature)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Epoch " << std::setw(4) << epoch
              << " | R: " << std::setw(7) << stats.mean_reward
              << " +/- " << std::setw(6) << stats.std_reward
              << " | Win: " << std::setw(5) << (stats.win_rate * 100) << "%"
              << " | Steps: " << std::setw(5) << stats.total_steps
              << " | H: " << std::setw(4) << stats.mean_entropy
              << " | D: " << std::setw(4) << stats.action_diversity  // Action diversity
              << " | e: " << std::setw(5) << epsilon
              << " | T: " << std::setw(4) << temperature
              << " | [F/C/H/P/A]: ";

    for (int i = 0; i < kNumActions; ++i)
    {
        std::cout << stats.action_counts[i];
        if (i < kNumActions - 1)
            std::cout << "/";
    }
    std::cout << std::endl;
}

#endif // __POKER_RL_TRAINER_HPP__
