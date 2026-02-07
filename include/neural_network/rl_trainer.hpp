#ifndef __POKER_RL_TRAINER_HPP__
#define __POKER_RL_TRAINER_HPP__
#include <random>
#include <dlib/dnn.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
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
};

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
                            float temperature = 1.0f,
                            bool self_play = true,
                            std::size_t hero_id = 0)  // Only collect for this player if not self_play
{
    g.startNewHand(rng);
    const auto n = g.players().size();

    std::vector<std::uint32_t> chips_before(n);
    for (auto const &p : g.players())
        chips_before[p.id] = p.chips;

    std::vector<std::vector<TrajStep>> per_actor(n);

    static thread_local omp::XoroShiro128Plus exrng{std::random_device{}()};
    std::uniform_real_distribution<double> U(0.0, 1.0);

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
        bool use_policy = self_play || (cur == hero_id);
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
            aidx = policy_sample(net, s, leg, exrng, temperature);
            auto probs = get_action_probs(net, s, leg);
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
    // Compute returns and advantages
    for (std::size_t i = 0; i < n; ++i)
    {
        float delta_bb = (static_cast<float>(g.players()[i].chips) - static_cast<float>(chips_before[i])) / bb;
        for (auto &st : per_actor[i])
        {
            st.R = delta_bb;
            st.advantage = delta_bb - st.value;  // Simple advantage: R - V(s)
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

    // Count actions to detect collapse
    std::array<int, kNumActions> action_counts = {};
    for (const auto &t : traj)
        action_counts[t.a]++;
    
    int total_actions = static_cast<int>(traj.size());

    // Compute normalized advantages with entropy bonus and all-in penalty
    std::vector<float> adv(traj.size());
    for (size_t i = 0; i < traj.size(); ++i)
    {
        adv[i] = traj[i].advantage;
        
        // Add entropy bonus to advantage (encourages exploration)
        adv[i] += config.entropy_coef * traj[i].entropy;
        if (traj[i].a != 4) // Not all-in
        {
            continue;
        }
        // Penalize all-in if it's being overused
        float allin_ratio = static_cast<float>(action_counts[4]) / static_cast<float>(total_actions) * 100.f;
        if (allin_ratio > static_cast<float>(config.max_allin_ratio))
        {
            adv[i] -= config.allin_penalty * (allin_ratio - config.max_allin_ratio) / 100.f;
        }
    }

    // Normalize advantages (mean 0, std 1)
    float mu = 0.f;
    for (float v : adv)
        mu += v;
    mu /= static_cast<float>(adv.size());

    float var = 0.f;
    for (float v : adv)
    {
        float d = v - mu;
        var += d * d;
    }
    var /= static_cast<float>(std::max<size_t>(1, adv.size()));
    float sd = std::sqrt(std::max(1e-6f, var));

    for (float &v : adv)
        v = (v - mu) / sd;

    // Build batches - INCLUDE ALL SAMPLES to prevent collapse
    X_policy.reserve(X_policy.size() + traj.size() * 2);
    y_policy.reserve(y_policy.size() + traj.size() * 2);
    X_value.reserve(X_value.size() + traj.size());
    y_value.reserve(y_value.size() + traj.size());
    advantages_out.reserve(advantages_out.size() + traj.size() * 2);

    for (size_t i = 0; i < traj.size(); ++i)
    {
        const auto &t = traj[i];
        const float a = adv[i];

        // Include ALL samples for policy learning (key change!)
        X_policy.push_back(t.s);
        y_policy.push_back(t.a);
        advantages_out.push_back(a);

        // For positive advantage, add extra samples (but limit to prevent overfit)
        if (a > 0.5f)
        {
            X_policy.push_back(t.s);
            y_policy.push_back(t.a);
            advantages_out.push_back(a * 0.5f);
        }

        // For value network: always include all samples
        X_value.push_back(t.s);
        y_value.push_back(t.R);
    }
    
    // Add diversity-encouraging samples for underrepresented actions
    for (size_t i = 0; i < traj.size(); ++i)
    {
        const auto &t = traj[i];
        unsigned action = t.a;
        
        // If this action is rare but was successful, reinforce it more
        float action_freq = static_cast<float>(action_counts[action]) / static_cast<float>(total_actions);
        if (action_freq < 0.1f && t.R > 0.f && action != 4)  // Rare non-allin winning action
        {
            X_policy.push_back(t.s);
            y_policy.push_back(t.a);
            advantages_out.push_back(std::max(0.5f, adv[i]));
        }
    }
}

template <class TRng>
EpochStats train_epoch(policy_net &net, value_net &vnet, Game &g, TRng &rng,
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
        // Reset chips if needed
        bool needs_reset = false;
        for (const auto &p : g.players())
        {
            if (p.chips < blinds.bigBlind * 3)
            {
                needs_reset = true;
                break;
            }
        }
        if (needs_reset)
        {
            g.resetPlayerChips(starting_chips);
        }

        float reward = play_one_hand_collect(g, rng, net, vnet, blinds, traj, epsilon, pool, 
                                             temperature, config.self_play, 0);
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
    if (stats.action_diversity < 0.3f)  // Low diversity - reduce learning rate
    {
        policy_lr *= 0.5f;
    }

    {
        dlib::dnn_trainer<policy_net> trainer(net, dlib::sgd(0.0005, 0.9));
        trainer.set_learning_rate(policy_lr);
        trainer.set_min_learning_rate(1e-6);
        trainer.set_mini_batch_size(std::min<size_t>(256, X_policy.size()));
        trainer.set_max_num_epochs(1);
        trainer.be_quiet();
        trainer.set_iterations_without_progress_threshold(200);
        trainer.train(X_policy, y_policy);
    }

    // Train value network
    {
        dlib::dnn_trainer<value_net> trainer(vnet, dlib::sgd(0.0005, 0.9));
        trainer.set_learning_rate(5e-4);
        trainer.set_min_learning_rate(1e-6);
        trainer.set_mini_batch_size(std::min<size_t>(256, X_value.size()));
        trainer.set_max_num_epochs(1);
        trainer.be_quiet();
        trainer.set_iterations_without_progress_threshold(200);
        trainer.train(X_value, y_value);
    }

    // Adaptive epsilon decay based on action diversity
    // Decay slower if actions are collapsing (need more exploration)
    if (stats.action_diversity > 0.5f)
    {
        epsilon = std::max(0.05, epsilon * 0.997);  // Normal decay
    }
    else if (stats.action_diversity > 0.3f)
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
