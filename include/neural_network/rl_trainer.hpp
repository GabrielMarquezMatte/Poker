#ifndef __POKER_RL_TRAINER_HPP__
#define __POKER_RL_TRAINER_HPP__
#include <random>
#include <dlib/dnn.h>
#include "dlib_policy.hpp"
#include "rl_featurizer.hpp"
#include "rl_actions.hpp"

struct TrajStep
{
    dlib::matrix<float> s;
    unsigned a;
    std::size_t actor = 0;
    float R = 0.f;
};

template <class TRng>
float play_one_hand_collect(Game &g, TRng &rng, policy_net &net,
                            const Blinds &blinds,
                            std::vector<TrajStep> &out_steps,
                            double epsilon)
{
    g.startNewHand(rng);
    const auto n = g.players().size();

    std::vector<std::uint32_t> chips_before(n);
    for (auto const &p : g.players())
        chips_before[p.id] = p.chips;

    std::vector<std::vector<TrajStep>> per_actor(n);

    static thread_local omp::XoroShiro128Plus exrng{123};
    std::uniform_real_distribution<double> U(0.0, 1.0);

    while (g.state() != GameState::Finished)
    {
        if (!g.hasCurrentActor())
        {
            g.applyAction(rng, ActionStruct{ActionType::Check, 0});
            continue;
        }

        auto cur = g.currentPlayer().id;
        auto s = featurize(exrng, g, cur, blinds);
        auto leg = legal_actions(g, cur, blinds);
        if (leg.empty())
        {
            g.applyAction(rng, ActionStruct{ActionType::Fold, 0});
            continue;
        }

        unsigned aidx;
        if (U(exrng) < epsilon)
        {
            omp::FastUniformIntDistribution<size_t> D(0, leg.size() - 1);
            aidx = leg[D(exrng)];
        }
        else
        {
            aidx = policy_greedy(net, s, leg);
        }

        per_actor[cur].push_back({s, aidx, cur, 0.f});
        g.applyAction(rng, to_engine_action(aidx, g, cur, blinds));
    }

    const float bb = float(std::max<std::uint32_t>(1, blinds.bigBlind));
    float table_mean_R = 0.f;

    for (std::size_t i = 0; i < n; ++i)
    {
        float delta_bb = (float(g.players()[i].chips) - float(chips_before[i])) / bb;
        table_mean_R += delta_bb;
        for (auto &st : per_actor[i])
        {
            st.R = delta_bb;
        }
    }
    table_mean_R /= std::max<std::size_t>(1, n);

    for (auto &vec : per_actor)
    {
        out_steps.insert(out_steps.end(),
                         std::make_move_iterator(vec.begin()),
                         std::make_move_iterator(vec.end()));
    }
    return table_mean_R;
}

inline void build_weighted_supervised_batch(
    const std::vector<TrajStep> &traj,
    const std::vector<float> &seat_baseline, // NEW
    std::vector<dlib::matrix<float>> &X,
    std::vector<unsigned long> &y,
    float entropy_prob = 0.02f, // NEW param (default 2%)
    float dup_scale = 3.0f,     // NEW: scales duplication strength
    int dup_cap = 6)            // NEW: cap
{
    if (traj.empty())
        return;

    // 1) Compute raw advantages per step
    std::vector<float> adv(traj.size());
    for (size_t i = 0; i < traj.size(); ++i)
    {
        const float b = (traj[i].actor < seat_baseline.size()) ? seat_baseline[traj[i].actor] : 0.f;
        adv[i] = std::clamp(traj[i].R, -20.f, 20.f) - b;
    }

    // 2) Normalize advantages (mean 0, std 1) to reduce variance
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
    var /= std::max<size_t>(1, adv.size());
    float sd = std::sqrt(std::max(1e-6f, var));

    for (float &v : adv)
        v = std::clamp((v - mu) / sd, -3.f, 3.f); // clip

    // 3) Build supervised batch via oversampling
    X.reserve(X.size() + traj.size());
    y.reserve(y.size() + traj.size());

    for (size_t i = 0; i < traj.size(); ++i)
    {
        const auto &t = traj[i];
        const float a = adv[i];

        int mult = (a > 0.f) ? int(std::round(std::min(dup_cap * 1.f, a * dup_scale)))
                             : -int(std::round(std::min(dup_cap * 1.f, (-a) * dup_scale)));

        if (mult > 0)
        {
            for (int k = 0; k < mult; ++k)
            {
                X.push_back(t.s);
                y.push_back(t.a);
            }
        }
        else if (mult < 0)
        {
            unsigned wrong = (t.a + 1) % kNumActions;
            X.push_back(t.s);
            y.push_back(wrong);
        }
        else
        {
            if ((rand() % int(std::round(1.0f / entropy_prob))) == 0)
            {
                X.push_back(t.s);
                y.push_back(rand() % kNumActions);
            }
        }
    }
}

template <class TRng>
void train_reinforce_epoch(policy_net &net, Game &g, TRng &rng, const Blinds &blinds,
                           int hands_per_epoch,
                           std::vector<float> &seat_baseline, // NEW
                           double &epsilon)                   // NEW
{
    std::vector<TrajStep> traj;
    traj.reserve(4096);

    // Collect
    for (int h = 0; h < hands_per_epoch; ++h)
    {
        play_one_hand_collect(g, rng, net, blinds, traj, epsilon);
    }
    const std::size_t n_players = g.players().size();
    if (seat_baseline.size() != n_players)
    {
        seat_baseline.assign(n_players, 0.f);
    }

    std::vector<double> sumR(n_players, 0.0);
    std::vector<int> cnt(n_players, 0);
    for (auto const &t : traj)
    {
        sumR[t.actor] += t.R;
        cnt[t.actor]++;
    }

    const float alpha = 0.10f;
    for (std::size_t i = 0; i < n_players; ++i)
    {
        if (cnt[i] > 0)
        {
            float meanRi = float(sumR[i] / double(cnt[i]));
            seat_baseline[i] = (1.f - alpha) * seat_baseline[i] + alpha * meanRi;
        }
    }
    std::vector<dlib::matrix<float>> X;
    std::vector<unsigned long> y;
    build_weighted_supervised_batch(traj, seat_baseline, X, y, 0.02f, 3.0f, 6);

    if (X.empty())
    {
        return;
    }
    dlib::dnn_trainer<policy_net> trainer(net);
    trainer.set_learning_rate(1e-3);
    trainer.set_min_learning_rate(1e-5);
    trainer.set_mini_batch_size(256);
    trainer.set_iterations_without_progress_threshold(500);
    trainer.be_verbose();
    trainer.train(X, y);
    epsilon = std::max(0.02, epsilon * 0.99);
}

#endif // __POKER_RL_TRAINER_HPP__