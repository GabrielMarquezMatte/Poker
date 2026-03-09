#ifndef __POKER_DLIB_POLICY_HPP__
#define __POKER_DLIB_POLICY_HPP__

#include <dlib/dnn.h>
#include <vector>
#include <algorithm>
#include <span>
#include <cmath>
#include <random>
#include <utility>

constexpr int kInputDims  = 32;
constexpr int kNumActions = 5;  // Fold, Check/Call, ½pot, Pot, All-in

// ──────────────────────────── Network architectures ───────────────────────────
//
//  Policy:  input(32) → FC(512) → ReLU → FC(256) → ReLU → FC(128) → ReLU → FC(5)
//  Value:   input(32) → FC(128) → ReLU → FC(64)  → ReLU → FC(1)
//
// The _body aliases name the hidden layers so the loss/output wrappers are
// visible at a glance when reading the full network type.

using policy_body = dlib::relu<dlib::fc<128,
                   dlib::relu<dlib::fc<256,
                   dlib::relu<dlib::fc<512,
                   dlib::input<dlib::matrix<float>>>>>>>>;

using policy_net = dlib::loss_multiclass_log<dlib::fc<kNumActions, policy_body>>;

using value_body = dlib::relu<dlib::fc<64,
                  dlib::relu<dlib::fc<128,
                  dlib::input<dlib::matrix<float>>>>>>;

using value_net = dlib::loss_mean_squared<dlib::fc<1, value_body>>;

// ───────────────────────────── Inference helpers ──────────────────────────────

// Runs one sample through `net` and returns the subnet's output tensor.
// Both get_action_logits and predict_value delegate here to avoid repeating
// the to_tensor → forward → get_output boilerplate.
template <typename Net>
inline const dlib::tensor& forward_single(Net &net, const dlib::matrix<float> &s)
{
    std::vector<dlib::matrix<float>> input = {s};
    dlib::resizable_tensor buf;
    net.to_tensor(input.data(), input.data() + 1, buf);
    net.subnet().forward(buf);
    return net.subnet().get_output();
}

// Returns the raw pre-softmax logits for all kNumActions outputs.
inline std::vector<float> get_action_logits(policy_net &net, const dlib::matrix<float> &s)
{
    const auto &out = forward_single(net, s);
    std::vector<float> logits(kNumActions);
    for (int i = 0; i < kNumActions; ++i)
        logits[i] = out.host()[i];
    return logits;
}

// Returns the scalar value estimate V(s).
inline float predict_value(value_net &vnet, const dlib::matrix<float> &s)
{
    const auto &out = forward_single(vnet, s);
    return (out.size() > 0) ? out.host()[0] : 0.f;
}

// ───────────────────────────── Probability helpers ────────────────────────────

// Softmax over `legal` actions only; illegal action slots are set to 0.
// Subtracts the max logit for numerical stability before exponentiating.
inline std::vector<float> softmax_legal(const std::span<const float>   logits,
                                        const std::span<const unsigned> legal)
{
    std::vector<float> probs(kNumActions, 0.f);
    if (legal.empty()) return probs;

    float max_logit = -1e30f;
    for (unsigned a : legal) max_logit = std::max(max_logit, logits[a]);

    float sum = 0.f;
    for (unsigned a : legal) { probs[a] = std::exp(logits[a] - max_logit); sum += probs[a]; }

    if (sum > 1e-8f)
        for (unsigned a : legal) probs[a] /= sum;
    else  // degenerate: all logits underflowed — fall back to uniform
        for (unsigned a : legal) probs[a] = 1.f / static_cast<float>(legal.size());

    return probs;
}

// Entropy H(p) in nats.  Zero-probability entries are skipped to avoid log(0).
inline float compute_entropy(const std::span<const float> probs)
{
    float h = 0.f;
    for (float p : probs)
        if (p > 1e-8f) h -= p * std::log(p);
    return h;
}

// ────────────────────────────── Sampling helpers ──────────────────────────────

// Samples an action from the masked policy distribution.
template <class TRng>
inline unsigned policy_sample(policy_net &net,
                              const dlib::matrix<float> &s,
                              const std::span<const unsigned> legal,
                              TRng &rng,
                              float temperature = 1.0f)
{
    if (legal.empty()) return 0u;

    auto logits = get_action_logits(net, s);
    if (temperature != 1.0f)
        for (auto &l : logits) l /= temperature;

    const auto probs = softmax_legal(logits, legal);

    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float r = U(rng);
    float cumsum = 0.f;
    for (unsigned a : legal)
    {
        cumsum += probs[a];
        if (r <= cumsum) return a;
    }
    return legal.back();
}

// Samples an action and returns the full probability vector in one forward pass,
// avoiding the redundant call that policy_sample + get_action_probs would require.
template <class TRng>
inline std::pair<unsigned, std::vector<float>> policy_sample_with_probs(
    policy_net &net,
    const dlib::matrix<float> &s,
    const std::span<const unsigned> legal,
    TRng &rng,
    float temperature = 1.0f)
{
    if (legal.empty())
        return {0u, std::vector<float>(kNumActions, 0.f)};

    auto logits = get_action_logits(net, s);
    if (temperature != 1.0f)
        for (auto &l : logits) l /= temperature;

    auto probs = softmax_legal(logits, legal);

    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float r = U(rng);
    float cumsum = 0.f;
    unsigned action = legal.back();
    for (unsigned a : legal)
    {
        cumsum += probs[a];
        if (r <= cumsum) { action = a; break; }
    }
    return {action, std::move(probs)};
}

// Returns the action probability vector without sampling.
inline std::vector<float> get_action_probs(policy_net &net,
                                           const dlib::matrix<float> &s,
                                           const std::span<const unsigned> legal)
{
    return softmax_legal(get_action_logits(net, s), legal);
}

// Greedy action: the legal action with the highest probability.
inline unsigned policy_greedy(policy_net &net,
                              const dlib::matrix<float> &s,
                              const std::span<const unsigned> legal)
{
    if (legal.empty()) return 0u;
    const auto probs = get_action_probs(net, s, legal);
    return *std::max_element(legal.begin(), legal.end(),
                             [&](unsigned a, unsigned b){ return probs[a] < probs[b]; });
}

#endif // __POKER_DLIB_POLICY_HPP__
