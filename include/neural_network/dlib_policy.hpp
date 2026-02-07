#ifndef __POKER_DLIB_POLICY_HPP__
#define __POKER_DLIB_POLICY_HPP__

#include <dlib/dnn.h>
#include <vector>
#include <algorithm>
#include <span>
#include <cmath>
#include <random>

constexpr int kInputDims = 32; // Increased for richer features
constexpr int kNumActions = 5; // Fold, Check/Call, 1/2 pot, pot, all-in

// Policy network: 32 -> 256 -> 128 -> 64 -> 5
using policy_net = dlib::loss_multiclass_log<
    dlib::fc<kNumActions,
             dlib::relu<dlib::fc<128,
                                 dlib::relu<dlib::fc<256,
                                                     dlib::relu<dlib::fc<512,
                                                                         dlib::input<dlib::matrix<float>>>>>>>>>>;

// Value network: 32 -> 128 -> 64 -> 1
using value_net = dlib::loss_mean_squared<
    dlib::fc<1,
             dlib::relu<dlib::fc<64,
                                 dlib::relu<dlib::fc<128,
                                                     dlib::input<dlib::matrix<float>>>>>>>>;

// Extract logits (pre-softmax) from the network for a given input
inline std::vector<float> get_action_logits(policy_net &net, const dlib::matrix<float> &s)
{
    auto &subnet = net.subnet();
    std::vector<dlib::matrix<float>> inputs = {s};
    dlib::resizable_tensor input_tensor;
    net.to_tensor(&inputs[0], &inputs[0] + 1, input_tensor);
    subnet.forward(input_tensor);

    const auto &out = subnet.get_output();
    std::vector<float> logits(kNumActions);
    for (int i = 0; i < kNumActions; ++i)
    {
        logits[i] = out.host()[i];
    }
    return logits;
}

// Convert logits to probabilities via softmax, masking illegal actions
inline std::vector<float> softmax_legal(const std::span<const float> logits,
                                        const std::span<const unsigned> legal)
{
    std::vector<float> probs(kNumActions, 0.0f);

    // Find max logit among legal actions for numerical stability
    float max_logit = -1e30f;
    for (unsigned a : legal)
    {
        if (logits[a] > max_logit)
            max_logit = logits[a];
    }

    // Compute exp and sum for legal actions only
    float sum_exp = 0.0f;
    for (unsigned a : legal)
    {
        probs[a] = std::exp(logits[a] - max_logit);
        sum_exp += probs[a];
    }

    // Normalize
    if (sum_exp > 1e-8f)
    {
        for (unsigned a : legal)
        {
            probs[a] /= sum_exp;
        }
        return probs;
    }
    if (!legal.empty())
    {
        float uniform = 1.0f / static_cast<float>(legal.size());
        for (unsigned a : legal)
        {
            probs[a] = uniform;
        }
    }

    return probs;
}

// Compute entropy of action distribution (for exploration bonus)
inline float compute_entropy(const std::span<const float> probs)
{
    float entropy = 0.0f;
    for (float p : probs)
    {
        if (p > 1e-8f)
        {
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

// Sample from policy distribution (for training/exploration)
template <class TRng>
inline unsigned policy_sample(policy_net &net,
                              const dlib::matrix<float> &s,
                              const std::span<const unsigned> legal,
                              TRng &rng,
                              float temperature = 1.0f)
{
    if (legal.empty())
        return 0u;

    auto logits = get_action_logits(net, s);

    // Apply temperature
    if (temperature != 1.0f)
    {
        for (auto &l : logits)
            l /= temperature;
    }

    auto probs = softmax_legal(logits, legal);

    // Sample from categorical distribution
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    float r = U(rng);
    float cumsum = 0.0f;
    for (unsigned a : legal)
    {
        cumsum += probs[a];
        if (r <= cumsum)
            return a;
    }
    return legal.back();
}

// Get action probabilities for a state (useful for logging)
inline std::vector<float> get_action_probs(policy_net &net,
                                           const dlib::matrix<float> &s,
                                           const std::span<const unsigned> legal)
{
    auto logits = get_action_logits(net, s);
    return softmax_legal(logits, legal);
}

// Greedy policy (for evaluation)
inline unsigned policy_greedy(policy_net &net,
                              const dlib::matrix<float> &s,
                              const std::span<const unsigned> legal)
{
    if (legal.empty())
        return 0u;

    auto probs = get_action_probs(net, s, legal);

    unsigned best = legal[0];
    float best_prob = probs[best];
    for (unsigned a : legal)
    {
        if (probs[a] > best_prob)
        {
            best_prob = probs[a];
            best = a;
        }
    }
    return best;
}

// Predict value from value network
inline float predict_value(value_net &vnet, const dlib::matrix<float> &s)
{
    // Do forward pass through subnet to get the actual value
    auto &subnet = vnet.subnet();
    std::vector<dlib::matrix<float>> inputs = {s};
    dlib::resizable_tensor input_tensor;
    vnet.to_tensor(&inputs[0], &inputs[0] + 1, input_tensor);
    subnet.forward(input_tensor);
    const auto &out = subnet.get_output();
    if (out.size() > 0)
    {
        return out.host()[0];
    }
    return 0.0f;
}

#endif // __POKER_DLIB_POLICY_HPP__
