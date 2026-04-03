#ifndef __POKER_DLIB_POLICY_HPP__
#define __POKER_DLIB_POLICY_HPP__

#include <dlib/dnn.h>
#include <vector>
#include <algorithm>
#include <span>
#include <cmath>
#include <random>
#include <utility>
#include <array>

constexpr int kInputDims  = 32;
constexpr int kNumActions = 5;  // Fold, Check/Call, ½pot, Pot, All-in

// ─────────────────────────── PPO label type ───────────────────────────────────
//
// Bundles everything the clipped surrogate objective needs for one sample.
//   action      — index of the action that was taken (0..kNumActions-1)
//   old_log_prob — log π_θ_old(a|s) recorded at trajectory collection time
//   advantage   — normalized advantage A (post entropy-bonus adjustment)

struct PPOLabel {
    unsigned action;
    float    old_log_prob;
    float    advantage;
};

// ──────────────────────────── Custom PPO loss layer ───────────────────────────
//
// Implements the PPO clipped surrogate objective (Schulman et al. 2017):
//   J_CLIP = E[ min( ρ·A, clip(ρ, 1−ε, 1+ε)·A ) ]   ρ = π_new/π_old
//
// We minimise −J_CLIP.  Gradient derivation:
//   When unclipped: d(-J)/d(logit_j) = A·ρ · (softmax_j − δ_{j,a})
//   When clipped:   gradient is zero (trust region boundary reached).

class loss_ppo_
{
public:
    static constexpr float kClipEpsilon = 0.2f;

    // dlib uses training_label_type for the trainer's train() call,
    // output_label_type for the inference operator()'s return type, and
    // label_type as the canonical alias.  All three must agree for dlib's
    // template meta-programming to dispatch correctly.
    typedef PPOLabel label_type;
    typedef PPOLabel training_label_type;
    typedef PPOLabel output_label_type;

    // Inference: populate labels with a dummy PPOLabel whose action is the greedy argmax.
    // In practice we always use forward_single() + get_action_logits() for inference;
    // this exists only to satisfy the dlib add_loss_layer interface contract.
    // OutputIterator is deduced to match whatever iterator dlib passes (PPOLabel*, etc.).
    template <typename SUBNET, typename OutputIterator>
    void to_label(const dlib::tensor& /*input*/, const SUBNET& sub, OutputIterator label) const
    {
        const auto& out = sub.get_output();
        const long  na  = out.k();
        for (long i = 0; i < out.num_samples(); ++i)
        {
            const float* li = out.host() + i * na;
            const long greedy = std::max_element(li, li + na) - li;
            *label++ = PPOLabel{static_cast<unsigned>(greedy), 0.f, 0.f};
        }
    }

    // Forward + backward: compute loss value and fill gradient tensor.
    template <typename const_label_iterator, typename SUBNET>
    double compute_loss_value_and_gradient(
        const dlib::tensor& /*input*/,
        const_label_iterator truth,
        SUBNET& sub) const
    {
        auto&       out  = sub.get_output();
        auto&       g    = sub.get_gradient_input();
        const long  B    = out.num_samples();
        const long  NA   = out.k();

        const float* logits = out.host();
        float*       grad   = g.host_write_only();
        std::fill(grad, grad + g.size(), 0.f);

        double total_loss = 0.0;

        for (long i = 0; i < B; ++i, ++truth)
        {
            const float* li = logits + i * NA;
            float*       gi = grad   + i * NA;

            const auto&  lbl = *truth;
            const long   a   = static_cast<long>(lbl.action);
            const float  A   = lbl.advantage;
            const float  olp = lbl.old_log_prob;

            // Numerically stable softmax over all action logits.
            float max_l = *std::max_element(li, li + NA);
            float sum_e = 0.f;
            std::array<float, kNumActions> sm{};
            for (long j = 0; j < NA; ++j) { sm[j] = std::exp(li[j] - max_l); sum_e += sm[j]; }
            for (auto& v : sm) { v /= sum_e; }

            // Importance ratio ρ = π_new(a|s) / π_old(a|s).
            // Clamp the log-ratio to ±5 to prevent fp explosion on initial updates.
            const float new_lp = std::log(std::max(sm[a], 1e-8f));
            const float rho    = std::exp(std::clamp(new_lp - olp, -5.f, 5.f));

            // Clipped surrogate loss contribution and gradient scaling.
            // g_scale = A·ρ when unclipped, 0 when clipped.
            float g_scale = 0.f;
            if (A >= 0.f)
            {
                if (rho < 1.f + kClipEpsilon)
                {
                    total_loss -= static_cast<double>(rho * A);
                    g_scale     = A * rho;
                }
                else
                {
                    total_loss -= static_cast<double>((1.f + kClipEpsilon) * A);
                    // g_scale stays 0: clipped, no gradient
                }
            }
            else
            {
                if (rho > 1.f - kClipEpsilon)
                {
                    total_loss -= static_cast<double>(rho * A);
                    g_scale     = A * rho;
                }
                else
                {
                    total_loss -= static_cast<double>((1.f - kClipEpsilon) * A);
                    // g_scale stays 0: clipped, no gradient
                }
            }

            // ∂(−J_CLIP)/∂(logit_j) = g_scale · (softmax_j − δ_{j,a})
            for (long j = 0; j < NA; ++j) { gi[j] = g_scale * sm[j]; }
            gi[a] -= g_scale;
        }

        return total_loss / static_cast<double>(B);
    }

    // No-label overload: called by dlib's inference-mode gradient path (no supervision).
    // For PPO this path is never used meaningfully — zero the gradient and return 0.
    template <typename SUBNET>
    double compute_loss_value_and_gradient(
        const dlib::tensor& /*input*/,
        SUBNET& sub) const
    {
        auto& g    = sub.get_gradient_input();
        float* grad = g.host_write_only();
        std::fill(grad, grad + g.size(), 0.f);
        return 0.0;
    }

    // No trainable parameters — serialize/deserialize are no-ops.
    friend void         serialize(const loss_ppo_&, std::ostream&)        {}
    friend void         deserialize(loss_ppo_&, std::istream&)            {}
    friend std::ostream& operator<<(std::ostream& s, const loss_ppo_&)   { return s << "loss_ppo"; }
    friend void         to_xml(const loss_ppo_&, std::ostream& s)         { s << "<loss_ppo/>"; }
};

template <typename SUBNET>
using loss_ppo = dlib::add_loss_layer<loss_ppo_, SUBNET>;

// ──────────────────────────── Network architectures ───────────────────────────
//
//  Policy:  input(32) → FC(512) → ReLU → FC(256) → ReLU → FC(128) → ReLU → FC(5)
//           Loss: PPO clipped surrogate (loss_ppo)
//  Value:   input(32) → FC(128) → ReLU → FC(64)  → ReLU → FC(1)
//           Loss: mean squared error

using policy_body = dlib::relu<dlib::fc<128,
                   dlib::relu<dlib::fc<256,
                   dlib::relu<dlib::fc<512,
                   dlib::input<dlib::matrix<float>>>>>>>>;

using policy_net = loss_ppo<dlib::fc<kNumActions, policy_body>>;

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
    std::array<dlib::matrix<float>, 1> input = {s};
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
    {
        logits[i] = out.host()[i];
    }
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
    if (legal.empty()) { return probs; }

    float max_logit = -1e30f;
    for (unsigned a : legal) { max_logit = std::max(max_logit, logits[a]); }

    float sum = 0.f;
    for (unsigned a : legal) { probs[a] = std::exp(logits[a] - max_logit); sum += probs[a]; }

    if (sum > 1e-8f)
    {
        for (unsigned a : legal) { probs[a] /= sum; }
    }
    else  // degenerate: all logits underflowed — fall back to uniform
    {
        for (unsigned a : legal) { probs[a] = 1.f / static_cast<float>(legal.size()); }
    }

    return probs;
}

// Entropy H(p) in nats.  Zero-probability entries are skipped to avoid log(0).
inline float compute_entropy(const std::span<const float> probs)
{
    float h = 0.f;
    for (float p : probs)
    {
        if (p > 1e-8f) { h -= p * std::log(p); }
    }
    return h;
}

// ────────────────────────────── Sampling helpers ──────────────────────────────

static inline constexpr void apply_temperature(std::span<float> logits, float temperature) noexcept
{
    if (temperature != 1.0f)
    {
        for (auto &l : logits) { l /= temperature; }
    }
}

// Samples an action from the masked policy distribution.
template <class TRng>
inline unsigned policy_sample(policy_net &net,
                              const dlib::matrix<float> &s,
                              const std::span<const unsigned> legal,
                              TRng &rng,
                              float temperature = 1.0f)
{
    if (legal.empty()) { return 0u; }

    auto logits = get_action_logits(net, s);
    apply_temperature(logits, temperature);

    const auto probs = softmax_legal(logits, legal);

    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float r = U(rng);
    float cumsum = 0.f;
    for (unsigned a : legal)
    {
        cumsum += probs[a];
        if (r <= cumsum) { return a; }
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
    {
        return {0u, std::vector<float>(kNumActions, 0.f)};
    }

    auto logits = get_action_logits(net, s);
    apply_temperature(logits, temperature);

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
    if (legal.empty()) { return 0u; }
    const auto probs = get_action_probs(net, s, legal);
    return *std::max_element(legal.begin(), legal.end(),
                             [&](unsigned a, unsigned b){ return probs[a] < probs[b]; });
}

#endif // __POKER_DLIB_POLICY_HPP__
