#ifndef __POKER_CFR_MCCFR_HPP__
#define __POKER_CFR_MCCFR_HPP__
#include "../random.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

using CfrRng = omp::XoroShiro128Plus;

// Game contract for the solver. State is a cheap value type; actions are indices in [0, numActions(s)).
// utility(s, p) is the payoff for player p at a terminal state.
template <typename G>
concept CfrGame = requires(const typename G::State &s, std::size_t action, std::size_t player, CfrRng &rng) {
    { G::numPlayers } -> std::convertible_to<std::size_t>;
    { G::maxActions } -> std::convertible_to<std::size_t>;
    { G::initial() } -> std::same_as<typename G::State>;
    { G::isTerminal(s) } -> std::same_as<bool>;
    { G::utility(s, player) } -> std::same_as<double>;
    { G::isChance(s) } -> std::same_as<bool>;
    { G::sampleChance(s, rng) } -> std::same_as<typename G::State>;
    { G::currentPlayer(s) } -> std::same_as<std::size_t>;
    { G::numActions(s) } -> std::same_as<std::size_t>;
    { G::apply(s, action) } -> std::same_as<typename G::State>;
    { G::infosetKey(s) } -> std::same_as<std::uint64_t>;
};

// External-sampling MCCFR with Linear CFR weighting (iteration t weighs regrets and strategy by t).
template <CfrGame G>
class Mccfr
{
public:
    using Strategy = std::array<double, G::maxActions>;
    struct Node
    {
        Strategy regretSum{};
        Strategy strategySum{};
    };

    inline void train(std::uint64_t iterations, CfrRng &rng)
    {
        for (std::uint64_t i = 0; i < iterations; ++i)
        {
            ++m_iteration;
            for (std::size_t p = 0; p < G::numPlayers; ++p)
            {
                traverse(G::initial(), p, rng);
            }
        }
    }

    // Uniform for infosets never visited.
    inline Strategy averageStrategy(std::uint64_t key, std::size_t numActions) const
    {
        const auto it = m_nodes.find(key);
        return normalize(it == m_nodes.end() ? Strategy{} : it->second.strategySum, numActions);
    }

    inline std::size_t numInfosets() const noexcept { return m_nodes.size(); }
    inline std::uint64_t iterations() const noexcept { return m_iteration; }

private:
    // ponytail: unordered_map + double arrays; switch to a flat hash table / float when NLHE memory matters.
    std::unordered_map<std::uint64_t, Node> m_nodes;
    std::uint64_t m_iteration = 0;

    // Positive part normalized; uniform when nothing is positive. Serves as regret matching and averaging.
    static inline Strategy normalize(const Strategy &values, std::size_t n) noexcept
    {
        Strategy out{};
        double total = 0.0;
        for (std::size_t a = 0; a < n; ++a)
        {
            total += std::max(values[a], 0.0);
        }
        for (std::size_t a = 0; a < n; ++a)
        {
            out[a] = total > 0.0 ? std::max(values[a], 0.0) / total : 1.0 / static_cast<double>(n);
        }
        return out;
    }

    static inline std::size_t sample(const Strategy &sigma, std::size_t n, CfrRng &rng) noexcept
    {
        double r = static_cast<double>(rng() >> 11) * 0x1.0p-53;
        for (std::size_t a = 0; a + 1 < n; ++a)
        {
            r -= sigma[a];
            if (r < 0.0)
            {
                return a;
            }
        }
        return n - 1;
    }

    double traverse(const typename G::State &s, std::size_t traverser, CfrRng &rng)
    {
        if (G::isTerminal(s))
        {
            return G::utility(s, traverser);
        }
        if (G::isChance(s))
        {
            return traverse(G::sampleChance(s, rng), traverser, rng);
        }
        const std::size_t n = G::numActions(s);
        Node &node = m_nodes[G::infosetKey(s)]; // references survive rehash
        const Strategy sigma = normalize(node.regretSum, n);
        const double weight = static_cast<double>(m_iteration);
        if (G::currentPlayer(s) != traverser)
        {
            for (std::size_t a = 0; a < n; ++a)
            {
                node.strategySum[a] += weight * sigma[a];
            }
            return traverse(G::apply(s, sample(sigma, n, rng)), traverser, rng);
        }
        Strategy values{};
        double nodeValue = 0.0;
        for (std::size_t a = 0; a < n; ++a)
        {
            values[a] = traverse(G::apply(s, a), traverser, rng);
            nodeValue += sigma[a] * values[a];
        }
        for (std::size_t a = 0; a < n; ++a)
        {
            node.regretSum[a] += weight * (values[a] - nodeValue);
        }
        return nodeValue;
    }
};
#endif // __POKER_CFR_MCCFR_HPP__
