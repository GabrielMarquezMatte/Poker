#ifndef __POKER_CFR_RIVER_SOLVER_HPP__
#define __POKER_CFR_RIVER_SOLVER_HPP__
#include "push_fold.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

// Resolves a river subgame with exact hands (no card abstraction): vector-form Discounted CFR
// (alpha 1.5, beta 0, gamma 2) over the public river betting tree of Hunl<C>, from both players' ranges.
// Hands blocked by the board are dropped and the rest are stored densely, weakest first, so every
// per-hand loop is contiguous. Card removal is exact; showdowns cost O(hands) via per-card sums.
template <typename C>
class RiverSolver
{
public:
    using G = Hunl<C>;
    using Hands = std::array<double, holeCombos>;
    using Strategy = std::array<double, G::maxActions>;

    // `root` is the river's first decision; ranges[p] holds player p's reach for every hand.
    RiverSolver(const typename G::State &root, const std::array<Hands, 2> &ranges)
    {
        std::vector<std::pair<ClassificationResult, std::uint16_t>> ranked;
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if ((holes[h] & root.board) == 0)
            {
                ranked.emplace_back(Hand::classify(Deck::from_mask(holes[h] | root.board)), static_cast<std::uint16_t>(h));
            }
        }
        std::sort(ranked.begin(), ranked.end());
        m_hands = ranked.size();
        m_denseOf.fill(-1);
        for (std::size_t i = 0; i < m_hands; ++i)
        {
            const std::uint16_t h = ranked[i].second;
            if (i == 0 || ranked[i].first != ranked[i - 1].first)
            {
                m_groups.push_back(i);
            }
            m_denseOf[h] = static_cast<int>(i);
            m_low.push_back(static_cast<std::uint8_t>(std::countr_zero(holes[h])));
            m_high.push_back(static_cast<std::uint8_t>(63 - std::countl_zero(holes[h])));
            for (std::size_t p = 0; p < 2; ++p)
            {
                m_ranges[p].push_back(static_cast<float>(ranges[p][h]));
            }
        }
        m_groups.push_back(m_hands);
        const std::size_t depth = build(root, 0);
        m_scratch.resize(depth + 1);
        for (auto &scratch : m_scratch)
        {
            scratch.sigma.resize(G::maxActions * m_hands);
            scratch.values.resize(G::maxActions * m_hands);
            scratch.reach.resize(m_hands);
        }
    }

    void solve(std::size_t iterations)
    {
        std::vector<float> out(m_hands);
        for (std::size_t i = 0; i < iterations; ++i)
        {
            ++m_iteration;
            const double t = static_cast<double>(m_iteration - 1); // discounts apply to sums through t
            m_positiveDiscount = static_cast<float>(std::pow(t, 1.5) / (std::pow(t, 1.5) + 1.0));
            m_strategyDiscount = static_cast<float>(std::pow(t / (t + 1.0), 2.0));
            for (std::size_t p = 0; p < 2; ++p)
            {
                cfr(0, p, m_ranges[1 - p].data(), out.data(), 0);
            }
        }
    }

    // Average strategy at subgame state `s` for the player to act holding hand `hand` (uniform if unreached).
    Strategy strategy(const typename G::State &s, std::size_t hand) const
    {
        const Node &node = m_nodes[m_byHistory.at(s.history)];
        const std::size_t n = node.children.size();
        Strategy out{};
        out.fill(1.0 / static_cast<double>(n));
        if (m_denseOf[hand] < 0)
        {
            return out;
        }
        const std::size_t i = static_cast<std::size_t>(m_denseOf[hand]);
        double total = 0.0;
        for (std::size_t a = 0; a < n; ++a)
        {
            total += m_strategy[node.offset + a * m_hands + i];
        }
        for (std::size_t a = 0; a < n && total > 0.0; ++a)
        {
            out[a] = m_strategy[node.offset + a * m_hands + i] / total;
        }
        return out;
    }

    // Terminal building blocks per hand (0 for hands the board blocks), exposed for tests: opponent
    // reach sharing no card with the hand, and its showdown value for `stake` chips each.
    Hands disjointMass(const Hands &opponentReach) const { return perHand(opponentReach, [&](const float *reach, float *out)
                                                                          { disjointMass(reach, out); }); }
    Hands showdown(const Hands &opponentReach, float stake) const { return perHand(opponentReach, [&](const float *reach, float *out)
                                                                                   { showdown(reach, stake, out); }); }

    inline bool contains(const typename G::State &s) const { return m_byHistory.contains(s.history); }
    inline std::size_t numNodes() const noexcept { return m_nodes.size(); }

    // Average gain in chips of a best response over both players against the average strategies.
    double exploitability() const
    {
        double total = 0.0;
        double joint = 0.0;
        std::vector<float> values(m_hands), mass(m_hands);
        for (std::size_t p = 0; p < 2; ++p)
        {
            bestResponse(0, p, m_ranges[1 - p], values);
            disjointMass(m_ranges[1 - p].data(), mass.data());
            for (std::size_t i = 0; i < m_hands; ++i)
            {
                total += static_cast<double>(m_ranges[p][i]) * values[i];
                joint += p == 0 ? static_cast<double>(m_ranges[p][i]) * mass[i] : 0.0;
            }
        }
        return joint > 0.0 ? total / joint / 2.0 : 0.0;
    }

private:
    struct Node
    {
        typename G::State state;
        std::vector<std::size_t> children;
        std::size_t offset = 0; // into m_regret / m_strategy: [action * m_hands + hand]
    };
    struct Scratch
    {
        std::vector<float> sigma, values, reach;
    };

    std::size_t m_hands = 0;
    std::array<int, holeCombos> m_denseOf{};
    std::vector<std::uint8_t> m_low, m_high; // card indices per dense hand
    std::vector<std::size_t> m_groups;       // start of each equal-strength run, plus the end
    std::array<std::vector<float>, 2> m_ranges;
    std::vector<Node> m_nodes;
    std::unordered_map<std::uint64_t, std::size_t> m_byHistory;
    std::vector<float> m_regret, m_strategy;
    std::vector<Scratch> m_scratch; // one per tree depth
    std::size_t m_iteration = 0;
    float m_positiveDiscount = 0.0f;
    float m_strategyDiscount = 0.0f;

    // Returns the subtree depth.
    std::size_t build(const typename G::State &s, std::size_t depth)
    {
        const std::size_t idx = m_nodes.size();
        m_nodes.push_back({s, {}, 0});
        m_byHistory[s.history] = idx;
        if (G::isTerminal(s))
        {
            return depth;
        }
        const std::size_t n = G::numActions(s);
        m_nodes[idx].offset = m_regret.size();
        m_regret.resize(m_regret.size() + n * m_hands, 0.0f);
        m_strategy.resize(m_strategy.size() + n * m_hands, 0.0f);
        std::vector<std::size_t> children(n);
        std::size_t deepest = depth;
        for (std::size_t a = 0; a < n; ++a)
        {
            children[a] = m_nodes.size();
            deepest = std::max(deepest, build(G::apply(s, a), depth + 1));
        }
        m_nodes[idx].children = std::move(children);
        return deepest;
    }

    // Hot loops copy m_hands to a local and take __restrict pointers: otherwise MSVC assumes float
    // stores may alias the bound or each other and vectorizes none of them.

    // Opponent reach over hands sharing no card with each hand.
    void disjointMass(const float *__restrict reach, float *__restrict out) const
    {
        const std::size_t hands = m_hands;
        const std::uint8_t *__restrict low = m_low.data();
        const std::uint8_t *__restrict high = m_high.data();
        std::array<float, 52> card{};
        float total = 0.0f;
        for (std::size_t i = 0; i < hands; ++i)
        {
            total += reach[i];
            card[low[i]] += reach[i];
            card[high[i]] += reach[i];
        }
        for (std::size_t i = 0; i < hands; ++i)
        {
            out[i] = total - card[low[i]] - card[high[i]] + reach[i];
        }
    }

    // Traverser's value per hand at a terminal, weighted by opponent reach.
    void terminal(const typename G::State &s, std::size_t p, const float *__restrict reach, float *__restrict out) const
    {
        const std::size_t hands = m_hands;
        if (s.folder != G::nobody)
        {
            const float payoff = s.folder == p ? -static_cast<float>(s.invested[p]) : static_cast<float>(s.invested[s.folder]);
            disjointMass(reach, out);
            for (std::size_t i = 0; i < hands; ++i)
            {
                out[i] *= payoff;
            }
            return;
        }
        showdown(reach, static_cast<float>(s.invested[p]), out);
    }

    template <typename F>
    Hands perHand(const Hands &opponentReach, F compute) const
    {
        std::vector<float> reach(m_hands), out(m_hands);
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if (m_denseOf[h] >= 0)
            {
                reach[static_cast<std::size_t>(m_denseOf[h])] = static_cast<float>(opponentReach[h]);
            }
        }
        compute(reach.data(), out.data());
        Hands result{};
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            result[h] = m_denseOf[h] >= 0 ? out[static_cast<std::size_t>(m_denseOf[h])] : 0.0;
        }
        return result;
    }

    // Showdown value per hand for `stake` chips each, in one ascending pass. With W the disjoint opponent
    // mass strictly weaker, W' weaker or tied and D all of it: win - lose = W - (D - W') = W + W' - D.
    // Inclusion-exclusion: a hand adds itself to both of its card sums, so W' and D add it back once.
    // (Per-card hand lists and branch-free prefix sums with indexed reads were both measured slower.)
    void showdown(const float *__restrict reach, float stake, float *__restrict out) const
    {
        const std::size_t hands = m_hands;
        const std::uint8_t *__restrict low = m_low.data();
        const std::uint8_t *__restrict high = m_high.data();
        const std::size_t *__restrict groups = m_groups.data();
        const std::size_t numGroups = m_groups.size() - 1;
        std::array<float, 52> card{};
        float total = 0.0f;
        for (std::size_t g = 0; g < numGroups; ++g)
        {
            const std::size_t begin = groups[g], end = groups[g + 1];
            for (std::size_t i = begin; i < end; ++i)
            {
                out[i] = total - card[low[i]] - card[high[i]];
            }
            for (std::size_t i = begin; i < end; ++i)
            {
                total += reach[i];
                card[low[i]] += reach[i];
                card[high[i]] += reach[i];
            }
            for (std::size_t i = begin; i < end; ++i)
            {
                out[i] += total - card[low[i]] - card[high[i]] + reach[i];
            }
        }
        for (std::size_t i = 0; i < hands; ++i)
        {
            out[i] = stake * (out[i] - (total - card[low[i]] - card[high[i]] + reach[i]));
        }
    }

    // Regret matching per hand into sigma[action * m_hands + hand].
    void currentStrategy(const Node &node, std::size_t n, float *__restrict sigma) const
    {
        const std::size_t hands = m_hands;
        const float *__restrict regret = &m_regret[node.offset];
        float *__restrict total = sigma + (n - 1) * hands; // last row doubles as the running total, filled last
        for (std::size_t i = 0; i < hands; ++i)
        {
            total[i] = 0.0f;
        }
        for (std::size_t a = 0; a < n; ++a)
        {
            const float *__restrict r = regret + a * hands;
            for (std::size_t i = 0; i < hands; ++i)
            {
                total[i] += std::max(r[i], 0.0f);
            }
        }
        const float uniform = 1.0f / static_cast<float>(n);
        for (std::size_t a = 0; a + 1 < n; ++a)
        {
            const float *__restrict r = regret + a * hands;
            float *__restrict out = sigma + a * hands;
            for (std::size_t i = 0; i < hands; ++i)
            {
                out[i] = total[i] > 0.0f ? std::max(r[i], 0.0f) / total[i] : uniform;
            }
        }
        const float *__restrict last = regret + (n - 1) * hands;
        for (std::size_t i = 0; i < hands; ++i)
        {
            total[i] = total[i] > 0.0f ? std::max(last[i], 0.0f) / total[i] : uniform;
        }
    }

    void cfr(std::size_t idx, std::size_t p, const float *__restrict reach, float *__restrict out, std::size_t depth)
    {
        const Node &node = m_nodes[idx];
        if (node.children.empty())
        {
            terminal(node.state, p, reach, out);
            return;
        }
        const std::size_t hands = m_hands;
        const std::size_t n = node.children.size();
        Scratch &scratch = m_scratch[depth];
        float *__restrict sigma = scratch.sigma.data();
        currentStrategy(node, n, sigma);
        for (std::size_t i = 0; i < hands; ++i)
        {
            out[i] = 0.0f;
        }
        if (node.state.toAct == p)
        {
            float *__restrict values = scratch.values.data();
            for (std::size_t a = 0; a < n; ++a)
            {
                float *__restrict child = values + a * hands;
                cfr(node.children[a], p, reach, child, depth + 1);
                const float *__restrict s = sigma + a * hands;
                for (std::size_t i = 0; i < hands; ++i)
                {
                    out[i] += s[i] * child[i];
                }
            }
            float *__restrict regret = &m_regret[node.offset];
            const float positive = m_positiveDiscount;
            for (std::size_t a = 0; a < n; ++a)
            {
                const float *__restrict child = values + a * hands;
                float *__restrict r = regret + a * hands;
                for (std::size_t i = 0; i < hands; ++i)
                {
                    r[i] = r[i] * (r[i] > 0.0f ? positive : 0.5f) + (child[i] - out[i]);
                }
            }
            return;
        }
        float *__restrict childReach = scratch.reach.data();
        float *__restrict childOut = scratch.values.data();
        float *__restrict strategy = &m_strategy[node.offset];
        const float discount = m_strategyDiscount;
        for (std::size_t a = 0; a < n; ++a)
        {
            const float *__restrict s = sigma + a * hands;
            float *__restrict sum = strategy + a * hands;
            for (std::size_t i = 0; i < hands; ++i)
            {
                childReach[i] = reach[i] * s[i];
                sum[i] = sum[i] * discount + childReach[i];
            }
            cfr(node.children[a], p, childReach, childOut, depth + 1);
            for (std::size_t i = 0; i < hands; ++i)
            {
                out[i] += childOut[i];
            }
        }
    }

    // Traverser best-responds per hand; the opponent plays its average strategy.
    void bestResponse(std::size_t idx, std::size_t p, const std::vector<float> &reach, std::vector<float> &out) const
    {
        const Node &node = m_nodes[idx];
        if (node.children.empty())
        {
            terminal(node.state, p, reach.data(), out.data());
            return;
        }
        const std::size_t n = node.children.size();
        std::vector<float> child(m_hands);
        if (node.state.toAct == p)
        {
            std::fill(out.begin(), out.end(), -std::numeric_limits<float>::infinity());
            for (std::size_t a = 0; a < n; ++a)
            {
                bestResponse(node.children[a], p, reach, child);
                for (std::size_t i = 0; i < m_hands; ++i)
                {
                    out[i] = std::max(out[i], child[i]);
                }
            }
            return;
        }
        std::fill(out.begin(), out.end(), 0.0f);
        std::vector<float> childReach(m_hands);
        for (std::size_t a = 0; a < n; ++a)
        {
            for (std::size_t i = 0; i < m_hands; ++i)
            {
                double total = 0.0;
                for (std::size_t b = 0; b < n; ++b)
                {
                    total += m_strategy[node.offset + b * m_hands + i];
                }
                const double share = total > 0.0 ? m_strategy[node.offset + a * m_hands + i] / total : 1.0 / static_cast<double>(n);
                childReach[i] = static_cast<float>(reach[i] * share);
            }
            bestResponse(node.children[a], p, childReach, child);
            for (std::size_t i = 0; i < m_hands; ++i)
            {
                out[i] += child[i];
            }
        }
    }
};
#endif // __POKER_CFR_RIVER_SOLVER_HPP__
