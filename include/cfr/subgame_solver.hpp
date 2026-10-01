#ifndef __POKER_CFR_SUBGAME_SOLVER_HPP__
#define __POKER_CFR_SUBGAME_SOLVER_HPP__
#include "subgame_tree.hpp"
#include <cmath>
#include <vector>

// Resolves a postflop subgame with exact hands (no card abstraction): vector-form Discounted CFR
// (alpha 1.5, beta 0, gamma 2) over a SubgameTree, from both players' ranges. Chance nodes gather into
// and scatter out of child spaces.
template <typename C>
class SubgameSolver
{
public:
    using Tree = SubgameTree<C>;
    using G = Hunl<C>;
    using Hands = typename Tree::Hands;
    using Strategy = typename Tree::Strategy;

    // See SubgameTree for the arguments.
    SubgameSolver(const typename G::State &root, const std::array<Hands, 2> &ranges, bool averageLaterStreets = true, double minReach = 0.0,
                  std::size_t chanceSamples = 0, std::size_t allInSamples = 0)
        : m_tree(root, ranges, averageLaterStreets, minReach, chanceSamples, allInSamples), m_regret(m_tree.regrets, 0.0f), m_strategy(m_tree.averages, 0.0f)
    {
        m_scratch.resize(m_tree.depth + 1);
        for (auto &scratch : m_scratch)
        {
            scratch.sigma.resize(G::maxActions * m_tree.hands);
            scratch.values.resize(G::maxActions * m_tree.hands);
            scratch.reach.resize(m_tree.hands);
        }
    }

    void solve(std::size_t iterations)
    {
        std::vector<float> out(m_tree.hands);
        for (std::size_t i = 0; i < iterations; ++i)
        {
            ++m_iteration;
            const double t = static_cast<double>(m_iteration - 1); // discounts apply to sums through t
            m_positiveDiscount = static_cast<float>(std::pow(t, 1.5) / (std::pow(t, 1.5) + 1.0));
            m_strategyDiscount = static_cast<float>(std::pow(t / (t + 1.0), 2.0));
            for (std::size_t p = 0; p < 2; ++p)
            {
                cfr(0, p, m_tree.ranges[1 - p].data(), out.data(), 0);
            }
        }
    }

    // Average strategy at a root-street state `s` (any Hunl state type: only its action history is used)
    // for the player to act holding hand `hand`, uniform if unreached.
    template <typename S>
    Strategy strategy(const S &s, std::size_t hand) const { return m_tree.strategy(m_strategy, s, hand); }

    // Terminal building blocks per hand on a river root (0 for hands the board blocks), exposed for tests:
    // opponent reach sharing no card with the hand, and its showdown value for `stake` chips each.
    Hands disjointMass(const Hands &opponentReach) const { return perHand(opponentReach, [&](const float *reach, float *out)
                                                                          { disjointMass(m_tree.spaces.front(), reach, out); }); }
    Hands showdown(const Hands &opponentReach, float stake) const { return perHand(opponentReach, [&](const float *reach, float *out)
                                                                                   { showdown(m_tree.spaces.front(), reach, stake, out); }); }

    template <typename S>
    inline bool contains(const S &s) const { return m_tree.byHistory.contains(s.history); }
    inline std::size_t numNodes() const noexcept { return m_tree.nodes.size(); }
    inline std::size_t numValues() const noexcept { return m_regret.size() + m_strategy.size(); } // floats stored
    inline const Tree &tree() const noexcept { return m_tree; }
    inline const std::vector<float> &averageSums() const noexcept { return m_strategy; }

    // Per hand of player p (0 for hands the root board blocks): its best-response value against the
    // opponent's average strategy, weighted by the opponent's reach.
    Hands bestResponseValues(std::size_t p) const
    {
        std::vector<float> values(m_tree.hands);
        bestResponse(0, p, m_tree.ranges[1 - p], values, m_strategy);
        return m_tree.expand(values);
    }

    // Average gain in chips of a best response over both players against the average strategies: this
    // solver's, or ones laid out the same way over the same tree (another solver's averageSums()).
    double exploitability() const { return exploitability(m_strategy); }
    double exploitability(const std::vector<float> &sums) const
    {
        double total = 0.0;
        double joint = 0.0;
        std::vector<float> values(m_tree.hands), mass(m_tree.hands);
        for (std::size_t p = 0; p < 2; ++p)
        {
            bestResponse(0, p, m_tree.ranges[1 - p], values, sums);
            disjointMass(m_tree.spaces.front(), m_tree.ranges[1 - p].data(), mass.data());
            for (std::size_t i = 0; i < m_tree.hands; ++i)
            {
                total += static_cast<double>(m_tree.ranges[p][i]) * values[i];
                joint += p == 0 ? static_cast<double>(m_tree.ranges[p][i]) * mass[i] : 0.0;
            }
        }
        return joint > 0.0 ? total / joint / 2.0 : 0.0;
    }

private:
    using Node = typename Tree::Node;
    using Space = typename Tree::Space;
    using Kind = typename Tree::Kind;
    static constexpr std::size_t none = Tree::none;
    struct Scratch
    {
        std::vector<float> sigma, values, reach;
    };

    Tree m_tree;
    std::vector<float> m_regret, m_strategy;
    std::vector<Scratch> m_scratch; // one per tree depth
    std::size_t m_iteration = 0;
    float m_positiveDiscount = 0.0f;
    float m_strategyDiscount = 0.0f;

    // Hot loops copy the hand count to a local and take __restrict pointers: otherwise MSVC assumes float
    // stores may alias the bound or each other and vectorizes none of them.

    // Opponent reach over hands sharing no card with each hand.
    static void disjointMass(const Space &space, const float *__restrict reach, float *__restrict out)
    {
        const std::size_t hands = space.size();
        const std::uint8_t *__restrict low = space.low.data();
        const std::uint8_t *__restrict high = space.high.data();
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

    // Showdown value per hand for `stake` chips each, in one ascending pass. With W the disjoint opponent
    // mass strictly weaker, W' weaker or tied and D all of it: win - lose = W + W' - D.
    // Inclusion-exclusion: a hand adds itself to both of its card sums, so W' and D add it back once.
    // (Per-card hand lists and branch-free prefix sums with indexed reads were both measured slower.)
    static void showdown(const Space &space, const float *__restrict reach, float stake, float *__restrict out)
    {
        const std::size_t hands = space.size();
        const std::uint8_t *__restrict low = space.low.data();
        const std::uint8_t *__restrict high = space.high.data();
        const std::size_t *__restrict groups = space.groups.data();
        const std::size_t numGroups = space.groups.size() - 1;
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

    // Traverser's value per hand at a terminal, weighted by opponent reach.
    void terminal(const Node &node, std::size_t p, const float *__restrict reach, float *__restrict out) const
    {
        const Space &space = m_tree.spaces[node.space];
        const auto &s = node.state;
        if (node.kind == Kind::showdown)
        {
            showdown(space, reach, static_cast<float>(s.invested[p]), out);
            return;
        }
        const std::size_t hands = space.size();
        const float payoff = s.folder == p ? -static_cast<float>(s.invested[p]) : static_cast<float>(s.invested[s.folder]);
        disjointMass(space, reach, out);
        for (std::size_t i = 0; i < hands; ++i)
        {
            out[i] *= payoff;
        }
    }

    template <typename F>
    Hands perHand(const Hands &opponentReach, F compute) const
    {
        std::vector<float> reach(m_tree.hands), out(m_tree.hands);
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if (m_tree.denseOf[h] >= 0)
            {
                reach[static_cast<std::size_t>(m_tree.denseOf[h])] = static_cast<float>(opponentReach[h]);
            }
        }
        compute(reach.data(), out.data());
        Hands result{};
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            result[h] = m_tree.denseOf[h] >= 0 ? out[static_cast<std::size_t>(m_tree.denseOf[h])] : 0.0;
        }
        return result;
    }

    // Deals each child's card, weighted by node.weight: hands holding it are not in the child's space.
    template <typename F>
    void chance(const Node &node, const float *__restrict reach, float *__restrict out, Scratch &scratch, F recurse) const
    {
        const std::size_t hands = m_tree.spaces[node.space].size();
        float *__restrict childReach = scratch.reach.data();
        float *__restrict childOut = scratch.values.data();
        for (std::size_t i = 0; i < hands; ++i)
        {
            out[i] = 0.0f;
        }
        for (const std::size_t child : node.children)
        {
            const Space &space = m_tree.spaces[m_tree.nodes[child].space];
            const std::size_t live = space.size();
            const std::uint16_t *__restrict parent = space.parent.data();
            for (std::size_t j = 0; j < live; ++j)
            {
                childReach[j] = reach[parent[j]];
            }
            recurse(child, childReach, childOut);
            for (std::size_t j = 0; j < live; ++j)
            {
                out[parent[j]] += childOut[j];
            }
        }
        const float weight = node.weight;
        for (std::size_t i = 0; i < hands; ++i)
        {
            out[i] *= weight;
        }
    }

    // Regret matching per hand into sigma[action * hands + hand].
    void currentStrategy(const Node &node, std::size_t n, float *__restrict sigma) const
    {
        const std::size_t hands = m_tree.spaces[node.space].size();
        const float *__restrict regret = &m_regret[node.offset];
        float *__restrict total = sigma + (n - 1) * hands; // last row doubles as the running total, filled last
        for (std::size_t i = 0; i < hands; ++i)
        {
            total[i] = 0.0f;
        }
        // Plain selects, not std::max (MSVC vectorizes none of these loops with it).
        for (std::size_t a = 0; a < n; ++a)
        {
            const float *__restrict r = regret + a * hands;
            for (std::size_t i = 0; i < hands; ++i)
            {
                total[i] += r[i] > 0.0f ? r[i] : 0.0f;
            }
        }
        // One division per hand: total becomes its reciprocal, or 0 with no positive regret (uniform then).
        for (std::size_t i = 0; i < hands; ++i)
        {
            total[i] = total[i] > 0.0f ? 1.0f / total[i] : 0.0f;
        }
        const float uniform = 1.0f / static_cast<float>(n);
        for (std::size_t a = 0; a + 1 < n; ++a)
        {
            const float *__restrict r = regret + a * hands;
            float *__restrict out = sigma + a * hands;
            for (std::size_t i = 0; i < hands; ++i)
            {
                out[i] = (r[i] > 0.0f ? r[i] : 0.0f) * total[i] + (total[i] > 0.0f ? 0.0f : uniform);
            }
        }
        const float *__restrict last = regret + (n - 1) * hands;
        for (std::size_t i = 0; i < hands; ++i)
        {
            total[i] = (last[i] > 0.0f ? last[i] : 0.0f) * total[i] + (total[i] > 0.0f ? 0.0f : uniform);
        }
    }

    void cfr(std::size_t idx, std::size_t p, const float *__restrict reach, float *__restrict out, std::size_t depth)
    {
        const Node &node = m_tree.nodes[idx];
        if (node.kind == Kind::fold || node.kind == Kind::showdown)
        {
            terminal(node, p, reach, out);
            return;
        }
        Scratch &scratch = m_scratch[depth];
        if (node.kind == Kind::chance)
        {
            chance(node, reach, out, scratch, [&](std::size_t child, const float *childReach, float *childOut)
                   { cfr(child, p, childReach, childOut, depth + 1); });
            return;
        }
        const std::size_t hands = m_tree.spaces[node.space].size();
        const std::size_t n = node.children.size();
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
        const float discount = m_strategyDiscount;
        for (std::size_t a = 0; a < n; ++a)
        {
            const float *__restrict s = sigma + a * hands;
            for (std::size_t i = 0; i < hands; ++i)
            {
                childReach[i] = reach[i] * s[i];
            }
            if (node.average != none)
            {
                float *__restrict sum = &m_strategy[node.average + a * hands];
                for (std::size_t i = 0; i < hands; ++i)
                {
                    sum[i] = sum[i] * discount + childReach[i];
                }
            }
            cfr(node.children[a], p, childReach, childOut, depth + 1);
            for (std::size_t i = 0; i < hands; ++i)
            {
                out[i] += childOut[i];
            }
        }
    }

    // Traverser best-responds per hand; the opponent plays its average strategy (its current one where
    // no average is kept).
    void bestResponse(std::size_t idx, std::size_t p, const std::vector<float> &reach, std::vector<float> &out, const std::vector<float> &sums) const
    {
        const Node &node = m_tree.nodes[idx];
        const std::size_t hands = m_tree.spaces[node.space].size();
        if (node.kind == Kind::fold || node.kind == Kind::showdown)
        {
            terminal(node, p, reach.data(), out.data());
            return;
        }
        const std::size_t n = node.children.size();
        std::vector<float> child(m_tree.hands);
        if (node.kind == Kind::chance)
        {
            Scratch scratch{{}, std::vector<float>(m_tree.hands), std::vector<float>(m_tree.hands)};
            chance(node, reach.data(), out.data(), scratch, [&](std::size_t next, const float *childReach, float *childOut)
                   {
                       const std::vector<float> in(childReach, childReach + m_tree.hands);
                       bestResponse(next, p, in, child, sums);
                       std::copy(child.begin(), child.end(), childOut); });
            return;
        }
        if (node.state.toAct == p)
        {
            std::fill(out.begin(), out.begin() + hands, -std::numeric_limits<float>::infinity());
            for (std::size_t a = 0; a < n; ++a)
            {
                bestResponse(node.children[a], p, reach, child, sums);
                for (std::size_t i = 0; i < hands; ++i)
                {
                    out[i] = std::max(out[i], child[i]);
                }
            }
            return;
        }
        std::vector<float> sigma(n * hands);
        if (node.average == none)
        {
            currentStrategy(node, n, sigma.data());
        }
        else
        {
            for (std::size_t i = 0; i < hands; ++i)
            {
                double total = 0.0;
                for (std::size_t a = 0; a < n; ++a)
                {
                    total += sums[node.average + a * hands + i];
                }
                for (std::size_t a = 0; a < n; ++a)
                {
                    sigma[a * hands + i] = total > 0.0 ? static_cast<float>(sums[node.average + a * hands + i] / total) : 1.0f / static_cast<float>(n);
                }
            }
        }
        std::fill(out.begin(), out.begin() + hands, 0.0f);
        std::vector<float> childReach(m_tree.hands);
        for (std::size_t a = 0; a < n; ++a)
        {
            for (std::size_t i = 0; i < hands; ++i)
            {
                childReach[i] = reach[i] * sigma[a * hands + i];
            }
            bestResponse(node.children[a], p, childReach, child, sums);
            for (std::size_t i = 0; i < hands; ++i)
            {
                out[i] += child[i];
            }
        }
    }
};

// The same game with coarser betting from street `From` on, for lookahead: a 0.75-pot bet or all-in,
// then only fold or call. Earlier streets and action histories are unchanged.
template <typename C, std::size_t From>
struct Coarse : C
{
    static constexpr auto raiseFractions = []
    {
        auto fractions = C::raiseFractions;
        for (std::size_t street = From; street < 4; ++street)
        {
            fractions[street] = {};
            if (!fractions[street].empty())
            {
                fractions[street][0] = 0.75;
            }
        }
        return fractions;
    }();
    static constexpr auto maxRaises = []
    {
        auto raises = C::maxRaises;
        for (std::size_t street = From; street < 4; ++street)
        {
            raises[street] = std::min<std::uint8_t>(raises[street], 1);
        }
        return raises;
    }();
};
template <typename C>
using CoarseRiver = Coarse<C, 3>; // lookahead from the turn
template <typename C>
using CoarseTurn = Coarse<C, 2>; // lookahead from the flop

#endif // __POKER_CFR_SUBGAME_SOLVER_HPP__
