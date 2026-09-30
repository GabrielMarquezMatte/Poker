#ifndef __POKER_CFR_SUBGAME_SOLVER_HPP__
#define __POKER_CFR_SUBGAME_SOLVER_HPP__
#include "push_fold.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

// Resolves a postflop subgame with exact hands (no card abstraction): vector-form Discounted CFR
// (alpha 1.5, beta 0, gamma 2) over the public betting tree of Hunl<C> from the root to showdown,
// from both players' ranges. Later board cards are chance nodes with one child per card.
// Every board in the tree has its own dense vector of the hands it leaves alive (a space), so per-hand
// loops are contiguous and card removal is exact; on five cards the space is sorted weakest first and
// showdowns cost O(hands) via per-card sums. Chance nodes gather into and scatter out of child spaces.
template <typename C>
class SubgameSolver
{
public:
    using G = Hunl<C>;
    using Hands = std::array<double, holeCombos>;
    using Strategy = std::array<double, G::maxActions>;

    // `root` is a street's first decision; ranges[p] holds player p's reach for every hand. Hands whose
    // reach is at most `minReach` times the largest in both ranges are left out (they play uniformly).
    // Average strategies are kept for root-street nodes only unless `averageLaterStreets` (later streets
    // are resolved again when reached, but exploitability needs them).
    SubgameSolver(const typename G::State &root, const std::array<Hands, 2> &ranges, bool averageLaterStreets = true, double minReach = 0.0)
        : m_averageLaterStreets(averageLaterStreets)
    {
        const bool river = std::popcount(root.board) == 5;
        std::array<double, 2> floor{};
        for (std::size_t p = 0; p < 2; ++p)
        {
            floor[p] = minReach * *std::max_element(ranges[p].begin(), ranges[p].end());
        }
        std::vector<std::pair<ClassificationResult, std::uint16_t>> dense;
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if ((holes[h] & root.board) == 0 && (ranges[0][h] > floor[0] || ranges[1][h] > floor[1]))
            {
                dense.emplace_back(river ? Hand::classify(Deck::from_mask(holes[h] | root.board)) : ClassificationResult{},
                                   static_cast<std::uint16_t>(h));
            }
        }
        std::sort(dense.begin(), dense.end()); // by hand index unless on the river
        m_hands = dense.size();
        m_denseOf.fill(-1);
        Space &space = m_spaces.emplace_back();
        for (std::size_t i = 0; i < m_hands; ++i)
        {
            const std::uint16_t h = dense[i].second;
            m_denseOf[h] = static_cast<int>(i);
            space.low.push_back(static_cast<std::uint8_t>(std::countr_zero(holes[h])));
            space.high.push_back(static_cast<std::uint8_t>(63 - std::countl_zero(holes[h])));
            if (river && (i == 0 || dense[i].first != dense[i - 1].first))
            {
                space.groups.push_back(i);
            }
            for (std::size_t p = 0; p < 2; ++p)
            {
                m_ranges[p].push_back(static_cast<float>(ranges[p][h]));
            }
        }
        space.groups.push_back(m_hands);
        const std::size_t depth = build(root, root.board, 0, 0);
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

    // Average strategy at a root-street state `s` (any Hunl state type: only its action history is used)
    // for the player to act holding hand `hand`, uniform if unreached.
    template <typename S>
    Strategy strategy(const S &s, std::size_t hand) const
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
            total += m_strategy[node.average + a * m_hands + i];
        }
        for (std::size_t a = 0; a < n && total > 0.0; ++a)
        {
            out[a] = m_strategy[node.average + a * m_hands + i] / total;
        }
        return out;
    }

    // Terminal building blocks per hand on a river root (0 for hands the board blocks), exposed for tests:
    // opponent reach sharing no card with the hand, and its showdown value for `stake` chips each.
    Hands disjointMass(const Hands &opponentReach) const { return perHand(opponentReach, [&](const float *reach, float *out)
                                                                          { disjointMass(m_spaces.front(), reach, out); }); }
    Hands showdown(const Hands &opponentReach, float stake) const { return perHand(opponentReach, [&](const float *reach, float *out)
                                                                                   { showdown(m_spaces.front(), reach, stake, out); }); }

    template <typename S>
    inline bool contains(const S &s) const { return m_byHistory.contains(s.history); }
    inline std::size_t numNodes() const noexcept { return m_nodes.size(); }
    inline std::size_t numValues() const noexcept { return m_regret.size() + m_strategy.size(); } // floats stored

    // Per hand of player p (0 for hands the root board blocks): its best-response value against the
    // opponent's average strategy, weighted by the opponent's reach.
    Hands bestResponseValues(std::size_t p) const
    {
        std::vector<float> values(m_hands);
        bestResponse(0, p, m_ranges[1 - p], values);
        Hands result{};
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            result[h] = m_denseOf[h] >= 0 ? values[static_cast<std::size_t>(m_denseOf[h])] : 0.0;
        }
        return result;
    }

    // Average gain in chips of a best response over both players against the average strategies.
    double exploitability() const
    {
        double total = 0.0;
        double joint = 0.0;
        std::vector<float> values(m_hands), mass(m_hands);
        for (std::size_t p = 0; p < 2; ++p)
        {
            bestResponse(0, p, m_ranges[1 - p], values);
            disjointMass(m_spaces.front(), m_ranges[1 - p].data(), mass.data());
            for (std::size_t i = 0; i < m_hands; ++i)
            {
                total += static_cast<double>(m_ranges[p][i]) * values[i];
                joint += p == 0 ? static_cast<double>(m_ranges[p][i]) * mass[i] : 0.0;
            }
        }
        return joint > 0.0 ? total / joint / 2.0 : 0.0;
    }

private:
    static constexpr std::size_t none = std::numeric_limits<std::size_t>::max();

    enum class Kind : std::uint8_t
    {
        decision,
        fold,
        showdown,
        chance,
    };
    struct Node
    {
        typename G::State state;
        std::vector<std::size_t> children;
        std::size_t offset = 0;     // into m_regret: [action * hands + hand]
        std::size_t average = none; // into m_strategy, same layout, if kept
        std::size_t space = 0;
        Kind kind = Kind::decision;
    };
    // The hands alive on one board, densely.
    struct Space
    {
        std::vector<std::uint8_t> low, high; // card indices per hand
        std::vector<std::size_t> groups;     // five cards: start of each equal-strength run (weakest first), plus the end
        std::vector<std::uint16_t> parent;   // below a chance node: each hand's index in the parent's space
        inline std::size_t size() const noexcept { return low.size(); }
    };
    struct Scratch
    {
        std::vector<float> sigma, values, reach;
    };

    bool m_averageLaterStreets;
    std::size_t m_hands = 0; // in the root space, the largest
    std::array<int, holeCombos> m_denseOf{};
    std::array<std::vector<float>, 2> m_ranges;
    std::vector<Node> m_nodes;
    std::unordered_map<std::uint64_t, std::size_t> m_byHistory; // root-street nodes only
    std::vector<Space> m_spaces;
    std::map<std::pair<std::uint64_t, std::size_t>, std::size_t> m_spaceOf; // by board and parent space
    std::vector<float> m_regret, m_strategy;
    std::vector<Scratch> m_scratch; // one per tree depth
    std::size_t m_iteration = 0;
    float m_positiveDiscount = 0.0f;
    float m_strategyDiscount = 0.0f;

    // The space of `board`, one card more than the board of space `parentIndex`.
    std::size_t spaceFor(std::uint64_t board, std::size_t parentIndex)
    {
        if (const auto it = m_spaceOf.find({board, parentIndex}); it != m_spaceOf.end())
        {
            return it->second;
        }
        const bool river = std::popcount(board) == 5;
        std::vector<std::pair<ClassificationResult, std::uint16_t>> ranked;
        {
            const Space &parent = m_spaces[parentIndex];
            for (std::size_t j = 0; j < parent.size(); ++j)
            {
                const std::uint64_t hole = (1ull << parent.low[j]) | (1ull << parent.high[j]);
                if ((hole & board) == 0)
                {
                    ranked.emplace_back(river ? Hand::classify(Deck::from_mask(hole | board)) : ClassificationResult{}, static_cast<std::uint16_t>(j));
                }
            }
        }
        std::sort(ranked.begin(), ranked.end());
        Space space;
        for (std::size_t k = 0; k < ranked.size(); ++k)
        {
            if (river && (k == 0 || ranked[k].first != ranked[k - 1].first))
            {
                space.groups.push_back(k);
            }
            space.parent.push_back(ranked[k].second);
            space.low.push_back(m_spaces[parentIndex].low[ranked[k].second]);
            space.high.push_back(m_spaces[parentIndex].high[ranked[k].second]);
        }
        space.groups.push_back(ranked.size());
        m_spaces.push_back(std::move(space));
        m_spaceOf[{board, parentIndex}] = m_spaces.size() - 1;
        return m_spaces.size() - 1;
    }

    // Returns the subtree depth.
    std::size_t build(const typename G::State &s, std::uint64_t rootBoard, std::size_t space, std::size_t depth)
    {
        const std::size_t idx = m_nodes.size();
        m_nodes.push_back({s, {}, 0, none, space, Kind::decision});
        if (s.board == rootBoard)
        {
            m_byHistory[s.history] = idx;
        }
        if (G::isTerminal(s))
        {
            m_nodes[idx].kind = s.folder != G::nobody ? Kind::fold : Kind::showdown;
            return depth;
        }
        std::vector<std::size_t> children;
        std::size_t deepest = depth;
        if (G::isChance(s))
        {
            m_nodes[idx].kind = Kind::chance;
            const std::uint64_t live = ((1ull << 52) - 1) & ~s.board;
            for (std::uint64_t c = live; c != 0; c &= c - 1)
            {
                typename G::State next = s;
                next.board |= c & (~c + 1);
                const std::size_t childSpace = spaceFor(next.board, space);
                children.push_back(m_nodes.size());
                deepest = std::max(deepest, build(next, rootBoard, childSpace, depth + 1));
            }
            m_nodes[idx].children = std::move(children);
            return deepest;
        }
        const std::size_t n = G::numActions(s);
        const std::size_t values = n * m_spaces[space].size();
        m_nodes[idx].offset = m_regret.size();
        m_regret.resize(m_regret.size() + values, 0.0f);
        if (s.board == rootBoard || m_averageLaterStreets)
        {
            m_nodes[idx].average = m_strategy.size();
            m_strategy.resize(m_strategy.size() + values, 0.0f);
        }
        for (std::size_t a = 0; a < n; ++a)
        {
            children.push_back(m_nodes.size());
            deepest = std::max(deepest, build(G::apply(s, a), rootBoard, space, depth + 1));
        }
        m_nodes[idx].children = std::move(children);
        return deepest;
    }

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
        const Space &space = m_spaces[node.space];
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

    // Deals each child's card: hands holding it are not in the child's space, and every hand pair sees
    // `children - 4` possible cards, equally likely.
    template <typename F>
    void chance(const Node &node, const float *__restrict reach, float *__restrict out, Scratch &scratch, F recurse) const
    {
        const std::size_t hands = m_spaces[node.space].size();
        float *__restrict childReach = scratch.reach.data();
        float *__restrict childOut = scratch.values.data();
        for (std::size_t i = 0; i < hands; ++i)
        {
            out[i] = 0.0f;
        }
        for (const std::size_t child : node.children)
        {
            const Space &space = m_spaces[m_nodes[child].space];
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
        const float weight = 1.0f / static_cast<float>(node.children.size() - 4);
        for (std::size_t i = 0; i < hands; ++i)
        {
            out[i] *= weight;
        }
    }

    // Regret matching per hand into sigma[action * hands + hand].
    void currentStrategy(const Node &node, std::size_t n, float *__restrict sigma) const
    {
        const std::size_t hands = m_spaces[node.space].size();
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
        const Node &node = m_nodes[idx];
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
        const std::size_t hands = m_spaces[node.space].size();
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
    void bestResponse(std::size_t idx, std::size_t p, const std::vector<float> &reach, std::vector<float> &out) const
    {
        const Node &node = m_nodes[idx];
        const std::size_t hands = m_spaces[node.space].size();
        if (node.kind == Kind::fold || node.kind == Kind::showdown)
        {
            terminal(node, p, reach.data(), out.data());
            return;
        }
        const std::size_t n = node.children.size();
        std::vector<float> child(m_hands);
        if (node.kind == Kind::chance)
        {
            Scratch scratch{{}, std::vector<float>(m_hands), std::vector<float>(m_hands)};
            chance(node, reach.data(), out.data(), scratch, [&](std::size_t next, const float *childReach, float *childOut)
                   {
                       const std::vector<float> in(childReach, childReach + m_hands);
                       bestResponse(next, p, in, child);
                       std::copy(child.begin(), child.end(), childOut); });
            return;
        }
        if (node.state.toAct == p)
        {
            std::fill(out.begin(), out.begin() + hands, -std::numeric_limits<float>::infinity());
            for (std::size_t a = 0; a < n; ++a)
            {
                bestResponse(node.children[a], p, reach, child);
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
                    total += m_strategy[node.average + a * hands + i];
                }
                for (std::size_t a = 0; a < n; ++a)
                {
                    sigma[a * hands + i] = total > 0.0 ? static_cast<float>(m_strategy[node.average + a * hands + i] / total) : 1.0f / static_cast<float>(n);
                }
            }
        }
        std::fill(out.begin(), out.begin() + hands, 0.0f);
        std::vector<float> childReach(m_hands);
        for (std::size_t a = 0; a < n; ++a)
        {
            for (std::size_t i = 0; i < hands; ++i)
            {
                childReach[i] = reach[i] * sigma[a * hands + i];
            }
            bestResponse(node.children[a], p, childReach, child);
            for (std::size_t i = 0; i < hands; ++i)
            {
                out[i] += child[i];
            }
        }
    }
};

// The same game with a coarser river, for lookahead from the turn: a 0.75-pot bet or all-in, then
// only fold or call. Earlier streets and action histories are unchanged.
template <typename C>
struct CoarseRiver : C
{
    static constexpr auto raiseFractions = []
    {
        auto fractions = C::raiseFractions;
        fractions[3] = {};
        if (!fractions[3].empty())
        {
            fractions[3][0] = 0.75;
        }
        return fractions;
    }();
    static constexpr auto maxRaises = []
    {
        auto raises = C::maxRaises;
        raises[3] = std::min<std::uint8_t>(raises[3], 1);
        return raises;
    }();
};
#endif // __POKER_CFR_SUBGAME_SOLVER_HPP__
