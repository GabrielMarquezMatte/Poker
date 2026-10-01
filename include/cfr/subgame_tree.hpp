#ifndef __POKER_CFR_SUBGAME_TREE_HPP__
#define __POKER_CFR_SUBGAME_TREE_HPP__
#include "push_fold.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

// How to build a SubgameTree (see its constructor).
struct SubgameOptions
{
    bool averageLaterStreets = true;
    double minReach = 0.0;
    std::size_t chanceSamples = 0, allInSamples = 0;
    std::optional<std::uint32_t> rootExtraAction; // chips the root's actor puts in, on top of the legal actions
};

// The public tree of a postflop subgame of Hunl<C>, from a street's first decision to showdown, with the
// hands each board leaves alive: what a vector-form CFR solver over exact hands iterates on.
// Later board cards are chance nodes with one child per card. Every board has its own dense vector of
// live hands (a space); on five cards it is sorted weakest first, so showdowns cost O(hands) via
// per-card sums. Decision nodes own `actions * hands` regrets (and average strategy sums, if kept) at
// their offsets, laid out [action * hands + hand].
template <typename C>
struct SubgameTree
{
    using G = Hunl<C>;
    using Hands = std::array<double, holeCombos>;
    static constexpr std::size_t maxChildren = G::maxActions + 1; // with a root's extra action
    using Strategy = std::array<double, maxChildren>;
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
        std::size_t offset = 0;     // decision: into the regrets
        std::size_t average = none; // decision: into the average strategy sums, if kept
        std::size_t space = 0;
        Kind kind = Kind::decision;
        float weight = 0.0f; // chance: each child's probability for a hand pair (see build)
    };
    // The hands alive on one board, densely.
    struct Space
    {
        std::vector<std::uint8_t> low, high; // card indices per hand
        std::vector<std::size_t> groups;     // five cards: start of each equal-strength run (weakest first), plus the end
        std::vector<std::uint16_t> parent;   // below a chance node: each hand's index in the parent's space
        inline std::size_t size() const noexcept { return low.size(); }
    };

    std::size_t hands = 0; // in the root space, the largest
    std::array<int, holeCombos> denseOf{};
    std::array<std::vector<float>, 2> ranges; // per root-space hand
    std::vector<Node> nodes;                  // nodes[0] is the root; children follow their parent
    std::unordered_map<std::uint64_t, std::size_t> byHistory; // root-street nodes only
    std::vector<Space> spaces;                                // spaces[0] is the root's
    std::size_t regrets = 0, averages = 0;                    // floats needed
    std::size_t depth = 0;

    // `root` is a decision (usually a street's first); ranges[p] holds player p's reach for every hand.
    // Options: hands whose reach is at most `minReach` times the largest in both ranges are left out (they
    // play uniformly). Average strategies are kept for root-street nodes only unless
    // `averageLaterStreets`. With `chanceSamples` > 0, each chance node deals only that many cards, drawn
    // at random (seeded by the board and history, so the tree is reproducible); `allInSamples` does the
    // same for the board that runs out after an all-in, whose value is all there is to those subtrees (0
    // deals every card). `rootExtraAction` adds an action at the root, after the legal ones (the child is
    // G::applyTo(root, chips, number of legal actions)): a real bet the abstraction lacks.
    SubgameTree(const typename G::State &root, const std::array<Hands, 2> &reach, const SubgameOptions &options = {})
        : m_averageLaterStreets(options.averageLaterStreets), m_chanceSamples(options.chanceSamples), m_allInSamples(options.allInSamples),
          m_rootExtraAction(options.rootExtraAction)
    {
        const double minReach = options.minReach;
        const bool river = std::popcount(root.board) == 5;
        std::array<double, 2> floor{};
        for (std::size_t p = 0; p < 2; ++p)
        {
            floor[p] = minReach * *std::max_element(reach[p].begin(), reach[p].end());
        }
        std::vector<std::pair<ClassificationResult, std::uint16_t>> dense;
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if ((holes[h] & root.board) == 0 && (reach[0][h] > floor[0] || reach[1][h] > floor[1]))
            {
                dense.emplace_back(river ? Hand::classify(Deck::from_mask(holes[h] | root.board)) : ClassificationResult{},
                                   static_cast<std::uint16_t>(h));
            }
        }
        std::sort(dense.begin(), dense.end()); // by hand index unless on the river
        hands = dense.size();
        denseOf.fill(-1);
        Space &space = spaces.emplace_back();
        for (std::size_t i = 0; i < hands; ++i)
        {
            const std::uint16_t h = dense[i].second;
            denseOf[h] = static_cast<int>(i);
            space.low.push_back(static_cast<std::uint8_t>(std::countr_zero(holes[h])));
            space.high.push_back(static_cast<std::uint8_t>(63 - std::countl_zero(holes[h])));
            if (river && (i == 0 || dense[i].first != dense[i - 1].first))
            {
                space.groups.push_back(i);
            }
            for (std::size_t p = 0; p < 2; ++p)
            {
                ranges[p].push_back(static_cast<float>(reach[p][h]));
            }
        }
        space.groups.push_back(hands);
        depth = build(root, root.board, 0, 0);
    }

    // The average strategy in `sums` at a root-street state `s` (any Hunl state type: only its action
    // history is used) for the player to act holding hand `hand`, uniform if unreached.
    template <typename S>
    Strategy strategy(const std::vector<float> &sums, const S &s, std::size_t hand) const
    {
        const Node &node = nodes[byHistory.at(s.history)];
        const std::size_t n = node.children.size();
        Strategy out{};
        std::fill_n(out.begin(), n, 1.0 / static_cast<double>(n)); // the rest stay 0
        if (denseOf[hand] < 0)
        {
            return out;
        }
        const std::size_t i = static_cast<std::size_t>(denseOf[hand]);
        double total = 0.0;
        for (std::size_t a = 0; a < n; ++a)
        {
            total += sums[node.average + a * hands + i];
        }
        for (std::size_t a = 0; a < n && total > 0.0; ++a)
        {
            out[a] = sums[node.average + a * hands + i] / total;
        }
        return out;
    }

    // A per-hand vector over the root space as a full hand array (0 for hands left out).
    Hands expand(const std::vector<float> &values) const
    {
        Hands result{};
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            result[h] = denseOf[h] >= 0 ? values[static_cast<std::size_t>(denseOf[h])] : 0.0;
        }
        return result;
    }

private:
    bool m_averageLaterStreets;
    std::size_t m_chanceSamples, m_allInSamples;
    std::optional<std::uint32_t> m_rootExtraAction;
    std::map<std::pair<std::uint64_t, std::size_t>, std::size_t> m_spaceOf; // by board and parent space

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
            const Space &parent = spaces[parentIndex];
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
            space.low.push_back(spaces[parentIndex].low[ranked[k].second]);
            space.high.push_back(spaces[parentIndex].high[ranked[k].second]);
        }
        space.groups.push_back(ranked.size());
        spaces.push_back(std::move(space));
        m_spaceOf[{board, parentIndex}] = spaces.size() - 1;
        return spaces.size() - 1;
    }

    // Returns the subtree depth.
    std::size_t build(const typename G::State &s, std::uint64_t rootBoard, std::size_t space, std::size_t level)
    {
        const std::size_t idx = nodes.size();
        nodes.push_back({s, {}, 0, none, space, Kind::decision});
        if (s.board == rootBoard)
        {
            byHistory[s.history] = idx;
        }
        if (G::isTerminal(s))
        {
            nodes[idx].kind = s.folder != G::nobody ? Kind::fold : Kind::showdown;
            return level;
        }
        std::vector<std::size_t> children;
        std::size_t deepest = level;
        if (G::isChance(s))
        {
            nodes[idx].kind = Kind::chance;
            std::vector<std::uint64_t> cards;
            for (std::uint64_t c = ((1ull << 52) - 1) & ~s.board; c != 0; c &= c - 1)
            {
                cards.push_back(c & (~c + 1));
            }
            // A hand pair sees `remaining - 4` of the remaining cards, equally likely. Dealing only a
            // sample, each dealt card stands for remaining / sampled of them: exact for a full deal and
            // unbiased on average for a sample.
            const auto remaining = static_cast<double>(cards.size());
            const std::size_t samples = s.street == G::showdown ? m_allInSamples : m_chanceSamples;
            if (samples > 0 && cards.size() > samples)
            {
                CfrRng rng{s.board * 0x9E3779B97F4A7C15ull ^ s.history};
                for (std::size_t i = 0; i < samples; ++i)
                {
                    std::swap(cards[i], cards[i + rng() % (cards.size() - i)]);
                }
                cards.resize(samples);
            }
            nodes[idx].weight = static_cast<float>(remaining / (static_cast<double>(cards.size()) * (remaining - 4.0)));
            for (const std::uint64_t card : cards)
            {
                typename G::State next = s;
                next.board |= card;
                const std::size_t childSpace = spaceFor(next.board, space);
                children.push_back(nodes.size());
                deepest = std::max(deepest, build(next, rootBoard, childSpace, level + 1));
            }
            nodes[idx].children = std::move(children);
            return deepest;
        }
        const std::size_t legal = G::numActions(s);
        const bool extra = idx == 0 && m_rootExtraAction.has_value();
        const std::size_t n = legal + (extra ? 1 : 0);
        const std::size_t values = n * spaces[space].size();
        nodes[idx].offset = regrets;
        regrets += values;
        if (s.board == rootBoard || m_averageLaterStreets)
        {
            nodes[idx].average = averages;
            averages += values;
        }
        for (std::size_t a = 0; a < n; ++a)
        {
            children.push_back(nodes.size());
            const auto child = a < legal ? G::apply(s, a) : G::applyTo(s, *m_rootExtraAction, a);
            deepest = std::max(deepest, build(child, rootBoard, space, level + 1));
        }
        nodes[idx].children = std::move(children);
        return deepest;
    }
};
#endif // __POKER_CFR_SUBGAME_TREE_HPP__
