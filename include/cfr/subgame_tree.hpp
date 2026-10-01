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
    bool exactFlopAllIns = false;
    std::optional<std::uint32_t> rootExtraAction; // chips the root's actor puts in, on top of the legal actions
};

// The strength rank of every hole pair on a five-card board: equal strengths share a rank, weaker hands
// have lower ones (pairs holding a board card are left at 0). Trees sort a river's hands by it. The
// runouts of different all-ins and the re-solves of a street ask for the same boards again and again,
// so each thread keeps the last ones it ranked: two generations, dropping the older one when the newer
// fills up, so the boards of the flop being worked on are never dropped halfway. The reference holds
// until the thread's next call.
inline const std::array<std::uint16_t, holeCombos> &boardRanks(std::uint64_t board)
{
    thread_local std::unordered_map<std::uint64_t, std::array<std::uint16_t, holeCombos>> cache, older;
    if (const auto it = cache.find(board); it != cache.end())
    {
        return it->second;
    }
    if (const auto it = older.find(board); it != older.end())
    {
        return it->second;
    }
    if (cache.size() >= 2048) // a flop has 1176 runouts; ~5 MB a generation
    {
        older = std::move(cache);
        cache.clear();
    }
    std::vector<std::pair<ClassificationResult, std::uint16_t>> ranked;
    ranked.reserve(holeCombos);
    for (std::size_t h = 0; h < holeCombos; ++h)
    {
        if ((holes[h] & board) == 0)
        {
            ranked.emplace_back(Hand::classify(Deck::from_mask(holes[h] | board)), static_cast<std::uint16_t>(h));
        }
    }
    std::sort(ranked.begin(), ranked.end());
    auto &ranks = cache[board];
    ranks.fill(0);
    std::uint16_t rank = 0;
    for (std::size_t k = 0; k < ranked.size(); ++k)
    {
        rank += k > 0 && ranked[k].first != ranked[k - 1].first ? 1 : 0;
        ranks[ranked[k].second] = rank;
    }
    return ranks;
}

// For a flop, how every hole pair fares against every other when the board runs out, over all 1176
// turn-and-river pairs: [i * holeCombos + j] is the runouts where hand i beats hand j minus those where it
// loses (0 for pairs sharing a card or holding a flop card). Two such hands see `flopAllInRunouts` of them.
// Each thread keeps its last flop: a street is re-solved on the same board.
inline constexpr float flopAllInRunouts = 990.0f; // C(45, 2)
inline const std::vector<std::int16_t> &flopAllInBalance(std::uint64_t flop)
{
    thread_local std::uint64_t cached = 0;
    thread_local std::vector<std::int16_t> balance;
    if (cached == flop && !balance.empty())
    {
        return balance;
    }
    cached = flop;
    balance.assign(holeCombos * holeCombos, 0);
    std::array<std::int16_t, holeCombos> live{}; // all ones for the hands a board leaves alive
    forEachPair(((1ull << 52) - 1) & ~flop, [&](std::uint64_t runout)
                {
        const std::uint16_t *__restrict ranks = boardRanks(flop | runout).data();
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            live[h] = (holes[h] & (flop | runout)) == 0 ? -1 : 0;
        }
        const std::int16_t *__restrict alive = live.data();
        for (std::size_t i = 0; i < holeCombos; ++i)
        {
            if (alive[i] == 0)
            {
                continue;
            }
            const std::uint16_t mine = ranks[i];
            std::int16_t *__restrict row = balance.data() + i * holeCombos;
            for (std::size_t j = i + 1; j < holeCombos; ++j) // the lower half is filled in at the end
            {
                row[j] = static_cast<std::int16_t>(row[j] + (alive[j] & ((ranks[j] < mine) - (ranks[j] > mine))));
            }
        } });
    for (std::size_t i = 0; i < holeCombos; ++i)
    {
        for (std::size_t j = i + 1; j < holeCombos; ++j)
        {
            std::int16_t &upper = balance[i * holeCombos + j];
            upper = (holes[i] & holes[j]) == 0 ? upper : std::int16_t{0};
            balance[j * holeCombos + i] = static_cast<std::int16_t>(-upper);
        }
    }
    return balance;
}

// The public tree of a postflop subgame of Hunl<C>, from a street's first decision to showdown, with the
// hands each board leaves alive: what a vector-form CFR solver over exact hands iterates on.
// Later board cards are chance nodes with one child per card. Every board has its own dense vector of
// live hands (a space); on five cards it is sorted weakest first, so showdowns cost O(hands) via
// per-card sums. Decision nodes own `actions * hands` regrets (and average strategy sums, if kept) at
// their offsets, laid out [action * hands + hand].
// With exact flop all-ins, an all-in called on a flop root's own street is a terminal too: its value takes
// allInBalance(), not the runouts' subtrees.
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
        allIn,
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
    std::size_t allIns = 0;                                   // nodes of kind allIn
    std::size_t depth = 0;

    // `root` is a decision (usually a street's first); ranges[p] holds player p's reach for every hand.
    // Options: hands whose reach is at most `minReach` times the largest in both ranges are left out (they
    // play uniformly). Average strategies are kept for root-street nodes only unless
    // `averageLaterStreets`. With `chanceSamples` > 0, each chance node deals only that many cards, drawn
    // at random (seeded by the board and history, so the tree is reproducible); `allInSamples` does the
    // same for the board that runs out after an all-in, whose value is all there is to those subtrees (0
    // deals every card). `exactFlopAllIns` values the all-ins on a flop root's own street over every
    // runout, with no subtree (nodes of kind allIn; ignored on other roots). `rootExtraAction` adds an
    // action at the root, after the legal ones (the child is G::applyTo(root, chips, number of legal
    // actions)): a real bet the abstraction lacks.
    SubgameTree(const typename G::State &root, const std::array<Hands, 2> &reach, const SubgameOptions &options = {})
        : m_averageLaterStreets(options.averageLaterStreets), m_chanceSamples(options.chanceSamples), m_allInSamples(options.allInSamples),
          m_exactFlopAllIns(options.exactFlopAllIns && std::popcount(root.board) == 3), m_rootExtraAction(options.rootExtraAction)
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

    // The hole index (see holeIndex) of root-space hand i.
    inline std::size_t holeOf(std::size_t i) const noexcept
    {
        return holeIndex((1ull << spaces.front().low[i]) | (1ull << spaces.front().high[i]));
    }

    // What the all-in nodes of a flop root pay: [i * hands + j] is the share of runouts where root-space
    // hand i beats hand j minus the share where it loses (antisymmetric). Takes ~70 ms on a thread's first
    // call for a flop (see flopAllInBalance).
    std::vector<float> allInBalance() const
    {
        const auto &balance = flopAllInBalance(nodes.front().state.board);
        std::vector<float> result(hands * hands);
        for (std::size_t i = 0; i < hands; ++i)
        {
            const std::int16_t *row = balance.data() + holeOf(i) * holeCombos;
            for (std::size_t j = 0; j < hands; ++j)
            {
                result[i * hands + j] = static_cast<float>(row[holeOf(j)]) / flopAllInRunouts;
            }
        }
        return result;
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
    bool m_exactFlopAllIns;
    std::optional<std::uint32_t> m_rootExtraAction;
    std::map<std::pair<std::uint64_t, std::size_t>, std::size_t> m_spaceOf; // by board and parent space

    // The space of `board`, one card more than the board of space `parentIndex`.
    std::size_t spaceFor(std::uint64_t board, std::size_t parentIndex)
    {
        if (const auto it = m_spaceOf.find({board, parentIndex}); it != m_spaceOf.end())
        {
            return it->second;
        }
        // The parent's hands the new card leaves alive, in the parent's order; on the river, weakest
        // first instead (ties in the parent's order): a counting sort by the board's ranks.
        const bool river = std::popcount(board) == 5;
        const Space &parent = spaces[parentIndex];
        const auto hole = [&](std::size_t j) { return (1ull << parent.low[j]) | (1ull << parent.high[j]); };
        Space space;
        if (river)
        {
            const auto &ranks = boardRanks(board);
            std::array<std::uint16_t, holeCombos + 1> first{}; // where each rank's run starts
            for (std::size_t j = 0; j < parent.size(); ++j)
            {
                if ((hole(j) & board) == 0)
                {
                    ++first[ranks[holeIndex(hole(j))] + 1];
                }
            }
            for (std::size_t r = 0; r < holeCombos; ++r)
            {
                if (first[r + 1] > 0)
                {
                    space.groups.push_back(first[r]);
                }
                first[r + 1] = static_cast<std::uint16_t>(first[r + 1] + first[r]);
            }
            space.parent.resize(first[holeCombos]);
            for (std::size_t j = 0; j < parent.size(); ++j)
            {
                if ((hole(j) & board) == 0)
                {
                    space.parent[first[ranks[holeIndex(hole(j))]]++] = static_cast<std::uint16_t>(j);
                }
            }
        }
        else
        {
            for (std::size_t j = 0; j < parent.size(); ++j)
            {
                if ((hole(j) & board) == 0)
                {
                    space.parent.push_back(static_cast<std::uint16_t>(j));
                }
            }
        }
        for (const std::uint16_t j : space.parent)
        {
            space.low.push_back(parent.low[j]);
            space.high.push_back(parent.high[j]);
        }
        space.groups.push_back(space.parent.size());
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
        if (m_exactFlopAllIns && s.street == G::showdown && s.board == rootBoard)
        {
            nodes[idx].kind = Kind::allIn;
            ++allIns;
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
                cards.push_back(lowestBit(c));
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
