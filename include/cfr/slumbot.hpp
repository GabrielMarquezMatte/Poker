#ifndef __POKER_CFR_SLUMBOT_HPP__
#define __POKER_CFR_SLUMBOT_HPP__
#include "subgame_solver.hpp"
#include <charconv>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Slumbot's game: blinds 50/100, 20,000-chip stacks (200bb). Our chips are Slumbot's / 50.
inline constexpr std::uint32_t slumbotChipsPerChip = 50;

// One action of a Slumbot action string: streets are separated by '/'; k check, c call, f fold, and
// b<n> bet or raise to n chips put in on this street (Slumbot's chips).
struct SlumbotAction
{
    std::uint8_t street = 0;
    char type = 'k';
    std::uint32_t size = 0; // b only
};

inline std::optional<std::vector<SlumbotAction>> parseSlumbotActions(std::string_view text)
{
    std::vector<SlumbotAction> actions;
    std::uint8_t street = 0;
    for (std::size_t i = 0; i < text.size();)
    {
        const char c = text[i++];
        if (c == '/')
        {
            ++street;
            continue;
        }
        if (c != 'k' && c != 'c' && c != 'f' && c != 'b')
        {
            return std::nullopt;
        }
        SlumbotAction action{street, c, 0};
        if (c == 'b')
        {
            const auto [end, error] = std::from_chars(text.data() + i, text.data() + text.size(), action.size);
            if (error != std::errc{})
            {
                return std::nullopt;
            }
            i = static_cast<std::size_t>(end - text.data());
        }
        actions.push_back(action);
    }
    return actions;
}

// Probability of translating a bet of pot fraction x to the smaller of its neighbours a < x < b in the
// abstraction (pseudo-harmonic mapping, Ganzfried & Sandholm 2013).
inline double pseudoHarmonicLow(double a, double b, double x)
{
    return (b - x) * (1.0 + a) / ((b - a) * (1.0 + x));
}

// A bet or raise to `size` chips on the street made legal under Slumbot's rules, given the street's
// highest bet, the last increment on the street and the most either player has in: raise by at least
// the last increment and a big blind, at most all-in (always allowed).
inline std::uint32_t slumbotLegalBet(std::uint32_t size, std::uint32_t betTo, std::uint32_t lastIncrement, std::uint32_t total)
{
    const std::uint32_t allIn = betTo + (20000 - total);
    const std::uint32_t minimum = std::min(betTo + std::max<std::uint32_t>(lastIncrement, 100), allIn);
    return std::clamp(size, minimum, allIn);
}

// A blueprint's preflop average strategy, copied out of its table: preflop is all the bot reads from
// the blueprint (postflop is resolved), and ~300 public nodes x 169 hand classes take ~2 MB instead of
// the whole table (7.5 GB at 200bb, whose untouched pages Windows pages out, then faults back in).
template <typename G>
class PreflopStrategy
{
public:
    using Strategy = typename Mccfr<G>::Strategy;

    explicit PreflopStrategy(const Mccfr<G> &blueprint) { copy(blueprint, G::initial()); }

    // Mccfr::averageStrategy for preflop keys: uniform where the blueprint has nothing.
    Strategy averageStrategy(std::uint64_t key, std::size_t numActions) const
    {
        if (const auto it = m_strategies.find(key); it != m_strategies.end())
        {
            return it->second;
        }
        Strategy uniform{};
        std::fill_n(uniform.begin(), numActions, 1.0 / static_cast<double>(numActions));
        return uniform;
    }

    inline std::size_t size() const noexcept { return m_strategies.size(); }

private:
    std::unordered_map<std::uint64_t, Strategy> m_strategies;

    void copy(const Mccfr<G> &blueprint, const typename G::State &s)
    {
        if (G::isTerminal(s) || s.street != 0)
        {
            return;
        }
        const std::size_t n = G::numActions(s);
        for (std::uint16_t bucket = 0; bucket < 169; ++bucket)
        {
            const std::uint64_t key = G::infosetKeyWithBucket(s, bucket);
            m_strategies.emplace(key, blueprint.averageStrategy(key, n));
        }
        for (std::size_t a = 0; a < n; ++a)
        {
            copy(blueprint, G::apply(s, a));
        }
    }
};

// The game the bot resolves in: the blueprint's, with a fourth raise per postflop street, all-in only.
// With the blueprint's three, a fourth raise (often Slumbot's all-in) had no counterpart to update its
// range with; allowing every size there more than doubled flop trees.
template <typename C>
struct PostflopRaises : C
{
    static constexpr auto maxRaises = []
    {
        auto raises = C::maxRaises;
        for (std::size_t street = 1; street < 4; ++street)
        {
            raises[street] = std::max<std::uint8_t>(raises[street], 4);
        }
        return raises;
    }();
    static constexpr std::array<std::uint8_t, 4> allInOnlyRaises{255, 4, 4, 4}; // the 4th raise of a postflop street
};

// Iterations of each resolve; flops deal `flopSamples` cards per chance node, and `allInSamples` per
// card of the runout after an all-in.
struct SlumbotResolves
{
    std::size_t flop = 60, turn = 60, river = 100;
    std::size_t flopSamples = 6;
    std::size_t allInSamples = 16; // per card of a flop all-in's runout: 36 runouts misjudged all-ins by ~8% equity
};

// Plays Slumbot's game with a blueprint of Hunl<C> (C::stack must be 400): the blueprint's preflop, with
// Slumbot's bets translated into the abstraction, and resolves after the flop. Postflop, a street is
// resolved from the real state (real pot and bets) and the ranges so far; the solution is followed
// while both players' actions are in its tree, and a new resolve starts from the real state after any
// action that is not. Ranges are updated with every action's probability under the model that
// produced it (the blueprint preflop, the current resolve postflop), for the opponent (range) and for
// ourselves (beliefs, the opponent's view of our hand, which seeds our resolves). An action the model
// can't produce leaves the range as it was.
// ponytail: unsafe resolving (Libratus-style safe nested solving would bound how exploitable re-solving
// after off-tree bets gets).
template <typename C, template <typename> class Resolver = SubgameSolver>
class SlumbotBot
{
public:
    using G = Hunl<C>;                  // the blueprint's game, for preflop
    using R = Hunl<PostflopRaises<C>>; // the real hand and the resolves
    using State = typename R::State;
    using Range = std::array<double, holeCombos>;
    static_assert(C::stack * slumbotChipsPerChip == 20000 && C::bigBlind * slumbotChipsPerChip == 100);

    SlumbotBot(const PreflopStrategy<G> &blueprint, SlumbotResolves resolves, std::uint64_t seed) : m_blueprint(blueprint), m_resolves(resolves), m_rng(seed) {}

    // `seat`: 0 small blind, 1 big blind. `hole`: our cards.
    void newHand(std::size_t seat, std::uint64_t hole)
    {
        m_seat = seat;
        m_hole = hole;
        m_real = R::initial();
        m_real.dealt = true;
        m_abstract = G::initial();
        m_abstract.dealt = true;
        m_abstractLost = false;
        m_seen = 0;
        m_streetStart = 0;
        m_slumbotBetTo = 100;
        m_slumbotIncrement = 50;
        m_slumbotTotal = 100;
        m_ourAbstract.reset();
        m_solvers = {};
        m_nudges = 0;
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            m_range[h] = (holes[h] & hole) == 0 ? 1.0 : 0.0;
            m_beliefs[h] = 1.0;
        }
    }

    // Replays the actions not seen yet (`board`: the cards dealt so far, in order) and returns the
    // increment to send if we are to act, or nothing.
    std::optional<std::string> update(const std::vector<SlumbotAction> &actions, const std::vector<std::uint64_t> &board)
    {
        for (; m_seen < actions.size(); ++m_seen)
        {
            if (R::isTerminal(m_real) || m_real.street >= 4)
            {
                return std::nullopt;
            }
            setBoard(board);
            observe(actions[m_seen]);
        }
        if (R::isTerminal(m_real) || m_real.street >= 4 || m_real.toAct != m_seat)
        {
            return std::nullopt;
        }
        setBoard(board);
        return decide();
    }

    inline const State &state() const noexcept { return m_real; }
    inline const Range &range() const noexcept { return m_range; }
    // Resolves this hand that kept our hand in the tree only thanks to the nudge in resolve().
    inline std::size_t nudges() const noexcept { return m_nudges; }

private:
    using FlopSolver = Resolver<CoarseTurn<PostflopRaises<C>>>;
    using TurnSolver = Resolver<CoarseRiver<PostflopRaises<C>>>;
    using RiverSolver = SubgameSolver<PostflopRaises<C>>;
    using Strategy = typename Mccfr<G>::Strategy;
    static constexpr double minReach = 1e-3;
    static constexpr std::array<int, 5> boardSize{0, 3, 4, 5, 5};

    // The current street's resolve and the node of its tree matching the real state, while on the tree.
    struct Solvers
    {
        std::unique_ptr<FlopSolver> flop;
        std::unique_ptr<TurnSolver> turn;
        std::unique_ptr<RiverSolver> river;
        State node{};
        bool onTree = false;
    };

    const PreflopStrategy<G> &m_blueprint;
    SlumbotResolves m_resolves;
    CfrRng m_rng;
    std::size_t m_seat = 0;
    std::uint64_t m_hole = 0;
    State m_real{};                       // real chips
    typename G::State m_abstract{};       // the blueprint's translation (preflop)
    bool m_abstractLost = false;  // the translation ran out of actions (more raises than the blueprint allows)
    std::size_t m_seen = 0;
    std::uint32_t m_streetStart = 0; // chips each player had in when the street began
    // Slumbot's own accounting, exact (ours rounds to 50-chip units), for its minimum raise: the street's
    // highest bet, the last bet or raise increment on the street, and the most either player has in.
    std::uint32_t m_slumbotBetTo = 0, m_slumbotIncrement = 0, m_slumbotTotal = 0;
    std::optional<std::size_t> m_ourAbstract; // the blueprint action behind our pending preflop action
    Range m_range{}, m_beliefs{};
    Solvers m_solvers;
    std::size_t m_nudges = 0;

    double uniform() { return static_cast<double>(m_rng() >> 11) * 0x1.0p-53; }

    void setBoard(const std::vector<std::uint64_t> &board)
    {
        std::uint64_t mask = 0;
        for (int i = 0; i < boardSize[std::min<std::size_t>(m_real.street, 4)] && i < static_cast<int>(board.size()); ++i)
        {
            mask |= board[static_cast<std::size_t>(i)];
        }
        if (mask == m_real.board)
        {
            return;
        }
        m_real.board = mask;
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if ((holes[h] & mask) != 0)
            {
                m_range[h] = 0.0;
                m_beliefs[h] = 0.0;
            }
        }
    }

    // Our chips put in in total after a Slumbot action by the player to act.
    std::uint32_t target(const SlumbotAction &action) const
    {
        const std::uint32_t facing = std::max(m_real.invested[0], m_real.invested[1]);
        if (action.type == 'f')
        {
            return G::fold;
        }
        if (action.type != 'b')
        {
            return facing;
        }
        const auto chips = static_cast<std::uint32_t>(std::lround(static_cast<double>(action.size) / slumbotChipsPerChip));
        const std::uint32_t minimum = std::min(facing + m_real.lastRaise, C::stack); // rounding can undercut the minimum raise
        return std::clamp(m_streetStart + chips, minimum, C::stack);
    }

    // The actions of `s` standing for putting `to` in, with their weights: fold and check/call exactly,
    // a raise by its fraction of the pot after calling in the real game (`fraction`).
    template <typename Game>
    static std::vector<std::pair<std::size_t, double>> translate(const typename Game::State &s, std::uint32_t to, double fraction)
    {
        const auto legal = Game::legalActions(s);
        const std::uint32_t facing = std::max(s.invested[0], s.invested[1]);
        std::vector<std::pair<std::size_t, double>> raises;
        for (std::size_t a = 0; a < legal.size; ++a)
        {
            if (legal.to[a] == to && to != G::fold && to <= facing)
            {
                return {{a, 1.0}}; // check or call
            }
            if (to == G::fold && legal.to[a] == G::fold)
            {
                return {{a, 1.0}};
            }
            if (legal.to[a] != G::fold && legal.to[a] > facing)
            {
                raises.emplace_back(a, (static_cast<double>(legal.to[a]) - facing) / (2.0 * facing));
            }
        }
        if (raises.empty() || to == G::fold || to <= facing)
        {
            return {};
        }
        if (fraction <= raises.front().second)
        {
            return {{raises.front().first, 1.0}};
        }
        if (fraction >= raises.back().second)
        {
            return {{raises.back().first, 1.0}};
        }
        for (std::size_t k = 0; k + 1 < raises.size(); ++k)
        {
            const auto [a, fa] = raises[k];
            const auto [b, fb] = raises[k + 1];
            if (fraction <= fb)
            {
                const double low = pseudoHarmonicLow(fa, fb, fraction);
                return {{a, low}, {b, 1.0 - low}};
            }
        }
        return {{raises.back().first, 1.0}};
    }

    Strategy blueprintStrategy(const typename G::State &s, std::size_t hand) const
    {
        const auto bucket = static_cast<std::uint16_t>(preflopClassIndex(Deck::from_mask(holes[hand])));
        return m_blueprint.averageStrategy(G::infosetKeyWithBucket(s, bucket), G::numActions(s));
    }

    Strategy resolvedStrategy(std::size_t hand) const
    {
        const Solvers &solvers = m_solvers;
        if (solvers.flop != nullptr)
        {
            return solvers.flop->strategy(solvers.node, hand);
        }
        if (solvers.turn != nullptr)
        {
            return solvers.turn->strategy(solvers.node, hand);
        }
        return solvers.river->strategy(solvers.node, hand);
    }

    // Resolves the current street from the real state and the ranges so far.
    void resolve()
    {
        std::array<Range, 2> ranges{};
        ranges[1 - m_seat] = m_range;
        ranges[m_seat] = m_beliefs;
        for (auto &range : ranges)
        {
            if (std::accumulate(range.begin(), range.end(), 0.0) <= 0.0)
            {
                range.fill(1.0); // a player left the model's support: assume any hand
            }
        }
        // Our own hand must stay in the tree, or we would play its uniform default: lines the model finds
        // unlikely for it can take its weight below the cut (minReach) that drops hands to save time. The
        // nudge is negligible to the opponent's view of our range; CFR's strategy for a hand doesn't
        // depend on its own reach.
        auto &mine = ranges[m_seat][holeIndex(m_hole)];
        const double kept = 2.0 * minReach * *std::max_element(ranges[m_seat].begin(), ranges[m_seat].end());
        m_nudges += mine < kept ? 1 : 0;
        mine = std::max(mine, kept);
        m_solvers = {};
        State root = m_real;
        root.history = 0x51u; // any: solver nodes are keyed by the actions from here
        m_solvers.node = root;
        m_solvers.onTree = true;
        if (m_real.street == 1)
        {
            m_solvers.flop = std::make_unique<FlopSolver>(std::bit_cast<typename FlopSolver::G::State>(root), ranges, false, minReach, m_resolves.flopSamples,
                                                          m_resolves.allInSamples);
            m_solvers.flop->solve(m_resolves.flop);
        }
        else if (m_real.street == 2)
        {
            m_solvers.turn = std::make_unique<TurnSolver>(std::bit_cast<typename TurnSolver::G::State>(root), ranges, false, minReach);
            m_solvers.turn->solve(m_resolves.turn);
        }
        else
        {
            m_solvers.river = std::make_unique<RiverSolver>(root, ranges, true, minReach);
            m_solvers.river->solve(m_resolves.river);
        }
    }

    // Multiplies `weights` by each hand's probability of an action standing for `options` under
    // `strategy`, unless no hand would take it (or nothing stands for it): that leaves the range as it was
    // rather than emptying it, which the next resolve would read as "any hand".
    template <typename F>
    static void condition(Range &weights, const std::vector<std::pair<std::size_t, double>> &options, F strategy)
    {
        if (options.empty())
        {
            return;
        }
        Range updated = weights;
        double total = 0.0;
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if (updated[h] > 0.0)
            {
                const auto sigma = strategy(h);
                double p = 0.0;
                for (const auto &[a, w] : options)
                {
                    p += w * sigma[a];
                }
                updated[h] *= p;
                total += updated[h];
            }
        }
        if (total > 0.0)
        {
            weights = updated;
        }
    }

    void observe(const SlumbotAction &action)
    {
        const std::size_t actor = m_real.toAct;
        const std::uint32_t to = target(action);
        const std::uint32_t facing = std::max(m_real.invested[0], m_real.invested[1]);
        const double fraction = to == G::fold ? 0.0 : (static_cast<double>(to) - facing) / (2.0 * facing);
        Range &weights = actor == m_seat ? m_beliefs : m_range;
        if (m_real.street == 0)
        {
            auto options = m_abstractLost ? std::vector<std::pair<std::size_t, double>>{} : translate<G>(m_abstract, to, fraction);
            if (actor == m_seat && m_ourAbstract.has_value())
            {
                options = {{*m_ourAbstract, 1.0}}; // exactly what we chose, not a translation of its rounding
            }
            m_ourAbstract.reset();
            if (options.empty())
            {
                m_abstractLost = true; // ponytail: ranges stop updating until the flop
            }
            else
            {
                condition(weights, options, [&](std::size_t h) { return blueprintStrategy(m_abstract, h); });
                double r = uniform();
                std::size_t chosen = options.back().first;
                for (const auto &[a, w] : options)
                {
                    if ((r -= w) < 0.0)
                    {
                        chosen = a;
                        break;
                    }
                }
                m_abstract = G::apply(m_abstract, chosen);
            }
        }
        else
        {
            if (!m_solvers.onTree)
            {
                resolve();
            }
            condition(weights, translate<R>(m_solvers.node, to, fraction), [&](std::size_t h) { return resolvedStrategy(h); });
            const auto legal = R::legalActions(m_solvers.node);
            m_solvers.onTree = false;
            for (std::size_t a = 0; a < legal.size; ++a)
            {
                if (legal.to[a] == to)
                {
                    m_solvers.node = R::apply(m_solvers.node, a);
                    m_solvers.onTree = m_solvers.node.street == m_real.street; // a new street is resolved afresh
                    break;
                }
            }
        }
        const std::uint8_t street = m_real.street;
        m_real = R::applyTo(m_real, to, m_seen);
        if (action.type == 'b')
        {
            m_slumbotIncrement = action.size - m_slumbotBetTo;
            m_slumbotTotal += m_slumbotIncrement;
            m_slumbotBetTo = action.size;
        }
        else
        {
            m_slumbotIncrement = 0;
        }
        if (m_real.street != street)
        {
            m_streetStart = m_real.invested[0];
            m_slumbotBetTo = 0;
            m_solvers = {};
        }
    }

    std::string increment(std::uint32_t to) const
    {
        const std::uint32_t facing = m_real.invested[1 - m_seat];
        if (to == G::fold)
        {
            return "f";
        }
        if (to <= facing)
        {
            return facing == m_real.invested[m_seat] ? "k" : "c";
        }
        // On Slumbot's exact chips: ours are rounded.
        const std::uint32_t size = to == C::stack ? 20000 : (to - m_streetStart) * slumbotChipsPerChip;
        return "b" + std::to_string(slumbotLegalBet(size, m_slumbotBetTo, m_slumbotIncrement, m_slumbotTotal));
    }

    std::string decide()
    {
        const std::size_t hand = holeIndex(m_hole);
        if (m_real.street == 0)
        {
            if (m_abstractLost)
            {
                return increment(m_real.invested[1 - m_seat]); // ponytail: call once the blueprint can't follow
            }
            const auto sigma = blueprintStrategy(m_abstract, hand);
            const auto legal = G::legalActions(m_abstract);
            const std::size_t a = sample(sigma, legal.size);
            m_ourAbstract = a;
            const std::uint32_t abstractTo = legal.to[a];
            const std::uint32_t abstractFacing = std::max(m_abstract.invested[0], m_abstract.invested[1]);
            if (abstractTo == G::fold || abstractTo <= abstractFacing)
            {
                return increment(abstractTo == G::fold ? G::fold : m_real.invested[1 - m_seat]);
            }
            // The same pot fraction in the real game.
            const std::uint32_t facing = std::max(m_real.invested[0], m_real.invested[1]);
            std::uint32_t to = C::stack;
            if (abstractTo < C::stack)
            {
                const double fraction = (static_cast<double>(abstractTo) - abstractFacing) / (2.0 * abstractFacing);
                to = static_cast<std::uint32_t>(std::lround(facing + fraction * 2.0 * facing));
            }
            return increment(std::clamp(to, std::min(facing + m_real.lastRaise, C::stack), C::stack));
        }
        if (!m_solvers.onTree)
        {
            resolve();
        }
        const auto legal = R::legalActions(m_solvers.node);
        return increment(legal.to[sample(resolvedStrategy(hand), legal.size)]);
    }

    std::size_t sample(const Strategy &sigma, std::size_t n)
    {
        double r = uniform();
        for (std::size_t a = 0; a + 1 < n; ++a)
        {
            if ((r -= sigma[a]) < 0.0)
            {
                return a;
            }
        }
        return n - 1;
    }
};
#endif // __POKER_CFR_SLUMBOT_HPP__
