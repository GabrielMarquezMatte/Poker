#ifndef __POKER_CFR_LBR_HPP__
#define __POKER_CFR_LBR_HPP__
#include "subgame_solver.hpp"
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

// LBR winnings in mbb/hand.
struct LbrResult
{
    std::array<double, 2> mbbPerHand{};    // per LBR seat
    std::array<double, 2> standardError{}; // mbb/hand
    inline double average() const noexcept { return (mbbPerHand[0] + mbbPerHand[1]) / 2.0; }
    inline double averageError() const noexcept { return std::hypot(standardError[0], standardError[1]) / 2.0; }
};

// Which streets the evaluated strategy resolves, and with how many DCFR iterations (0 = play the
// blueprint there). Each resolve starts from the ranges the strategy implies so far and solves to
// showdown: rivers exactly, turns with a coarser river (CoarseRiver), flops with coarser turn and river
// betting (CoarseTurn) dealing only `flopSamples` cards at each chance node. Later streets are resolved
// again when reached.
struct LbrResolves
{
    std::size_t flop = 0, turn = 0, river = 0;
    std::size_t flopSamples = 8;
    std::size_t allInSamples = 16; // per card of a flop all-in's runout
};

// Local best response (Lisy & Bowling 2017): plays real hands against a strategy's average policy,
// tracking the opponent's range, and at each decision picks the action with the best value assuming
// both players then check/call to showdown. Its winnings lower-bound the strategy's exploitability.
// It uses the game's own bet sizes, so it never leaves the strategy's betting tree.
// Flops and turns are resolved with `Resolver`: SubgameSolver on the CPU, or GpuSubgameSolver. Rivers
// always use SubgameSolver (their trees are too small for the GPU to win).
template <typename C, template <typename> class Resolver = SubgameSolver>
class LocalBestResponse
{
public:
    using G = Hunl<C>;
    using Range = std::array<double, holeCombos>;

    static constexpr unsigned allStreets = 0b1111;

    // Flop equities average `flopRunouts` sampled turn/river pairs; turn and river are exact.
    // LBR best-responds only on streets in `streets` (bit 0 preflop .. bit 3 river) and plays the
    // strategy itself elsewhere, which splits exploitability by street (0 = self-play).
    // `resolves` says where the strategy resolves instead of playing its blueprint.
    LocalBestResponse(const Mccfr<G> &opponent, const PreflopEquity &preflop, std::size_t flopRunouts = 100, unsigned streets = allStreets,
                      LbrResolves resolves = {})
        : m_opponent(opponent), m_preflop(preflop), m_flopRunouts(flopRunouts), m_streets(streets), m_resolves(resolves) {}

    // LBR's winnings in chips for one hand played from `seat` (0 small blind, 1 big blind).
    double playHand(std::size_t seat, CfrRng &rng) const
    {
        typename G::State s = G::sampleChance(G::initial(), rng);
        const std::size_t opp = 1 - seat;
        const std::uint64_t hero = s.hole[seat];
        const bool resolving = m_resolves.flop > 0 || m_resolves.turn > 0 || m_resolves.river > 0;
        // range: the opponent's hands as LBR sees them. beliefs: LBR's hands as the opponent's strategy
        // sees them, which seeds its resolves.
        Range range{}, beliefs{};
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            range[h] = (holes[h] & hero) == 0 ? 1.0 : 0.0;
            beliefs[h] = 1.0;
        }
        std::array<std::uint16_t, holeCombos> buckets{};
        int bucketStreet = -1;
        Solvers solvers;
        while (!G::isTerminal(s))
        {
            if (G::isChance(s))
            {
                s = G::sampleChance(s, rng);
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    const bool blocked = (holes[h] & s.board) != 0;
                    range[h] = blocked ? 0.0 : range[h];
                    beliefs[h] = blocked ? 0.0 : beliefs[h];
                }
                continue;
            }
            if (bucketStreet != s.street)
            {
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    buckets[h] = range[h] > 0.0 || (resolving && beliefs[h] > 0.0) ? G::bucketFor(s, holes[h]) : 0;
                }
                bucketStreet = s.street;
            }
            // The lookahead games share the state layout and the resolved street's betting. Averages are
            // kept for that street only: later streets are resolved again.
            if (m_resolves.flop > 0 && s.street == 1 && solvers.flop == nullptr)
            {
                solvers.flop = std::make_unique<FlopSolver>(std::bit_cast<typename FlopSolver::G::State>(s), resolveRanges(seat, range, beliefs), false,
                                                            minReach, m_resolves.flopSamples, m_resolves.allInSamples);
                solvers.flop->solve(m_resolves.flop);
            }
            if (m_resolves.turn > 0 && s.street == 2 && solvers.turn == nullptr)
            {
                solvers.turn = std::make_unique<TurnSolver>(std::bit_cast<typename TurnSolver::G::State>(s), resolveRanges(seat, range, beliefs), false,
                                                            minReach);
                solvers.turn->solve(m_resolves.turn);
            }
            if (m_resolves.river > 0 && s.street == 3 && solvers.river == nullptr)
            {
                solvers.river = std::make_unique<SubgameSolver<C>>(s, resolveRanges(seat, range, beliefs));
                solvers.river->solve(m_resolves.river);
            }
            const std::size_t n = G::numActions(s);
            std::size_t action = 0;
            if (s.toAct == seat && (m_streets >> s.street & 1u) != 0)
            {
                action = choose(s, seat, range, buckets, solvers, rng);
            }
            else
            {
                action = sample(policy(s, holeIndex(s.hole[s.toAct]), s.bucket[s.street][s.toAct], solvers), n, rng);
            }
            if (s.toAct == opp)
            {
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    if (range[h] > 0.0)
                    {
                        range[h] *= policy(s, h, buckets[h], solvers)[action];
                    }
                }
            }
            else if (resolving && solvers.river == nullptr)
            {
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    if (beliefs[h] > 0.0)
                    {
                        beliefs[h] *= policy(s, h, buckets[h], solvers)[action];
                    }
                }
            }
            s = G::apply(s, action);
        }
        return s.folder == G::nobody ? showdownValue(s, seat, range) : G::utility(s, seat);
    }

    using Result = LbrResult;

    Result evaluate(std::uint64_t handsPerSeat, std::size_t threads, std::uint64_t seed) const
    {
        std::vector<std::array<double, 4>> sums(threads); // per thread: sum and sum of squares for each seat
        {
            std::vector<std::jthread> workers;
            for (std::size_t t = 0; t < threads; ++t)
            {
                workers.emplace_back([&, t]
                                     {
                    CfrRng rng{seed * 1000 + t};
                    for (std::uint64_t i = t; i < handsPerSeat; i += threads)
                    {
                        for (std::size_t seat = 0; seat < 2; ++seat)
                        {
                            const double won = playHand(seat, rng);
                            sums[t][2 * seat] += won;
                            sums[t][2 * seat + 1] += won * won;
                        }
                    } });
            }
        }
        Result result;
        const double n = static_cast<double>(handsPerSeat);
        const double toMbb = 1000.0 / C::bigBlind;
        for (std::size_t seat = 0; seat < 2; ++seat)
        {
            double sum = 0.0, squares = 0.0;
            for (const auto &s : sums)
            {
                sum += s[2 * seat];
                squares += s[2 * seat + 1];
            }
            const double mean = sum / n;
            result.mbbPerHand[seat] = mean * toMbb;
            result.standardError[seat] = std::sqrt(std::max(squares / n - mean * mean, 0.0) / n) * toMbb;
        }
        return result;
    }

private:
    const Mccfr<G> &m_opponent;
    const PreflopEquity &m_preflop;
    std::size_t m_flopRunouts;
    unsigned m_streets;
    LbrResolves m_resolves;

    using FlopSolver = Resolver<CoarseTurn<C>>;
    using TurnSolver = Resolver<CoarseRiver<C>>;
    static_assert(sizeof(typename FlopSolver::G::State) == sizeof(typename G::State));
    static constexpr double minReach = 1e-3; // turn resolves leave out hands this unlikely for both players
    static_assert(sizeof(typename TurnSolver::G::State) == sizeof(typename G::State));
    // The strategy's resolves in the current hand, if any.
    struct Solvers
    {
        std::unique_ptr<FlopSolver> flop;
        std::unique_ptr<TurnSolver> turn;
        std::unique_ptr<SubgameSolver<C>> river;
    };

    static std::array<Range, 2> resolveRanges(std::size_t seat, const Range &range, const Range &beliefs)
    {
        std::array<Range, 2> ranges{};
        ranges[1 - seat] = range;
        ranges[seat] = beliefs;
        if (std::accumulate(beliefs.begin(), beliefs.end(), 0.0) <= 0.0)
        {
            ranges[seat].fill(1.0); // LBR left the strategy's support: assume any hand
        }
        return ranges;
    }

    // The strategy's policy at `s` for the player to act holding `hand` (whose bucket is `bucket`).
    typename Mccfr<G>::Strategy policy(const typename G::State &s, std::size_t hand, std::uint16_t bucket, const Solvers &solvers) const
    {
        if (s.street == 3 && solvers.river != nullptr && solvers.river->contains(s))
        {
            return solvers.river->strategy(s, hand);
        }
        if (s.street == 2 && solvers.turn != nullptr && solvers.turn->contains(s))
        {
            return solvers.turn->strategy(s, hand);
        }
        if (s.street == 1 && solvers.flop != nullptr && solvers.flop->contains(s))
        {
            return solvers.flop->strategy(s, hand);
        }
        return m_opponent.averageStrategy(G::infosetKeyWithBucket(s, bucket), G::numActions(s));
    }

    // At a showdown, the opponent's hand is distributed as `range` given everything public (its strategy
    // only sees its hand and public information), so LBR's expected winnings over that range are an
    // unbiased stand-in for the actual result, without the variance of which hand it held.
    static double showdownValue(const typename G::State &s, std::size_t seat, const Range &range)
    {
        const ClassificationResult mine = Hand::classify(Deck::from_mask(s.hole[seat] | s.board));
        double total = 0.0, won = 0.0, lost = 0.0;
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if (range[h] > 0.0)
            {
                const ClassificationResult theirs = Hand::classify(Deck::from_mask(holes[h] | s.board));
                total += range[h];
                won += mine > theirs ? range[h] : 0.0;
                lost += mine < theirs ? range[h] : 0.0;
            }
        }
        if (total <= 0.0)
        {
            return G::utility(s, seat); // underflowed range
        }
        return (won * s.invested[1 - seat] - lost * s.invested[seat]) / total;
    }

    static std::size_t sample(const typename Mccfr<G>::Strategy &sigma, std::size_t n, CfrRng &rng)
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

    // Per opponent hand: P(hero wins) + P(tie) / 2 over the remaining board cards.
    Range winShares(const typename G::State &s, std::uint64_t hero, const Range &range, CfrRng &rng) const
    {
        Range share{};
        if (s.street == 0)
        {
            const std::size_t mine = holeIndex(hero);
            for (std::size_t h = 0; h < holeCombos; ++h)
            {
                share[h] = m_preflop(mine, h);
            }
            return share;
        }
        const std::uint64_t live = ((1ull << 52) - 1) & ~(hero | s.board);
        std::vector<std::uint64_t> boards;
        const int missing = 5 - std::popcount(s.board);
        if (missing == 0)
        {
            boards.push_back(s.board);
        }
        else if (missing == 1)
        {
            for (std::uint64_t c = live; c != 0; c &= c - 1)
            {
                boards.push_back(s.board | (c & (~c + 1)));
            }
        }
        else
        {
            for (std::size_t i = 0; i < m_flopRunouts; ++i)
            {
                Deck deck = Deck::from_mask(live);
                boards.push_back(s.board | deck.popRandomCards(rng, 2).getMask());
            }
        }
        Range counted{};
        for (const std::uint64_t board : boards)
        {
            const ClassificationResult mine = Hand::classify(Deck::from_mask(hero | board));
            for (std::size_t h = 0; h < holeCombos; ++h)
            {
                if (range[h] > 0.0 && (holes[h] & board) == 0)
                {
                    const ClassificationResult theirs = Hand::classify(Deck::from_mask(holes[h] | board));
                    share[h] += mine > theirs ? 1.0 : (mine == theirs ? 0.5 : 0.0);
                    counted[h] += 1.0;
                }
            }
        }
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            share[h] = counted[h] > 0.0 ? share[h] / counted[h] : 0.5;
        }
        return share;
    }

    std::size_t choose(const typename G::State &s, std::size_t seat, const Range &range, const std::array<std::uint16_t, holeCombos> &buckets,
                       const Solvers &solvers, CfrRng &rng) const
    {
        const std::size_t opp = 1 - seat;
        const Range share = winShares(s, s.hole[seat], range, rng);
        const auto equity = [&](const Range &weights)
        {
            double won = 0.0, total = 0.0;
            for (std::size_t h = 0; h < holeCombos; ++h)
            {
                won += weights[h] * share[h];
                total += weights[h];
            }
            return total > 0.0 ? won / total : 0.5;
        };
        const auto legal = G::legalActions(s);
        std::size_t best = 0;
        double bestValue = -std::numeric_limits<double>::infinity();
        for (std::size_t a = 0; a < legal.size; ++a)
        {
            const std::uint32_t to = legal.to[a];
            double value = 0.0;
            if (to == G::fold)
            {
                value = -static_cast<double>(s.invested[seat]);
            }
            else if (to <= s.invested[opp])
            {
                value = (2.0 * equity(range) - 1.0) * s.invested[opp]; // check or call, then showdown
            }
            else
            {
                // Raise: the opponent folds per its strategy, otherwise calls down.
                const auto child = G::apply(s, a);
                Range callers{};
                double total = 0.0, folded = 0.0;
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    if (range[h] > 0.0)
                    {
                        const double foldProbability = policy(child, h, buckets[h], solvers)[0];
                        total += range[h];
                        folded += range[h] * foldProbability;
                        callers[h] = range[h] * (1.0 - foldProbability);
                    }
                }
                const double foldShare = total > 0.0 ? folded / total : 0.0;
                value = foldShare * s.invested[opp] + (1.0 - foldShare) * (2.0 * equity(callers) - 1.0) * to;
            }
            if (value > bestValue)
            {
                bestValue = value;
                best = a;
            }
        }
        return best;
    }
};
#endif // __POKER_CFR_LBR_HPP__
