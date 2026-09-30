#ifndef __POKER_CFR_LBR_HPP__
#define __POKER_CFR_LBR_HPP__
#include "push_fold.hpp"
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

// Local best response (Lisy & Bowling 2017): plays real hands against a strategy's average policy,
// tracking the opponent's range, and at each decision picks the action with the best value assuming
// both players then check/call to showdown. Its winnings lower-bound the strategy's exploitability.
// It uses the game's own bet sizes, so it never leaves the strategy's betting tree.
template <typename C>
class LocalBestResponse
{
public:
    using G = Hunl<C>;
    using Range = std::array<double, holeCombos>;

    static constexpr unsigned allStreets = 0b1111;

    // Flop equities average `flopRunouts` sampled turn/river pairs; turn and river are exact.
    // LBR best-responds only on streets in `streets` (bit 0 preflop .. bit 3 river) and plays the
    // strategy itself elsewhere, which splits exploitability by street (0 = self-play).
    LocalBestResponse(const Mccfr<G> &opponent, const PreflopEquity &preflop, std::size_t flopRunouts = 100, unsigned streets = allStreets)
        : m_opponent(opponent), m_preflop(preflop), m_flopRunouts(flopRunouts), m_streets(streets) {}

    // LBR's winnings in chips for one hand played from `seat` (0 small blind, 1 big blind).
    double playHand(std::size_t seat, CfrRng &rng) const
    {
        typename G::State s = G::sampleChance(G::initial(), rng);
        const std::uint64_t hero = s.hole[seat];
        Range range{};
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            range[h] = (holes[h] & hero) == 0 ? 1.0 : 0.0;
        }
        std::array<std::uint16_t, holeCombos> buckets{};
        int bucketStreet = -1;
        while (!G::isTerminal(s))
        {
            if (G::isChance(s))
            {
                s = G::sampleChance(s, rng);
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    range[h] = (holes[h] & s.board) == 0 ? range[h] : 0.0;
                }
                continue;
            }
            if (bucketStreet != s.street)
            {
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    buckets[h] = range[h] > 0.0 ? G::bucketFor(s, holes[h]) : 0;
                }
                bucketStreet = s.street;
            }
            const std::size_t n = G::numActions(s);
            std::size_t action = 0;
            if (s.toAct == seat && (m_streets >> s.street & 1u) == 0)
            {
                action = sample(m_opponent.averageStrategy(G::infosetKey(s), n), n, rng);
            }
            else if (s.toAct == seat)
            {
                action = choose(s, seat, range, buckets, rng);
            }
            else
            {
                action = sample(m_opponent.averageStrategy(G::infosetKey(s), n), n, rng);
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    if (range[h] > 0.0)
                    {
                        range[h] *= m_opponent.averageStrategy(G::infosetKeyWithBucket(s, buckets[h]), n)[action];
                    }
                }
            }
            s = G::apply(s, action);
        }
        return G::utility(s, seat);
    }

    struct Result
    {
        std::array<double, 2> mbbPerHand{};    // per LBR seat
        std::array<double, 2> standardError{}; // mbb/hand
        inline double average() const noexcept { return (mbbPerHand[0] + mbbPerHand[1]) / 2.0; }
        inline double averageError() const noexcept { return std::hypot(standardError[0], standardError[1]) / 2.0; }
    };

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

    std::size_t choose(const typename G::State &s, std::size_t seat, const Range &range, const std::array<std::uint16_t, holeCombos> &buckets, CfrRng &rng) const
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
                const std::size_t n = G::numActions(child);
                Range callers{};
                double total = 0.0, folded = 0.0;
                for (std::size_t h = 0; h < holeCombos; ++h)
                {
                    if (range[h] > 0.0)
                    {
                        const double foldProbability = m_opponent.averageStrategy(G::infosetKeyWithBucket(child, buckets[h]), n)[0];
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
