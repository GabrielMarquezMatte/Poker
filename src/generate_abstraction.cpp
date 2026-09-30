// Builds postflop card abstraction tables (see include/cfr/card_abstraction.hpp).
// River: equity vs a uniform random hand, bucketed by quantile.
// Turn/flop: histogram of river equity over all runouts, clustered by k-means under EMD
// (for 1-D histograms EMD is the L1 distance between CDFs).
#include "../include/cfr/card_abstraction.hpp"
#include "../include/deck.hpp"
#include "../include/hand.hpp"
#include "../include/random.hpp"
#include <BS_thread_pool.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

constexpr std::size_t bins = 32;
using Cdf = std::array<float, bins>;
constexpr std::uint64_t fullDeck = (1ull << 52) - 1;

static std::uint16_t riverEquity(std::uint64_t hole, std::uint64_t board)
{
    const ClassificationResult mine = Hand::classify(Deck::from_mask(hole | board));
    std::uint32_t points = 0; // 2 per win, 1 per tie
    std::uint32_t total = 0;
    for (std::uint64_t a = fullDeck & ~(hole | board); a != 0; a &= a - 1)
    {
        const std::uint64_t first = a & (~a + 1);
        for (std::uint64_t b = a & (a - 1); b != 0; b &= b - 1)
        {
            const ClassificationResult theirs = Hand::classify(Deck::from_mask(first | (b & (~b + 1)) | board));
            points += mine > theirs ? 2u : (mine == theirs ? 1u : 0u);
            total += 2;
        }
    }
    return static_cast<std::uint16_t>(std::uint64_t{points} * 65535 / total);
}

static float l1(const Cdf &a, const Cdf &b)
{
    float d = 0.0f;
    for (std::size_t i = 0; i < bins; ++i)
    {
        d += std::abs(a[i] - b[i]);
    }
    return d;
}

static std::size_t nearest(const Cdf &point, const std::vector<Cdf> &centroids)
{
    std::size_t best = 0;
    float bestDistance = std::numeric_limits<float>::max();
    for (std::size_t c = 0; c < centroids.size(); ++c)
    {
        const float d = l1(point, centroids[c]);
        if (d < bestDistance)
        {
            bestDistance = d;
            best = c;
        }
    }
    return best;
}

// ponytail: mean update minimizes L2, not L1 (k-medians would be exact); fine for bucketing.
static std::vector<Cdf> kmeans(const std::vector<Cdf> &points, std::size_t k, std::size_t iterations, omp::XoroShiro128Plus &rng, BS::thread_pool<BS::tp::none> &pool)
{
    // k-means++ seeding.
    std::vector<Cdf> centroids{points[rng() % points.size()]};
    std::vector<float> distance(points.size(), std::numeric_limits<float>::max());
    while (centroids.size() < k)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            distance[i] = std::min(distance[i], l1(points[i], centroids.back()));
            sum += distance[i];
        }
        double target = static_cast<double>(rng() >> 11) * 0x1.0p-53 * sum;
        std::size_t pick = 0;
        while (pick + 1 < points.size() && (target -= distance[pick]) > 0.0)
        {
            ++pick;
        }
        centroids.push_back(points[pick]);
    }
    std::vector<std::size_t> assignment(points.size());
    for (std::size_t it = 0; it < iterations; ++it)
    {
        pool.detach_blocks<std::size_t>(0, points.size(), [&](std::size_t begin, std::size_t end)
                                        {
            for (std::size_t i = begin; i < end; ++i)
            {
                assignment[i] = nearest(points[i], centroids);
            } });
        pool.wait();
        std::vector<std::array<double, bins>> sums(k);
        std::vector<std::size_t> counts(k);
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            ++counts[assignment[i]];
            for (std::size_t b = 0; b < bins; ++b)
            {
                sums[assignment[i]][b] += points[i][b];
            }
        }
        for (std::size_t c = 0; c < k; ++c)
        {
            for (std::size_t b = 0; b < bins && counts[c] > 0; ++b)
            {
                centroids[c][b] = static_cast<float>(sums[c][b] / static_cast<double>(counts[c]));
            }
        }
    }
    // Weakest first: a larger CDF mass means lower equity.
    std::sort(centroids.begin(), centroids.end(), [](const Cdf &a, const Cdf &b)
              { return std::accumulate(a.begin(), a.end(), 0.0f) > std::accumulate(b.begin(), b.end(), 0.0f); });
    return centroids;
}

// CDF of river equity over every runout of (hole, board).
static Cdf runoutCdf(std::uint64_t hole, std::uint64_t board, const HandIndexer &river, const std::vector<std::uint16_t> &equity)
{
    std::array<std::uint32_t, bins> histogram{};
    std::uint32_t total = 0;
    const auto add = [&](std::uint64_t fullBoard)
    {
        ++histogram[equity[river.index({hole, fullBoard})] * bins / 65536];
        ++total;
    };
    const std::uint64_t rest = fullDeck & ~(hole | board);
    for (std::uint64_t a = rest; a != 0; a &= a - 1)
    {
        const std::uint64_t first = a & (~a + 1);
        if (std::popcount(board) == 4)
        {
            add(board | first);
            continue;
        }
        for (std::uint64_t b = a & (a - 1); b != 0; b &= b - 1)
        {
            add(board | first | (b & (~b + 1)));
        }
    }
    Cdf cdf{};
    std::uint32_t running = 0;
    for (std::size_t i = 0; i < bins; ++i)
    {
        running += histogram[i];
        cdf[i] = static_cast<float>(running) / static_cast<float>(total);
    }
    return cdf;
}

static void clusterStreet(std::size_t street, std::size_t k, std::size_t sampleSize, CardAbstraction &abstraction,
                          const std::vector<std::uint16_t> &equity, omp::XoroShiro128Plus &rng, BS::thread_pool<BS::tp::none> &pool)
{
    const HandIndexer &indexer = abstraction.indexer(street);
    const HandIndexer &river = abstraction.indexer(2);
    const auto cdfOf = [&](std::uint64_t idx)
    {
        const auto rounds = indexer.unindex(idx);
        return runoutCdf(rounds[0], rounds[1], river, equity);
    };
    // ponytail: uniform sample over isomorphism classes, not weighted by class multiplicity.
    std::vector<std::uint64_t> sampleIndices(sampleSize);
    for (auto &idx : sampleIndices)
    {
        idx = rng() % indexer.size();
    }
    std::vector<Cdf> sample(sampleSize);
    pool.detach_blocks<std::size_t>(0, sampleSize, [&](std::size_t begin, std::size_t end)
                                    {
        for (std::size_t i = begin; i < end; ++i)
        {
            sample[i] = cdfOf(sampleIndices[i]);
        } });
    pool.wait();
    const std::vector<Cdf> centroids = kmeans(sample, k, 30, rng, pool);
    auto &table = abstraction.buckets[street];
    table.resize(indexer.size());
    pool.detach_blocks<std::uint64_t>(0, indexer.size(), [&](std::uint64_t begin, std::uint64_t end)
                                      {
        for (std::uint64_t i = begin; i < end; ++i)
        {
            table[i] = static_cast<std::uint8_t>(nearest(cdfOf(i), centroids));
        } }, 1024);
    pool.wait();
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <output path> [buckets per street (<= 256)] [k-means sample size]\n";
        return 1;
    }
    const std::size_t k = argc > 2 ? std::stoull(argv[2]) : 200;
    const std::size_t sampleSize = argc > 3 ? std::stoull(argv[3]) : 200'000;
    if (k == 0 || k > 256)
    {
        std::cerr << "buckets must be in [1, 256]\n";
        return 1;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto log = [&](const char *what)
    {
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cerr << "[" << seconds << " s] " << what << "\n";
    };
    BS::thread_pool<BS::tp::none> pool(std::thread::hardware_concurrency());
    omp::XoroShiro128Plus rng{2026};
    CardAbstraction abstraction;

    const HandIndexer &river = abstraction.indexer(2);
    std::vector<std::uint16_t> equity(river.size());
    pool.detach_blocks<std::uint64_t>(0, river.size(), [&](std::uint64_t begin, std::uint64_t end)
                                      {
        for (std::uint64_t i = begin; i < end; ++i)
        {
            const auto rounds = river.unindex(i);
            equity[i] = riverEquity(rounds[0], rounds[1]);
        } }, 1024);
    pool.wait();
    log("river equity");

    // River buckets: equal-population equity quantiles.
    std::vector<std::uint64_t> cumulative(65536);
    for (const std::uint16_t e : equity)
    {
        ++cumulative[e];
    }
    for (std::size_t e = 1; e < cumulative.size(); ++e)
    {
        cumulative[e] += cumulative[e - 1];
    }
    abstraction.buckets[2].resize(river.size());
    for (std::uint64_t i = 0; i < river.size(); ++i)
    {
        abstraction.buckets[2][i] = static_cast<std::uint8_t>((cumulative[equity[i]] - 1) * k / river.size());
    }
    log("river buckets");

    clusterStreet(1, k, sampleSize, abstraction, equity, rng, pool);
    log("turn buckets");
    clusterStreet(0, k, sampleSize, abstraction, equity, rng, pool);
    log("flop buckets");

    if (!abstraction.save(argv[1]))
    {
        std::cerr << "failed to write " << argv[1] << "\n";
        return 1;
    }
    log("saved");
    return 0;
}
