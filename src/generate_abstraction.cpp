// Builds postflop card abstraction tables (see include/cfr/card_abstraction.hpp).
// River: opponent cluster hand strength (OCHS, Johanson et al. 2013), i.e. equity against each of
// 8 groups of opponent starting hands, clustered by k-means. Equity against a uniform random hand
// alone lumps together hands that fare very differently against the strong ranges seen on the river.
// Turn/flop: histogram of river equity over all runouts, clustered by k-means under EMD
// (for 1-D histograms EMD is the L1 distance between CDFs).
#include "../include/cfr/card_abstraction.hpp"
#include "../include/deck.hpp"
#include "../include/game.hpp"
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
constexpr std::size_t opponentClusters = 8;
using Ochs = std::array<float, opponentClusters>;
constexpr std::uint64_t fullDeck = (1ull << 52) - 1;

static std::size_t pairIndex(std::uint64_t pair)
{
    const std::size_t low = static_cast<std::size_t>(std::countr_zero(pair));
    const std::size_t high = static_cast<std::size_t>(63 - std::countl_zero(pair));
    return high * (high - 1) / 2 + low;
}

// Opponent cluster of each two-card hand: preflop classes sorted by equity against a random hand,
// cut into groups of about 1326 / 8 combos.
static std::array<std::uint8_t, 1326> opponentClusterOfHands()
{
    std::array<std::uint64_t, 1326> hands{};
    std::array<std::size_t, 169> combos{};
    for (std::size_t high = 1, i = 0; high < 52; ++high)
    {
        for (std::size_t low = 0; low < high; ++low, ++i)
        {
            hands[i] = (1ull << high) | (1ull << low);
            ++combos[preflopClassIndex(Deck::from_mask(hands[i]))];
        }
    }
    std::array<double, 169> equity{};
    for (std::size_t c = 0; c < 169; ++c)
    {
        const PreflopEntry &e = preflopTable[0][c];
        equity[c] = (e.wins + 0.5 * e.ties) / static_cast<double>(e.wins + e.ties + e.losses);
    }
    std::array<std::size_t, 169> order{};
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b)
              { return equity[a] < equity[b]; });
    std::array<std::uint8_t, 169> clusterOfClass{};
    std::size_t before = 0;
    for (const std::size_t c : order)
    {
        clusterOfClass[c] = static_cast<std::uint8_t>(std::min(opponentClusters - 1, before * opponentClusters / 1326));
        before += combos[c];
    }
    std::array<std::uint8_t, 1326> out{};
    for (std::size_t i = 0; i < 1326; ++i)
    {
        out[i] = clusterOfClass[preflopClassIndex(Deck::from_mask(hands[i]))];
    }
    return out;
}

// Equity against a uniform random hand (for turn/flop histograms) and against each opponent cluster.
struct RiverStrength
{
    std::uint16_t equity;
    std::array<std::uint8_t, opponentClusters> ochs;
};

static RiverStrength riverStrength(std::uint64_t hole, std::uint64_t board, const std::array<std::uint8_t, 1326> &clusterOf)
{
    const ClassificationResult mine = Hand::classify(Deck::from_mask(hole | board));
    std::array<std::uint32_t, opponentClusters> points{}; // 2 per win, 1 per tie
    std::array<std::uint32_t, opponentClusters> total{};
    for (std::uint64_t a = fullDeck & ~(hole | board); a != 0; a &= a - 1)
    {
        const std::uint64_t first = lowestBit(a);
        for (std::uint64_t b = a & (a - 1); b != 0; b &= b - 1)
        {
            const std::uint64_t opponent = first | (lowestBit(b));
            const ClassificationResult theirs = Hand::classify(Deck::from_mask(opponent | board));
            const std::size_t cluster = clusterOf[pairIndex(opponent)];
            points[cluster] += mine > theirs ? 2u : (mine == theirs ? 1u : 0u);
            total[cluster] += 2;
        }
    }
    RiverStrength out{};
    const std::uint32_t allPoints = std::accumulate(points.begin(), points.end(), 0u);
    const std::uint32_t allTotal = std::accumulate(total.begin(), total.end(), 0u);
    out.equity = static_cast<std::uint16_t>(std::uint64_t{allPoints} * 65535 / allTotal);
    for (std::size_t c = 0; c < opponentClusters; ++c)
    {
        out.ochs[c] = static_cast<std::uint8_t>(total[c] == 0 ? 128 : std::uint64_t{points[c]} * 255 / total[c]);
    }
    return out;
}

template <std::size_t D>
static float l1(const std::array<float, D> &a, const std::array<float, D> &b)
{
    float d = 0.0f;
    for (std::size_t i = 0; i < D; ++i)
    {
        d += std::abs(a[i] - b[i]);
    }
    return d;
}

template <std::size_t D>
static std::size_t nearest(const std::array<float, D> &point, const std::vector<std::array<float, D>> &centroids)
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

// L1 distance; centroids sorted by ascending `strength`, so bucket ids go from weakest to strongest.
// ponytail: mean update minimizes L2, not L1 (k-medians would be exact); fine for bucketing.
template <std::size_t D, typename Strength>
static std::vector<std::array<float, D>> kmeans(const std::vector<std::array<float, D>> &points, std::size_t k, std::size_t iterations,
                                                omp::XoroShiro128Plus &rng, BS::thread_pool<BS::tp::none> &pool, Strength strength)
{
    using Point = std::array<float, D>;
    // k-means++ seeding.
    std::vector<Point> centroids{points[rng() % points.size()]};
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
        std::vector<std::array<double, D>> sums(k);
        std::vector<std::size_t> counts(k);
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            ++counts[assignment[i]];
            for (std::size_t b = 0; b < D; ++b)
            {
                sums[assignment[i]][b] += points[i][b];
            }
        }
        for (std::size_t c = 0; c < k; ++c)
        {
            for (std::size_t b = 0; b < D && counts[c] > 0; ++b)
            {
                centroids[c][b] = static_cast<float>(sums[c][b] / static_cast<double>(counts[c]));
            }
        }
    }
    std::sort(centroids.begin(), centroids.end(), [&](const Point &a, const Point &b)
              { return strength(a) < strength(b); });
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
        const std::uint64_t first = lowestBit(a);
        if (std::popcount(board) == 4)
        {
            add(board | first);
            continue;
        }
        for (std::uint64_t b = a & (a - 1); b != 0; b &= b - 1)
        {
            add(board | first | (lowestBit(b)));
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
    // A larger CDF mass means lower equity.
    const std::vector<Cdf> centroids = kmeans(sample, k, 30, rng, pool, [](const Cdf &c)
                                              { return -std::accumulate(c.begin(), c.end(), 0.0); });
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
    const std::array<std::uint8_t, 1326> clusterOf = opponentClusterOfHands();
    std::vector<std::uint16_t> equity(river.size());
    std::vector<std::array<std::uint8_t, opponentClusters>> ochs(river.size());
    pool.detach_blocks<std::uint64_t>(0, river.size(), [&](std::uint64_t begin, std::uint64_t end)
                                      {
        for (std::uint64_t i = begin; i < end; ++i)
        {
            const auto rounds = river.unindex(i);
            const RiverStrength strength = riverStrength(rounds[0], rounds[1], clusterOf);
            equity[i] = strength.equity;
            ochs[i] = strength.ochs;
        } }, 1024);
    pool.wait();
    log("river equity and OCHS");

    const auto ochsOf = [&](std::uint64_t idx)
    {
        Ochs point{};
        for (std::size_t c = 0; c < opponentClusters; ++c)
        {
            point[c] = ochs[idx][c] / 255.0f;
        }
        return point;
    };
    std::vector<Ochs> riverSample(sampleSize);
    for (auto &point : riverSample)
    {
        point = ochsOf(rng() % river.size());
    }
    const std::vector<Ochs> riverCentroids = kmeans(riverSample, k, 30, rng, pool, [](const Ochs &c)
                                                    { return std::accumulate(c.begin(), c.end(), 0.0); });
    abstraction.buckets[2].resize(river.size());
    pool.detach_blocks<std::uint64_t>(0, river.size(), [&](std::uint64_t begin, std::uint64_t end)
                                      {
        for (std::uint64_t i = begin; i < end; ++i)
        {
            abstraction.buckets[2][i] = static_cast<std::uint8_t>(nearest(ochsOf(i), riverCentroids));
        } }, 1024);
    pool.wait();
    ochs = {};
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
