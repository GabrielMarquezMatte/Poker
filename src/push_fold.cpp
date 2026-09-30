// Solves 20bb heads-up push/fold with MCCFR, printing exact exploitability as training goes,
// then the small blind push and big blind call charts.
#include "../include/cfr/push_fold.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>

// 13x13 chart: pairs on the diagonal, suited above it, offsuit below; percent of the time.
static void printChart(const char *title, const std::array<double, 169> &probability)
{
    static constexpr char ranks[] = "23456789TJQKA";
    std::printf("%s\n    ", title);
    for (int c = 12; c >= 0; --c)
    {
        std::printf("%4c", ranks[c]);
    }
    std::printf("\n");
    for (int r = 12; r >= 0; --r)
    {
        std::printf("%4c", ranks[r]);
        for (int c = 12; c >= 0; --c)
        {
            std::printf("%4d", static_cast<int>(probability[static_cast<std::size_t>(r * 13 + c)] * 100 + 0.5));
        }
        std::printf("\n");
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <preflop_equity.bin> [blocks=100] [iterations per block=20000]\n";
        return 1;
    }
    const std::string equityPath = argv[1];
    const std::uint64_t blocks = argc > 2 ? std::stoull(argv[2]) : 100;
    const std::uint64_t perBlock = argc > 3 ? std::stoull(argv[3]) : 20'000;

    PreflopEquity equity;
    if (!equity.load(equityPath))
    {
        std::cerr << "computing exact preflop equities (one-time, a few minutes)...\n";
        const auto start = std::chrono::steady_clock::now();
        BS::thread_pool<BS::tp::none> pool(std::thread::hardware_concurrency());
        equity = PreflopEquity::compute(pool);
        if (!equity.save(equityPath))
        {
            std::cerr << "cannot write " << equityPath << '\n';
            return 1;
        }
        std::cerr << "done in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    }

    Mccfr<PushFold> solver(1024);
    CfrRng rng{2026};
    std::printf("%12s %14s %12s %12s\n", "iterations", "mbb/hand", "SB BR", "BB BR");
    for (std::uint64_t t = 1; t <= blocks; ++t)
    {
        solver.train(perBlock, rng);
        solver.discount(static_cast<float>(t) / static_cast<float>(t + 1));
        if (std::has_single_bit(t) || t == blocks)
        {
            const auto e = pushFoldExploitability(equity, pushFoldStrategy(solver));
            std::printf("%12llu %14.3f %12.4f %12.4f\n", static_cast<unsigned long long>(solver.iterations()), e.mbbPerHand(), e.smallBlindBestResponse, e.bigBlindBestResponse);
        }
    }
    const PushFoldStrategy strategy = pushFoldStrategy(solver);
    printChart("small blind push %", strategy.push);
    printChart("big blind call %", strategy.call);
    return 0;
}
