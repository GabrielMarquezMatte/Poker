// Estimates a blueprint's exploitability with local best response (a lower bound), in mbb/hand.
#include "../include/cfr/gpu_subgame_solver.hpp"
#include "../include/cfr/lbr.hpp"
#include "../include/cfr/tools.hpp"
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>

int main(int argc, char **argv)
{
    std::map<std::string, double> options{{"--hands", 10'000.0},
                                          {"--threads", static_cast<double>(std::thread::hardware_concurrency())},
                                          {"--flop-runouts", 100.0},
                                          {"--seed", 1.0},
                                          {"--streets", 14.0}, // LBR's call-down assumption is poor preflop
                                          {"--river-iterations", 0.0},
                                          {"--turn-iterations", 0.0},
                                          {"--flop-iterations", 0.0},
                                          {"--flop-samples", 8.0},
                                          {"--gpu", 0.0}};
    if (!parseOptions(argc, argv, 4, options))
    {
        std::cerr << "usage: " << argv[0] << " <abstraction.bin> <blueprint.bin> <preflop_equity.bin>" << optionsUsage(options)
                  << "\n  --hands is per LBR seat; --streets is a bitmask of where LBR deviates"
                     " (1 preflop, 2 flop, 4 turn, 8 river; 0 = self-play)\n"
                     "  --river-iterations > 0 evaluates the blueprint resolving each river with that many DCFR iterations,\n"
                     "  --turn-iterations > 0 each turn (to showdown, with a coarser river),\n"
                     "  --flop-iterations > 0 each flop (coarser turn and river, --flop-samples cards per chance node)\n"
                     "  --gpu 1 resolves flops and turns on the first OpenCL GPU\n";
        return 1;
    }
    const auto abstraction = loadAbstraction(argv[1]);
    if (abstraction == nullptr)
    {
        return 1;
    }
    const auto blueprint = std::make_unique<Mccfr<Blueprint>>(blueprintCapacity(*abstraction));
    if (!blueprint->load(argv[2]))
    {
        std::cerr << "cannot load blueprint " << argv[2] << '\n';
        return 1;
    }
    const auto equity = std::make_unique<PreflopEquity>();
    if (!equity->load(argv[3]))
    {
        std::cerr << "cannot load " << argv[3] << " (Poker_PushFold creates it)\n";
        return 1;
    }
    std::cerr << "blueprint: " << blueprint->iterations() << " iterations, " << blueprint->numInfosets() << " infosets\n";

    const bool gpu = options["--gpu"] != 0.0;
    if (gpu && Gpu::instance() == nullptr)
    {
        std::cerr << "no OpenCL GPU\n";
        return 1;
    }
    const auto run = [&]<template <typename> class Solver>()
    {
        const LbrResolves resolves{static_cast<std::size_t>(options["--flop-iterations"]), static_cast<std::size_t>(options["--turn-iterations"]),
                                   static_cast<std::size_t>(options["--river-iterations"]), static_cast<std::size_t>(options["--flop-samples"])};
        const LocalBestResponse<BlueprintConfig, Solver> lbr(*blueprint, *equity, static_cast<std::size_t>(options["--flop-runouts"]),
                                                             static_cast<unsigned>(options["--streets"]), resolves);
        return lbr.evaluate(static_cast<std::uint64_t>(options["--hands"]), static_cast<std::size_t>(options["--threads"]),
                            static_cast<std::uint64_t>(options["--seed"]));
    };
    const auto start = std::chrono::steady_clock::now();
    const auto result = gpu ? run.template operator()<GpuSubgameSolver>() : run.template operator()<SubgameSolver>();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "LBR as small blind: " << result.mbbPerHand[0] << " +/- " << 1.96 * result.standardError[0] << " mbb/hand\n"
              << "LBR as big blind:   " << result.mbbPerHand[1] << " +/- " << 1.96 * result.standardError[1] << " mbb/hand\n"
              << "LBR average:        " << result.average() << " +/- " << 1.96 * result.averageError() << " mbb/hand (95% CI, "
              << seconds << " s)\n";
    return 0;
}
