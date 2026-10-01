// Benchmark of the GPU subgame solver on the trees the Slumbot bot resolves: for each workload, the
// tree's size, the time to build it, to upload it and to solve it, and where the solve's time goes by
// kernel. Also reports what the OpenCL device supports. Ranges are pseudo-random, so every hand stays in.
#include "../include/cfr/blueprint.hpp"
#include "../include/cfr/gpu_subgame_solver.hpp"
#include "../include/cfr/slumbot.hpp"
#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using Flop = CoarseTurn<PostflopRaises<Blueprint200Config>>;
using Turn = CoarseRiver<PostflopRaises<Blueprint200Config>>;

static double seconds(std::chrono::steady_clock::time_point since)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

static double median(std::vector<double> values)
{
    std::ranges::sort(values);
    return values[values.size() / 2];
}

static void describeDevice()
{
    const Gpu &gpu = *Gpu::instance();
    std::cout << gpu.device.getInfo<CL_DEVICE_NAME>() << ", " << gpu.device.getInfo<CL_DEVICE_VERSION>() << ", "
              << gpu.device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << ", driver " << gpu.device.getInfo<CL_DRIVER_VERSION>() << '\n';
    std::cout << "extensions: " << gpu.device.getInfo<CL_DEVICE_EXTENSIONS>() << '\n';
    try
    {
        std::cout << "OpenCL C features:";
        for (const cl_name_version &feature : gpu.device.getInfo<CL_DEVICE_OPENCL_C_FEATURES>())
        {
            std::cout << ' ' << feature.name;
        }
        std::cout << '\n';
    }
    catch (const cl::Error &)
    {
        std::cout << " (not reported)\n";
    }
    for (const char *standard : {"-cl-std=CL2.0", "-cl-std=CL3.0"})
    {
        cl::Program program(gpu.context, subgameKernels);
        try
        {
            program.build({gpu.device}, standard);
            std::cout << "kernels build with " << standard << '\n';
        }
        catch (const cl::Error &)
        {
            std::cout << "kernels do NOT build with " << standard << '\n';
        }
    }
}

// A street's first decision with `invested` chips in from each player.
template <typename C>
static typename Hunl<C>::State root(std::string_view board, std::uint8_t street, std::uint32_t invested)
{
    auto s = Hunl<C>::initial();
    s.dealt = true;
    s.board = Deck::parseHand(board).getMask();
    s.street = street;
    s.invested = {invested, invested};
    s.toAct = 1;
    s.history = 0x51;
    return s;
}

template <typename C>
static void run(const std::string &name, const typename Hunl<C>::State &state, const SubgameOptions &options, std::size_t iterations,
                std::size_t repeats)
{
    using Tree = SubgameTree<C>;
    std::array<typename Tree::Hands, 2> ranges{};
    CfrRng rng{7};
    for (auto &range : ranges)
    {
        for (auto &r : range)
        {
            r = 0.05 + static_cast<double>(rng() % 1000) / 1000.0;
        }
    }
    std::vector<double> build, setup, solve;
    GpuProfile profile;
    double profiledSolve = 0.0;
    for (std::size_t repeat = 0; repeat <= repeats; ++repeat) // the first one warms up
    {
        auto start = std::chrono::steady_clock::now();
        {
            const Tree tree(state, ranges, options);
            build.push_back(seconds(start));
        }
        start = std::chrono::steady_clock::now();
        GpuSubgameSolver<C> solver(state, ranges, options);
        setup.push_back(seconds(start));
        start = std::chrono::steady_clock::now();
        solver.solve(iterations);
        solve.push_back(seconds(start));
        if (repeat == 0)
        {
            build.clear();
            setup.clear();
            solve.clear();
            const Tree &tree = solver.tree();
            std::size_t folds = 0, showdowns = 0, chances = 0, allIns = 0;
            for (const auto &node : tree.nodes)
            {
                folds += node.kind == Tree::Kind::fold;
                showdowns += node.kind == Tree::Kind::showdown;
                chances += node.kind == Tree::Kind::chance;
                allIns += node.kind == Tree::Kind::allIn;
            }
            std::cout << std::format("{}: {} nodes ({} decisions, {} chance, {} folds, {} showdowns, {} all-ins), depth {}, {} hands, {} spaces, "
                                     "{:.1f}M regrets, {:.1f}M averages\n",
                                     name, tree.nodes.size(), tree.nodes.size() - folds - showdowns - chances - allIns, chances, folds, showdowns,
                                     allIns, tree.depth, tree.hands, tree.spaces.size(), static_cast<double>(tree.regrets) / 1e6,
                                     static_cast<double>(tree.averages) / 1e6);
            GpuSubgameSolver<C> profiled(state, ranges, options, true);
            start = std::chrono::steady_clock::now();
            profiled.solve(iterations);
            profiledSolve = seconds(start);
            profile = profiled.profile();
        }
    }
    const double tree = median(build), upload = median(setup) - tree, solved = median(solve);
    std::cout << std::format("  tree {:.0f} ms, upload {:.0f} ms, solve {:.0f} ms ({} iterations, {:.2f} ms each), total {:.0f} ms\n", 1e3 * tree,
                             1e3 * upload, 1e3 * solved, iterations, 1e3 * solved / static_cast<double>(iterations),
                             1e3 * (tree + upload + solved));
    const double kernels = profile.forward + profile.folds + profile.showdowns + profile.allIns + profile.backward;
    std::cout << std::format("  profiled solve {:.0f} ms: forward {:.0f}, folds {:.0f}, showdowns {:.0f}, all-ins {:.0f}, backward {:.0f}, outside kernels {:.0f}; "
                             "{} launches ({:.0f} per iteration)\n",
                             1e3 * profiledSolve, 1e3 * profile.forward, 1e3 * profile.folds, 1e3 * profile.showdowns, 1e3 * profile.allIns, 1e3 * profile.backward,
                             1e3 * (profiledSolve - kernels), profile.launches,
                             static_cast<double>(profile.launches) / static_cast<double>(iterations));
}

int main(int argc, char **argv)
{
    std::map<std::string, double> options{{"--iterations", 60.0}, {"--repeats", 5.0}, {"--device", 0.0}};
    bool valid = argc % 2 == 1;
    for (int i = 1; valid && i + 1 < argc; i += 2)
    {
        valid = options.contains(argv[i]);
        if (valid)
        {
            options[argv[i]] = std::stod(argv[i + 1]);
        }
    }
    if (!valid)
    {
        std::cerr << "usage: " << argv[0] << " [--iterations 60] [--repeats 5] [--device 0|1 (1 describes the OpenCL device)]\n";
        return 1;
    }
    if (Gpu::instance() == nullptr)
    {
        std::cerr << "no OpenCL GPU\n";
        return 1;
    }
    if (options["--device"] != 0.0)
    {
        describeDevice();
    }
    const auto iterations = static_cast<std::size_t>(options["--iterations"]), repeats = static_cast<std::size_t>(options["--repeats"]);
    // As SlumbotBot::resolve builds them.
    const SubgameOptions flop{.averageLaterStreets = false, .minReach = 1e-3, .chanceSamples = 6, .allInSamples = 16, .exactFlopAllIns = true};
    const SubgameOptions turn{.averageLaterStreets = false, .minReach = 1e-3};
    run<Flop>("flop, limped pot", root<Flop>("Ks 8d 3c", 1, 2), flop, iterations, repeats);
    run<Flop>("flop, raised pot", root<Flop>("Ks 8d 3c", 1, 6), flop, iterations, repeats);
    run<Flop>("flop, 3-bet pot", root<Flop>("Ks 8d 3c", 1, 20), flop, iterations, repeats);
    run<Turn>("turn, raised pot", root<Turn>("Ks 8d 3c 5h", 2, 14), turn, iterations, repeats);
    run<Turn>("turn, 3-bet pot", root<Turn>("Ks 8d 3c 5h", 2, 50), turn, iterations, repeats);
    return 0;
}
