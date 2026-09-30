// Trains the HU NLHE blueprint: multithreaded MCCFR in fixed-time blocks, a Linear CFR discount
// after each block, periodic checkpoints, and resume from an existing blueprint file.
#include "../include/cfr/blueprint.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

static std::uint64_t cards(std::string_view s) { return Deck::parseHand(s).getMask(); }

// Small blind's opening strategy for a few hands; opponent and runout only complete the deal.
static void printOpenings(const Mccfr<Blueprint> &solver)
{
    const auto legal = Blueprint::legalActions(Blueprint::initial());
    std::cerr << "small blind open, columns:";
    for (std::size_t a = 0; a < legal.size; ++a)
    {
        std::cerr << ' ' << (legal.to[a] == Blueprint::fold ? std::string("fold") : std::to_string(legal.to[a]));
    }
    std::cerr << " (chips invested, big blind = 2)\n";
    for (const char *hand : {"As Ah", "Ks Qs", "Ts 9s", "5c 5d", "Kc 7h", "7c 2d"})
    {
        const auto s = Blueprint::deal(Blueprint::initial(), cards(hand), cards("3h 3s"), cards("8c 8d 4h"), cards("Jc"), cards("Qd"));
        const auto sigma = solver.averageStrategy(Blueprint::infosetKey(s), legal.size);
        std::cerr << "  " << hand << ':';
        for (std::size_t a = 0; a < legal.size; ++a)
        {
            std::cerr << ' ' << static_cast<int>(sigma[a] * 100 + 0.5) << '%';
        }
        std::cerr << '\n';
    }
}

int main(int argc, char **argv)
{
    std::map<std::string, double> options{{"--minutes", 60.0},
                                          {"--threads", static_cast<double>(std::thread::hardware_concurrency())},
                                          {"--block-seconds", 60.0},
                                          {"--checkpoint-blocks", 10.0},
                                          {"--prune-threshold", 0.0},
                                          {"--prune-after-blocks", 10.0}};
    bool valid = argc >= 3 && argc % 2 == 1;
    for (int i = 3; valid && i + 1 < argc; i += 2)
    {
        valid = options.contains(argv[i]);
        if (valid)
        {
            options[argv[i]] = std::stod(argv[i + 1]);
        }
    }
    if (!valid)
    {
        std::cerr << "usage: " << argv[0] << " <abstraction.bin> <blueprint.bin>";
        for (const auto &[name, value] : options)
        {
            std::cerr << " [" << name << ' ' << value << ']';
        }
        std::cerr << "\n  --prune-threshold < 0 enables regret-based pruning once the blueprint has --prune-after-blocks blocks\n";
        return 1;
    }
    const std::string blueprintPath = argv[2];
    const double minutes = options["--minutes"];
    const std::size_t threads = static_cast<std::size_t>(options["--threads"]);
    const double blockSeconds = options["--block-seconds"];
    const std::uint64_t checkpointBlocks = static_cast<std::uint64_t>(options["--checkpoint-blocks"]);
    const float pruneThreshold = static_cast<float>(options["--prune-threshold"]);
    const std::uint64_t pruneAfterBlocks = static_cast<std::uint64_t>(options["--prune-after-blocks"]);

    const auto abstraction = std::make_unique<CardAbstraction>();
    if (!abstraction->load(argv[1]))
    {
        std::cerr << "cannot load abstraction " << argv[1] << " (generate it with Poker_Abstraction)\n";
        return 1;
    }
    BlueprintConfig::abstraction = abstraction.get();

    std::array<std::uint64_t, 4> nodes{};
    const std::uint64_t bound = blueprintInfosetBound(*abstraction, nodes);
    auto solver = std::make_unique<Mccfr<Blueprint>>(blueprintCapacity(*abstraction));
    const double gigabytes = static_cast<double>(solver->capacity()) * (8 + 8 * Blueprint::maxActions) / 1e9;
    std::cerr << "public nodes " << nodes[0] << '/' << nodes[1] << '/' << nodes[2] << '/' << nodes[3]
              << ", infoset bound " << bound << ", table " << solver->capacity() << " slots (" << gigabytes << " GB)\n";
    if (std::filesystem::exists(blueprintPath))
    {
        if (!solver->load(blueprintPath))
        {
            std::cerr << "cannot load " << blueprintPath << '\n';
            return 1;
        }
        std::cerr << "resumed: " << solver->iterations() << " iterations, " << solver->numInfosets() << " infosets\n";
    }

    using Clock = std::chrono::steady_clock;
    const auto seconds = [](double s) { return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s)); };
    const Clock::time_point start = Clock::now();
    const Clock::time_point deadline = start + seconds(minutes * 60);
    std::random_device seeds;
    bool pruning = false;
    for (std::uint64_t block = 1; std::chrono::steady_clock::now() < deadline; ++block)
    {
        // >= so a resumed run past the warm-up prunes from its first block.
        if (pruneThreshold < 0.0f && solver->discounts() >= pruneAfterBlocks && !pruning)
        {
            solver->enablePruning(pruneThreshold);
            pruning = true;
            std::cerr << "pruning enabled (threshold " << pruneThreshold << ")\n";
        }
        const std::uint64_t before = solver->iterations();
        const auto blockStart = std::chrono::steady_clock::now();
        {
            std::vector<std::jthread> workers;
            for (std::size_t i = 0; i < threads; ++i)
            {
                workers.emplace_back([&solver, seed = (std::uint64_t{seeds()} << 32) | seeds()](std::stop_token stop)
                                     {
                    CfrRng workerRng{seed};
                    while (!stop.stop_requested())
                    {
                        solver->train(64, workerRng);
                    } });
            }
            std::this_thread::sleep_until(std::min(deadline, blockStart + seconds(blockSeconds)));
        } // jthreads stop and join here
        const double t = static_cast<double>(solver->discounts() + 1);
        solver->discount(static_cast<float>(t / (t + 1)));

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double blockTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - blockStart).count();
        std::cerr << "[" << static_cast<int>(elapsed) << " s] iterations " << solver->iterations()
                  << " (" << static_cast<double>(solver->iterations() - before) / blockTime << "/s), infosets " << solver->numInfosets()
                  << " (" << 100.0 * static_cast<double>(solver->numInfosets()) / static_cast<double>(solver->capacity()) << "% full)\n";

        if (block % checkpointBlocks == 0 || std::chrono::steady_clock::now() >= deadline)
        {
            const std::string tmp = blueprintPath + ".tmp";
            if (!solver->save(tmp))
            {
                std::cerr << "checkpoint failed\n";
                return 1;
            }
            std::filesystem::rename(tmp, blueprintPath);
            std::cerr << "checkpoint saved\n";
        }
    }
    printOpenings(*solver);
    return 0;
}
