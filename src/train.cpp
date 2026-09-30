// Trains the HU NLHE blueprint: multithreaded MCCFR in fixed-time blocks, a Linear CFR discount
// after each block, periodic checkpoints, and resume from an existing blueprint file.
#include "../include/cfr/blueprint.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
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
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " <abstraction.bin> <blueprint.bin> [minutes=60] [threads=all] [block seconds=60] [checkpoint every N blocks=10]\n";
        return 1;
    }
    const std::string blueprintPath = argv[2];
    const double minutes = argc > 3 ? std::stod(argv[3]) : 60.0;
    const std::size_t threads = argc > 4 ? std::stoull(argv[4]) : std::thread::hardware_concurrency();
    const double blockSeconds = argc > 5 ? std::stod(argv[5]) : 60.0;
    const std::uint64_t checkpointBlocks = argc > 6 ? std::stoull(argv[6]) : 10;

    const auto abstraction = std::make_unique<CardAbstraction>();
    if (!abstraction->load(argv[1]))
    {
        std::cerr << "cannot load abstraction " << argv[1] << " (generate it with Poker_Abstraction)\n";
        return 1;
    }
    BlueprintConfig::abstraction = abstraction.get();

    // Capacity: every public node times its street's buckets bounds the infoset count.
    std::array<std::uint64_t, 4> nodes{};
    CfrRng rng{1};
    countPublicNodes<Blueprint>(Blueprint::initial(), nodes, rng);
    std::uint64_t bound = nodes[0] * 169;
    for (std::size_t street = 1; street < 4; ++street)
    {
        const auto &table = abstraction->buckets[street - 1];
        bound += nodes[street] * (1 + *std::max_element(table.begin(), table.end()));
    }
    auto solver = std::make_unique<Mccfr<Blueprint>>(bound * 3 / 2);
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
    for (std::uint64_t block = 1; std::chrono::steady_clock::now() < deadline; ++block)
    {
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
