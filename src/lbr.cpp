// Estimates a blueprint's exploitability with local best response (a lower bound), in mbb/hand.
#include "../include/cfr/blueprint.hpp"
#include "../include/cfr/lbr.hpp"
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
                                          {"--streets", 14.0}}; // LBR's call-down assumption is poor preflop
    bool valid = argc >= 4 && argc % 2 == 0;
    for (int i = 4; valid && i + 1 < argc; i += 2)
    {
        valid = options.contains(argv[i]);
        if (valid)
        {
            options[argv[i]] = std::stod(argv[i + 1]);
        }
    }
    if (!valid)
    {
        std::cerr << "usage: " << argv[0] << " <abstraction.bin> <blueprint.bin> <preflop_equity.bin>";
        for (const auto &[name, value] : options)
        {
            std::cerr << " [" << name << ' ' << value << ']';
        }
        std::cerr << "\n  --hands is per LBR seat; --streets is a bitmask of where LBR deviates"
                     " (1 preflop, 2 flop, 4 turn, 8 river; 0 = self-play)\n";
        return 1;
    }

    const auto abstraction = std::make_unique<CardAbstraction>();
    if (!abstraction->load(argv[1]))
    {
        std::cerr << "cannot load abstraction " << argv[1] << '\n';
        return 1;
    }
    BlueprintConfig::abstraction = abstraction.get();
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

    const auto start = std::chrono::steady_clock::now();
    const LocalBestResponse<BlueprintConfig> lbr(*blueprint, *equity, static_cast<std::size_t>(options["--flop-runouts"]),
                                                  static_cast<unsigned>(options["--streets"]));
    const auto result = lbr.evaluate(static_cast<std::uint64_t>(options["--hands"]), static_cast<std::size_t>(options["--threads"]),
                                     static_cast<std::uint64_t>(options["--seed"]));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "LBR as small blind: " << result.mbbPerHand[0] << " +/- " << 1.96 * result.standardError[0] << " mbb/hand\n"
              << "LBR as big blind:   " << result.mbbPerHand[1] << " +/- " << 1.96 * result.standardError[1] << " mbb/hand\n"
              << "LBR average:        " << result.average() << " +/- " << 1.96 * result.averageError() << " mbb/hand (95% CI, "
              << seconds << " s)\n";
    return 0;
}
