#ifndef __POKER_CFR_TOOLS_HPP__
#define __POKER_CFR_TOOLS_HPP__
#include "blueprint.hpp"
#include "slumbot.hpp"
#include <format>
#include <iostream>
#include <map>
#include <memory>
#include <string>

// What the command-line tools share: `--name value` options and loading the data files.

// Reads `--name value` pairs from argv[first] on into `options`, which lists the names allowed with
// their defaults. False on an unknown name or a name without a value.
inline bool parseOptions(int argc, char **argv, int first, std::map<std::string, double> &options)
{
    bool valid = argc >= first && (argc - first) % 2 == 0;
    for (int i = first; valid && i + 1 < argc; i += 2)
    {
        valid = options.contains(argv[i]);
        if (valid)
        {
            options[argv[i]] = std::stod(argv[i + 1]);
        }
    }
    return valid;
}

// " [--name default]" for every option, for a usage line.
inline std::string optionsUsage(const std::map<std::string, double> &options)
{
    std::string out;
    for (const auto &[name, value] : options)
    {
        out += " [" + name + ' ' + std::format("{}", value) + ']';
    }
    return out;
}

// The card abstraction the blueprint games bucket with, installed for them; nullptr (and a message) if
// the file can't be read.
inline std::unique_ptr<CardAbstraction> loadAbstraction(const std::string &path)
{
    auto abstraction = std::make_unique<CardAbstraction>();
    if (!abstraction->load(path))
    {
        std::cerr << "cannot load abstraction " << path << " (Poker_Abstraction generates it)\n";
        return nullptr;
    }
    BlueprintConfig::abstraction = abstraction.get();
    return abstraction;
}

// The preflop strategy of a 200bb blueprint file, which is all the bot reads from it: the rest of the
// table is freed on return. nullptr (and a message) if the file can't be read.
inline std::unique_ptr<PreflopStrategy<Blueprint200>> loadPreflop(const CardAbstraction &abstraction, const std::string &path)
{
    const auto blueprint = std::make_unique<Mccfr<Blueprint200>>(blueprintCapacity<Blueprint200>(abstraction));
    if (!blueprint->load(path))
    {
        std::cerr << "cannot load blueprint " << path << " (Poker_Train --stack-bb 200 trains one)\n";
        return nullptr;
    }
    std::cerr << path << ": " << blueprint->iterations() << " iterations, " << blueprint->discounts() << " blocks\n";
    return std::make_unique<PreflopStrategy<Blueprint200>>(*blueprint);
}
#endif // __POKER_CFR_TOOLS_HPP__
