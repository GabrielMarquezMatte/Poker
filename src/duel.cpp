// Duplicate match between two configurations of the Slumbot bot, played locally: every deal is played
// twice with the bots swapping seats and each seat keeping its random draws, so the two hands differ only
// where the bots' strategies do (the luck of the cards and of the draws cancels), and all-ins before the
// river score their equity. Reports bot A's winnings against bot B in mbb/hand; --csv <file> also writes
// every pair's cards, both hands' actions and A's chips in each.
#include "../include/cfr/blueprint.hpp"
#include "../include/cfr/gpu_subgame_solver.hpp"
#include "../include/cfr/slumbot.hpp"
#include "../include/cfr/slumbot_referee.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Bot = SlumbotBot<Blueprint200Config, GpuSubgameSolver>;

static std::unique_ptr<PreflopStrategy<Blueprint200>> loadPreflop(const CardAbstraction &abstraction, const std::string &path)
{
    const auto blueprint = std::make_unique<Mccfr<Blueprint200>>(blueprintCapacity<Blueprint200>(abstraction));
    if (!blueprint->load(path))
    {
        return nullptr;
    }
    std::cerr << path << ": " << blueprint->iterations() << " iterations\n";
    return std::make_unique<PreflopStrategy<Blueprint200>>(*blueprint);
}

// The cards of a mask as Slumbot writes them: "AsKd".
static std::string cards(std::uint64_t mask)
{
    std::string out;
    for (const Card card : Deck::from_mask(mask))
    {
        out += "23456789TJQKA"[getRankIndex(card.getRank())];
        out += "hdcs"[getSuitIndex(card.getSuit())];
    }
    return out;
}

struct Played
{
    double chips, adjusted; // the small blind's, as played and with all-ins before the river at their equity
    std::string action;
};

// One hand: `sb` and `bb` hold hands[1] and hands[0] (by Slumbot's pos: 1 small blind, 0 big blind) and
// draw from seeds[1] and seeds[0]. Returns nothing if a bot misbehaved.
static std::optional<Played> playHand(Bot &sb, Bot &bb, const std::array<std::uint64_t, 2> &hands, const std::vector<std::uint64_t> &order,
                                      const std::array<std::uint64_t, 2> &seeds)
{
    sb.newHand(0, hands[1], seeds[1]);
    bb.newHand(1, hands[0], seeds[0]);
    SlumbotReferee referee;
    while (!referee.done)
    {
        Bot &bot = referee.pos == 1 ? sb : bb;
        const auto actions = parseSlumbotActions(referee.action);
        const std::vector<std::uint64_t> seen(order.begin(), order.begin() + std::array<int, 4>{0, 3, 4, 5}[static_cast<std::size_t>(referee.street)]);
        const auto incr = bot.update(*actions, seen);
        if (!incr.has_value() || !referee.apply(*incr))
        {
            std::cerr << "bad play after " << referee.action << ": " << incr.value_or("(none)") << '\n';
            return std::nullopt;
        }
    }
    const std::uint64_t board = order[0] | order[1] | order[2] | order[3] | order[4];
    return Played{referee.winnings(1, hands, board, order, false), referee.winnings(1, hands, board, order, true), referee.action};
}

int main(int argc, char **argv)
{
    std::map<std::string, double> options{{"--pairs", 500.0}, {"--sessions", 3.0}, {"--seed", 1.0}};
    for (const char *side : {"--a-", "--b-"})
    {
        const SlumbotResolves defaults;
        options[std::string(side) + "flop-iterations"] = static_cast<double>(defaults.flop);
        options[std::string(side) + "turn-iterations"] = static_cast<double>(defaults.turn);
        options[std::string(side) + "river-iterations"] = static_cast<double>(defaults.river);
        options[std::string(side) + "flop-samples"] = static_cast<double>(defaults.flopSamples);
        options[std::string(side) + "all-in-samples"] = static_cast<double>(defaults.allInSamples);
        options[std::string(side) + "nested"] = defaults.nested ? 1.0 : 0.0;
        options[std::string(side) + "jitter"] = defaults.jitter;
    }
    std::string csvPath;
    bool valid = argc >= 4 && argc % 2 == 0;
    for (int i = 4; valid && i + 1 < argc; i += 2)
    {
        if (std::string(argv[i]) == "--csv")
        {
            csvPath = argv[i + 1];
            continue;
        }
        valid = options.contains(argv[i]);
        if (valid)
        {
            options[argv[i]] = std::stod(argv[i + 1]);
        }
    }
    if (!valid)
    {
        std::cerr << "usage: " << argv[0] << " <abstraction.bin> <blueprintA.bin> <blueprintB.bin>";
        for (const auto &[name, value] : options)
        {
            std::cerr << " [" << name << ' ' << value << ']';
        }
        std::cerr << " [--csv pairs.csv]\n  plays --pairs deals twice each (seats swapped) between bots A and B, over --sessions threads;\n"
                     "  the csv gets, per pair, the big blind's and small blind's cards, the board, and for A in the small blind\n"
                     "  and then in the big blind: the action, A's chips and A's chips with all-ins before the river at their equity\n";
        return 1;
    }
    std::ofstream csv;
    if (!csvPath.empty())
    {
        csv.open(csvPath);
        if (!csv)
        {
            std::cerr << "cannot write " << csvPath << '\n';
            return 1;
        }
        csv << "pair,bb_cards,sb_cards,board,a_sb_action,a_sb_chips,a_sb_adjusted,a_bb_action,a_bb_chips,a_bb_adjusted\n";
    }
    const auto abstraction = std::make_unique<CardAbstraction>();
    if (!abstraction->load(argv[1]))
    {
        std::cerr << "cannot load abstraction " << argv[1] << '\n';
        return 1;
    }
    BlueprintConfig::abstraction = abstraction.get();
    const auto preflopA = loadPreflop(*abstraction, argv[2]);
    const auto preflopB = std::string(argv[2]) == argv[3] ? nullptr : loadPreflop(*abstraction, argv[3]);
    if (preflopA == nullptr || (preflopB == nullptr && std::string(argv[2]) != argv[3]))
    {
        std::cerr << "cannot load a blueprint\n";
        return 1;
    }
    if (Gpu::instance() == nullptr)
    {
        std::cerr << "no OpenCL GPU\n";
        return 1;
    }
    const auto resolves = [&](const std::string &side)
    {
        const auto get = [&](const char *name) { return static_cast<std::size_t>(options[side + name]); };
        return SlumbotResolves{get("flop-iterations"), get("turn-iterations"), get("river-iterations"), get("flop-samples"), get("all-in-samples"),
                               get("nested") != 0, options[side + "jitter"]};
    };
    const SlumbotResolves resolvesA = resolves("--a-"), resolvesB = resolves("--b-");
    const auto pairs = static_cast<std::uint64_t>(options["--pairs"]);
    const auto seed = static_cast<std::uint64_t>(options["--seed"]);

    std::mutex mutex; // guards the totals and output
    std::atomic<std::uint64_t> next{0};
    std::atomic<bool> failed{false};
    double sum = 0.0, squares = 0.0, adjustedSum = 0.0, adjustedSquares = 0.0;
    std::uint64_t played = 0;
    const auto start = std::chrono::steady_clock::now();
    const auto session = [&](std::uint64_t session)
    {
        Bot a(*preflopA, resolvesA, seed * 7919 + session), b(preflopB != nullptr ? *preflopB : *preflopA, resolvesB, seed * 104729 + session);
        for (std::uint64_t pair = next++; pair < pairs && !failed; pair = next++)
        {
            CfrRng rng{seed * 1000003 + pair};
            Deck deck = Deck::createFullDeck();
            const std::array<std::uint64_t, 2> hands{deck.popPair(rng).getMask(), deck.popPair(rng).getMask()};
            std::vector<std::uint64_t> order;
            for (int i = 0; i < 5; ++i)
            {
                order.push_back(deck.popRandomCards(rng, 1).getMask());
            }
            const std::array<std::uint64_t, 2> seeds{rng(), rng()};
            // A in the small blind with hands[1], then B there with the same cards: A is the big blind with hands[0].
            const auto first = playHand(a, b, hands, order, seeds);
            const auto second = playHand(b, a, hands, order, seeds);
            if (!first.has_value() || !second.has_value())
            {
                failed = true;
                return;
            }
            const double raw = (first->chips - second->chips) / 2.0, adjusted = (first->adjusted - second->adjusted) / 2.0; // per hand
            const std::lock_guard lock(mutex);
            if (csv.is_open())
            {
                csv << pair << ',' << cards(hands[0]) << ',' << cards(hands[1]) << ',';
                for (const std::uint64_t card : order)
                {
                    csv << cards(card);
                }
                csv << ',' << first->action << ',' << first->chips << ',' << first->adjusted << ',' << second->action << ',' << -second->chips << ','
                    << -second->adjusted << std::endl;
            }
            const double mbb = raw / 100.0 * 1000.0, adjustedMbb = adjusted / 100.0 * 1000.0;
            sum += mbb;
            squares += mbb * mbb;
            adjustedSum += adjustedMbb;
            adjustedSquares += adjustedMbb * adjustedMbb;
            ++played;
            if (played % 10 == 0 || played == pairs)
            {
                const double n = static_cast<double>(played);
                const auto line = [&](double s, double q)
                {
                    const double mean = s / n;
                    return std::to_string(mean) + " +/- " + std::to_string(1.96 * std::sqrt((std::max)(q / n - mean * mean, 0.0) / n));
                };
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::cout << played << " pairs: A " << line(sum, squares) << " mbb/hand, all-in adjusted " << line(adjustedSum, adjustedSquares)
                          << " (" << seconds / n << " s/pair)" << std::endl;
            }
        }
    };
    {
        std::vector<std::jthread> threads;
        for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(options["--sessions"]); ++i)
        {
            threads.emplace_back(session, i);
        }
    }
    return failed ? 1 : 0;
}
