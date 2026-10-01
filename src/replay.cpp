// Goes over hands the Slumbot client played, from its results CSV: at each of our decisions, prints
// the options our strategy had with their probabilities and marks the one the hand took.
// The resolves are rebuilt from the hand's actions, so they match the ones played. Preflop, a Slumbot
// raise between two blueprint sizes is translated by a draw from the hand's seed: without the seed
// column (older files) the replay may translate it the other way and see slightly different ranges.
#include "../include/cfr/blueprint.hpp"
#include "../include/cfr/gpu_subgame_solver.hpp"
#include "../include/cfr/slumbot.hpp"
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Bot = SlumbotBot<Blueprint200Config, GpuSubgameSolver>;

// "AsKd7c" as one mask per card, in order.
static std::vector<std::uint64_t> cards(const std::string &text)
{
    std::vector<std::uint64_t> out;
    for (std::size_t i = 0; i + 1 < text.size(); i += 2)
    {
        out.push_back(Deck::parseHand(text.substr(i, 2)).getMask());
    }
    return out;
}

static std::string text(const SlumbotAction &action) { return action.type == 'b' ? "b" + std::to_string(action.size) : std::string(1, action.type); }

static bool replay(Bot &bot, const std::string &line, std::size_t number)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    for (std::string field; std::getline(stream, field, ',');)
    {
        fields.push_back(field);
    }
    const auto actions = fields.size() >= 6 ? parseSlumbotActions(fields[1]) : std::nullopt;
    if (!actions.has_value())
    {
        std::cerr << "line " << number << ": not a hand\n";
        return false;
    }
    const std::vector<std::uint64_t> hole = cards(fields[3]), board = cards(fields[4]);
    const bool seeded = fields.size() >= 8 && !fields[7].empty();
    std::cout << std::format("line {}: {} holding {} against {}, board {}, {} chips{}\n  {}\n", number, fields[0] == "0" ? "big blind" : "small blind",
                             fields[3], fields[5].empty() ? "??" : fields[5], fields[4].empty() ? "-" : fields[4], fields[2],
                             seeded ? "" : " (no seed: preflop translations may differ)", fields[1]);
    bot.newHand(fields[0] == "0" ? 1 : 0, hole[0] | hole[1], seeded ? std::optional(std::stoull(fields[7])) : std::nullopt);
    static constexpr std::array<std::size_t, 4> shown{0, 3, 4, 5};
    std::string seen; // the actions before the one at hand
    for (std::size_t k = 0; k < actions->size(); ++k)
    {
        const SlumbotAction &action = (*actions)[k];
        const std::string taken = text(action);
        seen += std::string(k > 0 ? action.street - (*actions)[k - 1].street : 0, '/');
        const std::vector<SlumbotAction> prefix(actions->begin(), actions->begin() + static_cast<std::ptrdiff_t>(k));
        const std::size_t dealt = std::min(shown[action.street], board.size());
        const auto decision = bot.peek(prefix, {board.begin(), board.begin() + static_cast<std::ptrdiff_t>(dealt)});
        const std::string before = seen.empty() ? "-" : seen;
        seen += taken;
        if (!decision.has_value())
        {
            continue;
        }
        std::cout << std::format("  {:7s} after {:<40s}", std::array{"preflop", "flop", "turn", "river"}[action.street], before);
        bool found = false;
        for (std::size_t a = 0; a < decision->options.size(); ++a)
        {
            const auto &option = decision->options[a];
            const bool match = !found && option.incr == taken;
            std::cout << std::format(" {}{} {:.1f}%", match ? "*" : "", option.incr, 100.0 * option.probability);
            if (match)
            {
                bot.choose(a);
                found = true;
            }
        }
        std::cout << (found ? "\n" : std::format("   (played {}: not among them)\n", taken));
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "usage: " << argv[0] << " <abstraction.bin> <blueprint200.bin> <results.csv> <line> [<line>...]\n"
                  << "  replays those hands (lines count from 1) of a Poker_Slumbot results file with the client's default\n"
                     "  resolves, printing our options at each decision and starring the one played\n";
        return 1;
    }
    const auto abstraction = std::make_unique<CardAbstraction>();
    if (!abstraction->load(argv[1]))
    {
        std::cerr << "cannot load abstraction " << argv[1] << '\n';
        return 1;
    }
    BlueprintConfig::abstraction = abstraction.get();
    std::unique_ptr<PreflopStrategy<Blueprint200>> preflop;
    {
        const auto blueprint = std::make_unique<Mccfr<Blueprint200>>(blueprintCapacity<Blueprint200>(*abstraction));
        if (!blueprint->load(argv[2]))
        {
            std::cerr << "cannot load blueprint " << argv[2] << '\n';
            return 1;
        }
        preflop = std::make_unique<PreflopStrategy<Blueprint200>>(*blueprint);
    }
    if (Gpu::instance() == nullptr)
    {
        std::cerr << "no OpenCL GPU\n";
        return 1;
    }
    std::vector<std::string> lines;
    std::ifstream in(argv[3]);
    for (std::string line; std::getline(in, line);)
    {
        lines.push_back(line);
    }
    Bot bot(*preflop, SlumbotResolves{}, 1);
    bool ok = true;
    for (int i = 4; i < argc; ++i)
    {
        const std::size_t number = std::stoull(argv[i]);
        if (number == 0 || number > lines.size())
        {
            std::cerr << "no line " << number << " in " << argv[3] << '\n';
            ok = false;
            continue;
        }
        ok = replay(bot, lines[number - 1], number) && ok;
    }
    return ok ? 0 : 1;
}
