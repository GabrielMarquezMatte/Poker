// How far a 200bb blueprint's preflop strategy moved between two checkpoints: the total variation
// distance between their average strategies at every preflop infoset, weighted by how often the second
// checkpoint reaches it (card removal ignored). Preflop is all the bot reads from the blueprint, so a
// small distance means more training of the same game no longer changes how the bot plays. The coarse
// distance merges the raise sizes: what is left of it is not hands trading one size for another.
#include "../include/cfr/blueprint.hpp"
#include "../include/cfr/slumbot.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using G = Blueprint200;
using Preflop = PreflopStrategy<G>;
using Reach = std::array<std::array<double, 169>, 2>; // [player][hand class], by the second checkpoint

struct Node
{
    std::string path;
    double reach = 0.0;    // probability a hand gets here
    double distance = 0.0; // of the actor's strategies, over the hands they get here with
    double coarse = 0.0;   // the same over fold, call and any raise
    std::string actions;   // each action's frequency here, earlier > later
    std::uint8_t raises = 0;
};

// Share of the 1326 hands in a class of preflopClassIndex: suited above the diagonal, pairs on it.
static double classWeight(std::size_t c)
{
    const std::size_t row = c / 13, column = c % 13;
    return (row == column ? 6.0 : (row > column ? 4.0 : 12.0)) / 1326.0;
}

static void walk(const Preflop &a, const Preflop &b, const G::State &s, const Reach &reach, const std::string &path, std::vector<Node> &nodes)
{
    if (G::isTerminal(s) || s.street != 0)
    {
        return;
    }
    const auto legal = G::legalActions(s);
    const std::size_t p = s.toAct;
    const std::uint32_t facing = std::max(s.invested[0], s.invested[1]);
    const auto kind = [&](std::size_t action) { return legal.to[action] == G::fold ? 0u : (legal.to[action] == facing ? 1u : 2u); };
    double opponent = 0.0, mass = 0.0, moved = 0.0, movedCoarse = 0.0;
    std::array<std::array<double, G::maxActions>, 2> frequency{}; // [earlier, later][action]
    std::array<Reach, G::maxActions> children;
    children.fill(reach);
    for (std::size_t c = 0; c < 169; ++c)
    {
        opponent += classWeight(c) * reach[1 - p][c];
        const std::uint64_t key = G::infosetKeyWithBucket(s, static_cast<std::uint16_t>(c));
        const auto first = a.averageStrategy(key, legal.size), second = b.averageStrategy(key, legal.size);
        const double weight = classWeight(c) * reach[p][c];
        double distance = 0.0;
        std::array<double, 3> kinds{};
        for (std::size_t action = 0; action < legal.size; ++action)
        {
            distance += 0.5 * std::abs(first[action] - second[action]);
            kinds[kind(action)] += first[action] - second[action];
            frequency[0][action] += weight * first[action];
            frequency[1][action] += weight * second[action];
            children[action][p][c] *= second[action];
        }
        mass += weight;
        moved += weight * distance;
        movedCoarse += weight * 0.5 * (std::abs(kinds[0]) + std::abs(kinds[1]) + std::abs(kinds[2]));
    }
    const auto label = [&](std::size_t action) -> std::string
    {
        return kind(action) == 0 ? "f" : (kind(action) == 1 ? "c" : std::format("r{:g}", legal.to[action] / 2.0)); // raise to, in big blinds
    };
    const double scale = mass > 0.0 ? 1.0 / mass : 0.0;
    std::string actions;
    for (std::size_t action = 0; action < legal.size; ++action)
    {
        actions += std::format("{} {:.0f}>{:.0f}  ", label(action), 100.0 * scale * frequency[0][action], 100.0 * scale * frequency[1][action]);
    }
    nodes.push_back({path.empty() ? "(open)" : path, mass * opponent, moved * scale, movedCoarse * scale, actions, s.raises});
    for (std::size_t action = 0; action < legal.size; ++action)
    {
        walk(a, b, G::apply(s, action), children[action], path.empty() ? label(action) : path + ' ' + label(action), nodes);
    }
}

static std::unique_ptr<Preflop> loadPreflop(const CardAbstraction &abstraction, const std::string &path)
{
    const auto blueprint = std::make_unique<Mccfr<G>>(blueprintCapacity<G>(abstraction));
    if (!blueprint->load(path))
    {
        return nullptr;
    }
    std::cerr << path << ": " << blueprint->iterations() << " iterations, " << blueprint->discounts() << " blocks\n";
    return std::make_unique<Preflop>(*blueprint);
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: " << argv[0] << " <abstraction.bin> <earlier.bin> <later.bin>\n";
        return 1;
    }
    const auto abstraction = std::make_unique<CardAbstraction>();
    if (!abstraction->load(argv[1]))
    {
        std::cerr << "cannot load abstraction " << argv[1] << '\n';
        return 1;
    }
    BlueprintConfig::abstraction = abstraction.get();
    const auto earlier = loadPreflop(*abstraction, argv[2]); // one table in memory at a time
    const auto later = earlier != nullptr ? loadPreflop(*abstraction, argv[3]) : nullptr;
    if (later == nullptr)
    {
        std::cerr << "cannot load a blueprint\n";
        return 1;
    }

    Reach reach;
    reach[0].fill(1.0);
    reach[1].fill(1.0);
    std::vector<Node> nodes;
    walk(*earlier, *later, G::initial(), reach, "", nodes);

    // Per raise count: decisions per hand played there, and their distances.
    std::array<double, 8> decisions{}, moved{}, movedCoarse{};
    for (const Node &node : nodes)
    {
        decisions[node.raises] += node.reach;
        moved[node.raises] += node.reach * node.distance;
        movedCoarse[node.raises] += node.reach * node.coarse;
    }
    double allDecisions = 0.0, allMoved = 0.0, allCoarse = 0.0;
    std::cout << "raises  decisions/hand  distance   coarse\n";
    for (std::size_t r = 0; r < decisions.size(); ++r)
    {
        if (decisions[r] > 0.0)
        {
            std::cout << std::format("{:6}  {:14.5f}  {:7.2f}%  {:6.2f}%\n", r, decisions[r], 100.0 * moved[r] / decisions[r], 100.0 * movedCoarse[r] / decisions[r]);
            allDecisions += decisions[r];
            allMoved += moved[r];
            allCoarse += movedCoarse[r];
        }
    }
    std::cout << std::format("   all  {:14.5f}  {:7.2f}%  {:6.2f}%\n\n", allDecisions, 100.0 * allMoved / allDecisions, 100.0 * allCoarse / allDecisions);

    std::ranges::sort(nodes, std::greater{}, &Node::reach);
    std::cout << "  reach  distance   coarse  node (raises to, in big blinds): action % earlier>later\n";
    for (const Node &node : nodes)
    {
        if (node.reach >= 0.002)
        {
            std::cout << std::format("{:6.2f}%  {:7.2f}%  {:6.2f}%  {}: {}\n", 100.0 * node.reach, 100.0 * node.distance, 100.0 * node.coarse, node.path, node.actions);
        }
    }
    return 0;
}
