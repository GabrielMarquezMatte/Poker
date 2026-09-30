#ifndef __POKER_CFR_BLUEPRINT_HPP__
#define __POKER_CFR_BLUEPRINT_HPP__
#include "card_abstraction.hpp"
#include "hunl.hpp"

// 100bb blueprint: preflop raises of 0.5/1/2 pot (4 raises max), postflop 0.33/0.75/1.5 pot (3 max),
// all-in always available. Postflop buckets come from a loaded CardAbstraction.
struct BlueprintConfig : Hunl100bbConfig
{
    static constexpr std::array<std::array<double, 3>, 4> raiseFractions{{{0.5, 1.0, 2.0}, {0.33, 0.75, 1.5}, {0.33, 0.75, 1.5}, {0.33, 0.75, 1.5}}};
    static constexpr std::array<std::uint8_t, 4> maxRaises{4, 3, 3, 3};
    static inline const CardAbstraction *abstraction = nullptr; // set before dealing
    static std::uint16_t postflopBucket(std::uint64_t hole, std::uint64_t board) { return abstraction->bucket(hole, board); }
};
using Blueprint = Hunl<BlueprintConfig>;

// Public decision nodes per street (the betting tree ignores cards).
template <typename G>
void countPublicNodes(const typename G::State &s, std::array<std::uint64_t, 4> &nodes, CfrRng &rng)
{
    if (G::isTerminal(s))
    {
        return;
    }
    if (G::isChance(s))
    {
        countPublicNodes<G>(G::sampleChance(s, rng), nodes, rng);
        return;
    }
    ++nodes[s.street];
    for (std::size_t a = 0; a < G::numActions(s); ++a)
    {
        countPublicNodes<G>(G::apply(s, a), nodes, rng);
    }
}
#endif // __POKER_CFR_BLUEPRINT_HPP__
