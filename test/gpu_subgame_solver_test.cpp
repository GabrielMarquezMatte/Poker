#include "../include/cfr/gpu_subgame_solver.hpp"
#include "../include/cfr/subgame_solver.hpp"
#include <gtest/gtest.h>

namespace
{
// Pot-sized bets only, one raise per street, so the turn tree with every river stays small.
struct GpuSmallTurn : Hunl100bbConfig
{
    static constexpr std::array<std::array<double, 3>, 4> raiseFractions{{{0.5, 1.0, 2.0}, {0.5, 1.0, 2.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}};
    static constexpr std::array<std::uint8_t, 4> maxRaises{3, 3, 1, 1};
};

// A pot-sized bet or all-in on the flop, then checks to showdown.
struct GpuSmallFlop : Hunl100bbConfig
{
    static constexpr std::array<std::array<double, 3>, 4> raiseFractions{{{0.5, 1.0, 2.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}};
    static constexpr std::array<std::uint8_t, 4> maxRaises{3, 1, 0, 0};
};

template <typename G>
typename G::State firstDecision(std::string_view board, std::uint8_t street)
{
    auto s = G::initial();
    s.dealt = true;
    s.board = Deck::parseHand(board).getMask();
    s.street = street;
    s.invested = {20, 20};
    s.toAct = 1;
    s.history = 0x1234;
    return s;
}

// Largest difference between the two solvers' average strategies over every node and hand.
template <typename C>
double largestDifference(const SubgameSolver<C> &cpu, const GpuSubgameSolver<C> &gpu)
{
    const auto &tree = cpu.tree();
    double largest = 0.0;
    for (const auto &node : tree.nodes)
    {
        if (node.average == SubgameTree<C>::none)
        {
            continue;
        }
        const std::size_t hands = tree.spaces[node.space].size(), n = node.children.size();
        for (std::size_t i = 0; i < hands; ++i)
        {
            double cpuTotal = 0.0, gpuTotal = 0.0;
            for (std::size_t a = 0; a < n; ++a)
            {
                cpuTotal += cpu.averageSums()[node.average + a * hands + i];
                gpuTotal += gpu.averageSums()[node.average + a * hands + i];
            }
            for (std::size_t a = 0; a < n; ++a)
            {
                const double x = cpuTotal > 0.0 ? cpu.averageSums()[node.average + a * hands + i] / cpuTotal : 1.0 / static_cast<double>(n);
                const double y = gpuTotal > 0.0 ? gpu.averageSums()[node.average + a * hands + i] / gpuTotal : 1.0 / static_cast<double>(n);
                largest = std::max(largest, std::abs(x - y));
            }
        }
    }
    return largest;
}

// One iteration matches the CPU up to rounding (regrets are half precision on the GPU). Later, rounding
// differences grow where a hand's reach is near zero (its average strategy there barely matters), so
// the solutions are compared by exploitability instead.
template <typename C>
void expectSameAsCpu(const typename Hunl<C>::State &root, const std::array<typename SubgameTree<C>::Hands, 2> &ranges, std::size_t iterations,
                     const SubgameOptions &options = {})
{
    if (Gpu::instance() == nullptr)
    {
        GTEST_SKIP() << "no OpenCL GPU";
    }
    SubgameSolver<C> cpu(root, ranges, options);
    GpuSubgameSolver<C> gpu(root, ranges, options);
    cpu.solve(1);
    gpu.solve(1);
    EXPECT_LT(largestDifference(cpu, gpu), 1e-3);
    cpu.solve(iterations - 1);
    gpu.solve(iterations - 1);
    const double pot = 2.0 * root.invested[0];
    const double expected = cpu.exploitability(), actual = cpu.exploitability(gpu.averageSums());
    EXPECT_NEAR(actual, expected, 0.001 * pot);
    EXPECT_LT(actual, 0.01 * pot);
}
} // namespace

TEST(GpuSubgameSolver, RiverMatchesCpu)
{
    using Nl = Hunl<Hunl100bbConfig>;
    SubgameTree<Hunl100bbConfig>::Hands uniform{};
    uniform.fill(1.0);
    expectSameAsCpu<Hunl100bbConfig>(firstDecision<Nl>("2c 7d 9h Js Ks", 3), {uniform, uniform}, 300);
}

TEST(GpuSubgameSolver, TurnMatchesCpu)
{
    CfrRng rng{3};
    std::array<SubgameTree<GpuSmallTurn>::Hands, 2> ranges{};
    for (auto &range : ranges)
    {
        for (auto &r : range)
        {
            r = static_cast<double>(rng() % 1000) / 1000.0;
        }
    }
    expectSameAsCpu<GpuSmallTurn>(firstDecision<Hunl<GpuSmallTurn>>("2c 7d 9h Js", 2), ranges, 300);
}

// All-ins called on the flop are terminals valued over every runout; the other lines deal two cards a street.
// A third of the hands are in neither range, so the tree's hands are not the flop's.
TEST(GpuSubgameSolver, FlopAllInsMatchCpu)
{
    using Tree = SubgameTree<GpuSmallFlop>;
    CfrRng rng{3};
    std::array<Tree::Hands, 2> ranges{};
    for (auto &range : ranges)
    {
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            range[h] = h % 3 == 0 ? 0.0 : static_cast<double>(1 + rng() % 1000) / 1000.0;
        }
    }
    const auto root = firstDecision<Hunl<GpuSmallFlop>>("2c 7d 9h", 1);
    const SubgameOptions options{.chanceSamples = 2, .exactFlopAllIns = true};
    const Tree tree(root, ranges, options);
    EXPECT_GT(tree.allIns, 1u);
    EXPECT_LT(tree.hands, 800u);
    expectSameAsCpu<GpuSmallFlop>(root, ranges, 300, options);
}
