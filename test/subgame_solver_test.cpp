#include "../include/cfr/subgame_solver.hpp"
#include <gtest/gtest.h>

// First decision of the river with both players having put `invested` chips in.
template <typename G>
static typename G::State riverRoot(std::string_view board, std::uint32_t invested)
{
    auto s = G::initial();
    s.dealt = true;
    s.board = Deck::parseHand(board).getMask();
    s.street = 3;
    s.invested = {invested, invested};
    s.toAct = 1;
    s.history = 0x1234;
    return s;
}

using Nl = Hunl<Hunl100bbConfig>;
using Hands = SubgameSolver<Hunl100bbConfig>::Hands;

TEST(SubgameSolver, ExploitabilityShrinksToNearZero)
{
    Hands uniform{};
    uniform.fill(1.0);
    SubgameSolver<Hunl100bbConfig> solver(riverRoot<Nl>("2c 7d 9h Js Ks", 20), {uniform, uniform});
    solver.solve(10);
    const double early = solver.exploitability();
    solver.solve(290);
    const double late = solver.exploitability();
    EXPECT_LT(late, early);
    EXPECT_LT(late, 0.01 * 40); // under 1% of the pot
}

// Clairvoyance game: the first player holds the nuts or air (half each), the second a bluff catcher,
// and the only bet is pot-sized. At equilibrium the nuts always bet, air bluffs half the time
// (bluffs are 1/3 of bets) and the bluff catcher calls half the time (pot / (pot + bet)).
struct Clairvoyance : Hunl100bbConfig
{
    static constexpr std::uint32_t stack = 60; // a pot-sized bet on a 40 pot is all-in
    static constexpr std::array<std::array<double, 3>, 4> raiseFractions{{{0.5, 1.0, 2.0}, {0.5, 1.0, 2.0}, {0.5, 1.0, 2.0}, {1.0, 0.0, 0.0}}};
    static constexpr std::array<std::uint8_t, 4> maxRaises{3, 3, 3, 1};
};

TEST(SubgameSolver, SolvesClairvoyanceGame)
{
    using Cg = Hunl<Clairvoyance>;
    const auto rank = [](std::size_t card)
    { return card % 13; }; // 0 = deuce .. 12 = ace
    const auto isHand = [&](std::size_t h, std::size_t r1, std::size_t r2)
    {
        const std::size_t a = rank(static_cast<std::size_t>(std::countr_zero(holes[h])));
        const std::size_t b = rank(static_cast<std::size_t>(63 - std::countl_zero(holes[h])));
        return (a == r1 && b == r2) || (a == r2 && b == r1);
    };
    // Board 2c 7d 9h Js Ks: Q-T makes the nut straight, 4-3 is air, 8-8 catches bluffs.
    SubgameSolver<Clairvoyance>::Hands bettor{}, catcher{};
    for (std::size_t h = 0; h < holeCombos; ++h)
    {
        bettor[h] = isHand(h, 10, 8) || isHand(h, 2, 1) ? 1.0 : 0.0;
        catcher[h] = isHand(h, 6, 6) ? 1.0 : 0.0;
    }
    const auto root = riverRoot<Cg>("2c 7d 9h Js Ks", 20);
    SubgameSolver<Clairvoyance> solver(root, {catcher, bettor});
    solver.solve(2000);
    ASSERT_EQ(Cg::numActions(root), 2u); // check, all-in
    const auto facingBet = Cg::apply(root, 1);
    double nutsBet = 0.0, airBet = 0.0, calls = 0.0;
    double nuts = 0.0, air = 0.0, catchers = 0.0;
    for (std::size_t h = 0; h < holeCombos; ++h)
    {
        if (isHand(h, 10, 8))
        {
            nutsBet += solver.strategy(root, h)[1];
            nuts += 1.0;
        }
        if (isHand(h, 2, 1))
        {
            airBet += solver.strategy(root, h)[1];
            air += 1.0;
        }
        if (catcher[h] > 0.0)
        {
            calls += solver.strategy(facingBet, h)[1];
            catchers += 1.0;
        }
    }
    EXPECT_NEAR(nutsBet / nuts, 1.0, 0.02);
    EXPECT_NEAR(airBet / air, 0.5, 0.03);
    EXPECT_NEAR(calls / catchers, 0.5, 0.03);
    EXPECT_LT(solver.exploitability(), 0.01 * 40);
}

// First decision of the turn with both players having put `invested` chips in.
template <typename G>
static typename G::State turnRoot(std::string_view board, std::uint32_t invested)
{
    auto s = riverRoot<G>(board, invested);
    s.street = 2;
    return s;
}

// No bets on turn or river: every hand's value is its showdown against the opponent's range, averaged
// over the river cards neither hand holds.
struct CheckDown : Hunl100bbConfig
{
    static constexpr std::array<std::uint8_t, 4> maxRaises{3, 3, 0, 0};
};

TEST(SubgameSolver, TurnCheckDownMatchesBruteForce)
{
    using Cd = Hunl<CheckDown>;
    const auto root = turnRoot<Cd>("2c 7d 9h Js", 20);
    CfrRng rng{5};
    Hands reach{};
    for (auto &r : reach)
    {
        r = (rng() % 4 == 0) ? 0.0 : static_cast<double>(rng() % 1000) / 1000.0;
    }
    Hands uniform{};
    uniform.fill(1.0); // every hand of player 0 gets a value
    SubgameSolver<CheckDown> solver(root, {uniform, reach});
    solver.solve(1);
    const auto values = solver.bestResponseValues(0);
    std::vector<std::uint64_t> rivers;
    std::vector<std::array<ClassificationResult, holeCombos>> strength;
    for (std::size_t c = 0; c < 52; ++c)
    {
        const std::uint64_t card = 1ull << c;
        if ((card & root.board) == 0)
        {
            rivers.push_back(card);
            auto &row = strength.emplace_back();
            for (std::size_t h = 0; h < holeCombos; ++h)
            {
                row[h] = Hand::classify(Deck::from_mask(holes[h] | root.board | card));
            }
        }
    }
    for (std::size_t h = 0; h < holeCombos; ++h)
    {
        if ((holes[h] & root.board) != 0)
        {
            continue;
        }
        double expected = 0.0;
        for (std::size_t r = 0; r < rivers.size(); ++r)
        {
            if ((holes[h] & rivers[r]) != 0)
            {
                continue;
            }
            for (std::size_t o = 0; o < holeCombos; ++o)
            {
                if ((holes[o] & (root.board | rivers[r] | holes[h])) == 0)
                {
                    const auto mine = strength[r][h], theirs = strength[r][o];
                    expected += reach[o] * 20.0 * (mine > theirs ? 1.0 : (mine < theirs ? -1.0 : 0.0)) / 44.0;
                }
            }
        }
        ASSERT_NEAR(values[h], expected, 1e-2) << "hand " << h;
    }
}

// Pot-sized bets only, one raise per street, so the turn tree with every river stays small.
struct SmallTurn : Hunl100bbConfig
{
    static constexpr std::array<std::array<double, 3>, 4> raiseFractions{{{0.5, 1.0, 2.0}, {0.5, 1.0, 2.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}};
    static constexpr std::array<std::uint8_t, 4> maxRaises{3, 3, 1, 1};
};

TEST(SubgameSolver, TurnExploitabilityShrinksToNearZero)
{
    Hands uniform{};
    uniform.fill(1.0);
    SubgameSolver<SmallTurn> solver(turnRoot<Hunl<SmallTurn>>("2c 7d 9h Js", 20), {uniform, uniform});
    solver.solve(10);
    const double early = solver.exploitability();
    solver.solve(290);
    const double late = solver.exploitability();
    EXPECT_LT(late, early);
    EXPECT_LT(late, 0.01 * 40); // under 1% of the pot
}

// The lookahead game keeps the turn's betting, so turn histories and action indices match the full game.
TEST(SubgameSolver, CoarseRiverKeepsTurnActions)
{
    using Coarse = Hunl<CoarseRiver<Hunl100bbConfig>>;
    auto s = turnRoot<Nl>("2c 7d 9h Js", 20);
    auto coarse = std::bit_cast<Coarse::State>(s);
    for (const std::size_t a : {2, 1}) // half-pot bet, call
    {
        ASSERT_EQ(Nl::numActions(s), Coarse::numActions(coarse));
        s = Nl::apply(s, a);
        coarse = Coarse::apply(coarse, a);
        ASSERT_EQ(s.history, coarse.history);
    }
    EXPECT_EQ(s.street, 3);
    EXPECT_LT(Coarse::numActions(coarse), Nl::numActions(s)); // the river is coarser
}

// No bets after the flop: sampled turn and river cards give, on average over samples, the values of
// dealing every card.
struct FlopCheckDown : Hunl100bbConfig
{
    static constexpr std::array<std::uint8_t, 4> maxRaises{3, 0, 0, 0};
};

TEST(SubgameSolver, SampledChanceIsUnbiased)
{
    using Cd = Hunl<FlopCheckDown>;
    auto root = turnRoot<Cd>("2c 7d 9h Js", 20);
    root.board = Deck::parseHand("2c 7d 9h").getMask();
    root.street = 1;
    CfrRng rng{9};
    Hands reach{}, uniform{};
    uniform.fill(1.0);
    for (auto &r : reach)
    {
        r = static_cast<double>(rng() % 1000) / 1000.0;
    }
    SubgameSolver<FlopCheckDown> full(root, {uniform, reach});
    const auto exact = full.bestResponseValues(0);
    constexpr std::size_t samples = 40;
    Hands average{};
    for (std::size_t k = 0; k < samples; ++k)
    {
        root.history = 0x1234 + k; // reseeds the sample
        SubgameSolver<FlopCheckDown> sampled(root, {uniform, reach}, {.chanceSamples = 10});
        const auto values = sampled.bestResponseValues(0);
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            average[h] += values[h] / samples;
        }
    }
    double mass = 0.0;
    for (const double r : reach)
    {
        mass += r;
    }
    double error = 0.0;
    for (std::size_t h = 0; h < holeCombos; ++h)
    {
        error += std::abs(average[h] - exact[h]) / holeCombos;
    }
    EXPECT_LT(error, 0.02 * 20.0 * mass); // of the stake against the whole range: 0.8% here, 29% with a naive 1 / (sampled - 4) weight
}

// Runouts after an all-in take their own sample size; streets dealt before more betting take theirs.
TEST(SubgameSolver, AllInRunoutsHaveTheirOwnSample)
{
    using Tree = SubgameTree<Hunl100bbConfig>;
    auto root = turnRoot<Nl>("2c 7d 9h Js", 20);
    root.board = Deck::parseHand("2c 7d 9h").getMask();
    root.street = 1;
    Hands uniform{};
    uniform.fill(1.0);
    const Tree tree(root, {uniform, uniform}, {.averageLaterStreets = false, .chanceSamples = 4, .allInSamples = 10});
    std::size_t allIns = 0, streets = 0;
    for (const auto &node : tree.nodes)
    {
        if (node.kind != Tree::Kind::chance)
        {
            continue;
        }
        const bool allIn = node.state.street == Nl::showdown;
        EXPECT_EQ(node.children.size(), allIn ? 10u : 4u);
        (allIn ? allIns : streets) += 1;
    }
    EXPECT_GT(allIns, 0u);
    EXPECT_GT(streets, 0u);
}

// A root's extra action (a real bet the abstraction lacks) is one more child, solved like the others.
TEST(SubgameSolver, RootExtraActionIsSolved)
{
    Hands uniform{};
    uniform.fill(1.0);
    const auto root = riverRoot<Nl>("2c 7d 9h Js Ks", 20);
    const std::size_t legal = Nl::numActions(root);
    SubgameSolver<Hunl100bbConfig> solver(root, {uniform, uniform}, {.rootExtraAction = 57}); // a bet to 57: no legal size
    const auto &tree = solver.tree();
    ASSERT_EQ(tree.nodes.front().children.size(), legal + 1);
    const auto &extra = tree.nodes[tree.nodes.front().children.back()].state;
    EXPECT_EQ(extra.invested[root.toAct], 57u);
    EXPECT_EQ(extra.history, Nl::applyTo(root, 57, legal).history);
    EXPECT_TRUE(solver.contains(extra)); // the answers to it are in the tree
    solver.solve(200);
    EXPECT_LT(solver.exploitability(), 0.01 * 40);
    double total = 0.0;
    for (const double p : solver.strategy(root, holeIndex(Deck::parseHand("As Ah").getMask())))
    {
        total += p;
    }
    EXPECT_NEAR(total, 1.0, 1e-6);
}

// Every river's hands are the turn's that its card leaves alive, weakest first with ties in the turn's
// order, and a group per strength: what the showdown sums rely on.
TEST(SubgameSolver, RiverSpacesAreSortedWeakestFirst)
{
    Hands reach{};
    CfrRng rng{9};
    for (auto &r : reach)
    {
        r = (rng() % 4 == 0) ? 0.0 : 1.0; // a turn space with gaps
    }
    const SubgameTree<CheckDown> tree(turnRoot<Hunl<CheckDown>>("2c 7d 9h Js", 20), {reach, reach});
    ASSERT_EQ(tree.spaces.size(), 49u); // the turn and its 48 rivers
    const auto &turn = tree.spaces[0];
    for (const auto &node : tree.nodes)
    {
        if (std::popcount(node.state.board) != 5)
        {
            continue;
        }
        std::vector<std::pair<ClassificationResult, std::uint16_t>> expected;
        for (std::size_t j = 0; j < turn.size(); ++j)
        {
            const std::uint64_t hole = (1ull << turn.low[j]) | (1ull << turn.high[j]);
            if ((hole & node.state.board) == 0)
            {
                expected.emplace_back(Hand::classify(Deck::from_mask(hole | node.state.board)), static_cast<std::uint16_t>(j));
            }
        }
        std::sort(expected.begin(), expected.end());
        const auto &space = tree.spaces[node.space];
        ASSERT_EQ(space.size(), expected.size());
        std::vector<std::size_t> groups;
        for (std::size_t k = 0; k < expected.size(); ++k)
        {
            if (k == 0 || expected[k].first != expected[k - 1].first)
            {
                groups.push_back(k);
            }
            ASSERT_EQ(space.parent[k], expected[k].second);
            ASSERT_EQ(space.low[k], turn.low[expected[k].second]);
            ASSERT_EQ(space.high[k], turn.high[expected[k].second]);
        }
        groups.push_back(expected.size());
        ASSERT_EQ(space.groups, groups);
    }
}

// The balance of two hands on a flop: the turn-and-river pairs the first wins minus those it loses.
TEST(SubgameSolver, FlopAllInBalanceMatchesBruteForce)
{
    const std::uint64_t flop = Deck::parseHand("2c 7d 9h").getMask();
    const auto &balance = flopAllInBalance(flop);
    CfrRng rng{5};
    std::size_t decided = 0;
    for (std::size_t k = 0; k < 300; ++k)
    {
        const std::size_t i = rng() % holeCombos, j = rng() % holeCombos;
        const std::uint64_t dead = flop | holes[i] | holes[j];
        int expected = 0;
        for (std::uint64_t turns = ((1ull << 52) - 1) & ~dead; turns != 0 && std::popcount(dead) == 7; turns &= turns - 1)
        {
            for (std::uint64_t rivers = turns & (turns - 1); rivers != 0; rivers &= rivers - 1)
            {
                const std::uint64_t board = flop | (turns & (~turns + 1)) | (rivers & (~rivers + 1));
                const auto mine = Hand::classify(Deck::from_mask(holes[i] | board));
                const auto theirs = Hand::classify(Deck::from_mask(holes[j] | board));
                expected += mine > theirs ? 1 : (mine < theirs ? -1 : 0);
            }
        }
        decided += expected != 0;
        ASSERT_EQ(balance[i * holeCombos + j], expected) << i << " against " << j;
        ASSERT_EQ(balance[j * holeCombos + i], -expected);
    }
    EXPECT_GT(decided, 150u);
}

// A called flop all-in is worth the same as a terminal with the balance as with every turn and river dealt.
TEST(SubgameSolver, ExactFlopAllInsMatchEveryRunout)
{
    using Tree = SubgameTree<Hunl100bbConfig>;
    auto root = turnRoot<Nl>("2c 7d 9h Js", 20);
    root.board = Deck::parseHand("2c 7d 9h").getMask();
    root.street = 1;
    root.invested = {Hunl100bbConfig::stack, 20}; // player 1 faces an all-in: fold or call
    root.actions = 1;
    root.raises = 1;
    CfrRng rng{11};
    std::array<Hands, 2> ranges{};
    double mass = 0.0;
    for (auto &range : ranges)
    {
        for (auto &r : range)
        {
            r = (rng() % 4 == 0) ? 0.0 : static_cast<double>(rng() % 1000) / 1000.0;
            mass += r / 2.0;
        }
    }
    SubgameSolver<Hunl100bbConfig> dealt(root, ranges), exact(root, ranges, {.exactFlopAllIns = true});
    const auto count = [](const Tree &tree, Tree::Kind kind)
    { return std::ranges::count_if(tree.nodes, [kind](const auto &node) { return node.kind == kind; }); };
    EXPECT_EQ(count(dealt.tree(), Tree::Kind::showdown), 49 * 48);
    EXPECT_EQ(count(exact.tree(), Tree::Kind::allIn), 1);
    EXPECT_EQ(count(exact.tree(), Tree::Kind::chance), 0);
    const double tolerance = 1e-4 * Hunl100bbConfig::stack * mass;
    for (std::size_t p = 0; p < 2; ++p)
    {
        const auto expected = dealt.bestResponseValues(p), actual = exact.bestResponseValues(p);
        double largest = 0.0;
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            ASSERT_NEAR(actual[h], expected[h], tolerance) << "player " << p << ", hand " << h;
            largest = std::max(largest, std::abs(expected[h]));
        }
        EXPECT_GT(largest, 100.0 * tolerance); // the values are not all near zero
    }
    dealt.solve(5);
    exact.solve(5);
    EXPECT_NEAR(exact.exploitability(), dealt.exploitability(), 1e-3 * Hunl100bbConfig::stack);
}

// Only a flop root's own all-ins become terminals: a turn root deals its all-ins' rivers as before.
TEST(SubgameSolver, ExactFlopAllInsLeaveOtherStreetsAlone)
{
    using Tree = SubgameTree<Hunl100bbConfig>;
    Hands uniform{};
    uniform.fill(1.0);
    const Tree tree(turnRoot<Nl>("2c 7d 9h Js", 20), {uniform, uniform}, {.averageLaterStreets = false, .exactFlopAllIns = true});
    EXPECT_TRUE(std::ranges::none_of(tree.nodes, [](const auto &node) { return node.kind == Tree::Kind::allIn; }));
    EXPECT_EQ(tree.allIns, 0u);
}
