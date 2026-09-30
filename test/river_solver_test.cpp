#include "../include/cfr/river_solver.hpp"
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
using Hands = RiverSolver<Hunl100bbConfig>::Hands;

TEST(RiverSolver, ExploitabilityShrinksToNearZero)
{
    Hands uniform{};
    uniform.fill(1.0);
    RiverSolver<Hunl100bbConfig> solver(riverRoot<Nl>("2c 7d 9h Js Ks", 20), {uniform, uniform});
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

TEST(RiverSolver, SolvesClairvoyanceGame)
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
    RiverSolver<Clairvoyance>::Hands bettor{}, catcher{};
    for (std::size_t h = 0; h < holeCombos; ++h)
    {
        bettor[h] = isHand(h, 10, 8) || isHand(h, 2, 1) ? 1.0 : 0.0;
        catcher[h] = isHand(h, 6, 6) ? 1.0 : 0.0;
    }
    const auto root = riverRoot<Cg>("2c 7d 9h Js Ks", 20);
    RiverSolver<Clairvoyance> solver(root, {catcher, bettor});
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
