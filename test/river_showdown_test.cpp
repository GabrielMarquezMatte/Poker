#include "../include/cfr/river_solver.hpp"
#include <gtest/gtest.h>

// Terminal values against a plain O(hands^2) evaluation, with random reach and card removal.
TEST(RiverSolver, TerminalValuesMatchBruteForce)
{
    using G = Hunl<Hunl100bbConfig>;
    CfrRng rng{11};
    for (const char *boardText : {"2c 7d 9h Js Ks", "Ah Kh Qh Jh Th", "5s 5d 5c 8h 8d", "2c 3c 4d 5h 9s"})
    {
        auto root = G::initial();
        root.dealt = true;
        root.board = Deck::parseHand(boardText).getMask();
        root.street = 3;
        root.invested = {20, 20};
        root.toAct = 1;
        RiverSolver<Hunl100bbConfig>::Hands reach{};
        for (auto &r : reach)
        {
            r = (rng() % 4 == 0) ? 0.0 : static_cast<double>(rng() % 1000) / 1000.0;
        }
        const RiverSolver<Hunl100bbConfig> solver(root, {reach, reach});
        const auto mass = solver.disjointMass(reach);
        const auto showdown = solver.showdown(reach, 3.0f);
        for (std::size_t h = 0; h < holeCombos; ++h)
        {
            if ((holes[h] & root.board) != 0)
            {
                continue;
            }
            const auto mine = Hand::classify(Deck::from_mask(holes[h] | root.board));
            double expectedMass = 0.0, expectedShowdown = 0.0;
            for (std::size_t o = 0; o < holeCombos; ++o)
            {
                if ((holes[o] & (root.board | holes[h])) != 0)
                {
                    continue;
                }
                const auto theirs = Hand::classify(Deck::from_mask(holes[o] | root.board));
                expectedMass += reach[o];
                expectedShowdown += reach[o] * 3.0 * (mine > theirs ? 1.0 : (mine < theirs ? -1.0 : 0.0));
            }
            ASSERT_NEAR(mass[h], expectedMass, 1e-3) << boardText << " hand " << h;
            ASSERT_NEAR(showdown[h], expectedShowdown, 1e-2) << boardText << " hand " << h;
        }
    }
}
