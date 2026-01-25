#include "../include/game/game.hpp"
#include <numeric>
#include <gtest/gtest.h>
static inline constexpr int sum_chips(std::span<const Player> players)
{
    return std::accumulate(players.begin(), players.end(), 0, [](int sum, const Player &p)
                           { return sum + p.chips; });
}

static inline constexpr std::ptrdiff_t count_alive(std::span<const Player> players)
{
    return std::count_if(players.begin(), players.end(), [](const Player &p)
                         { return p.alive(); });
}

static inline constexpr Game make_game(std::size_t n, int chips_each, const Blinds &b)
{
    Game g(b);
    for (std::size_t i = 0; i < n; ++i)
    {
        g.addPlayer(chips_each);
    }
    return g;
}

template <class TRng>
static inline constexpr void play_all_check_call(Game &g, TRng &rng)
{
    while (g.state() != GameState::Finished)
    {
        const Player &p = g.currentPlayer();
        const BetData &bd = g.betData();
        ActionStruct a{ActionType::Check, 0};
        if (bd.currentBet > p.committed)
        {
            a = {ActionType::Call, 0};
        }
        g.applyAction(rng, a);
    }
}

TEST(GameBasics, PostsBlindsAndSetsInitialBets)
{
    constexpr Blinds blinds{50, 100};
    auto g = [&blinds]()
    {
        omp::XoroShiro128Plus rng{42};
        auto game = make_game(3, 10000, blinds);
        game.startNewHand(rng);
        return game;
    }();

    // After posting blinds, pot == SB + BB
    EXPECT_EQ(g.betData().pot, blinds.smallBlind + blinds.bigBlind);

    // currentBet equals (effective) big blind
    EXPECT_EQ(g.betData().currentBet, blinds.bigBlind);

    // minRaise defaults to big blind at the start of preflop
    EXPECT_EQ(g.betData().minRaise, blinds.bigBlind);

    // There must be at least 2 alive
    EXPECT_GE(count_alive(g.players()), 2);
}