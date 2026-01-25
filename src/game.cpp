#include "../include/game/game.hpp"
#include "../include/game.hpp"
#include <iostream>

inline constexpr bool anyPlayersHaveSomeChips(const std::span<const Player> players) noexcept
{
    return std::count_if(players.begin(), players.end(), [](const Player &p)
                         { return p.chips > 0; }) > 1;
}

int main()
{
    omp::XoroShiro128Plus rng(std::random_device{}());
    Blinds blinds{50, 100};
    Game g(blinds);
    for (std::size_t i = 0; i < 3; ++i)
    {
        g.addPlayer(10000);
    }
    std::size_t counter = 0;
    if (anyPlayersHaveSomeChips(g.players()))
    {
        g.startNewHand(rng);
    }
    while (anyPlayersHaveSomeChips(g.players()))
    {
        while (g.state() != GameState::Finished)
        {
            const Player &p = g.currentPlayer();
            const BetData &bd = g.betData();
            ActionStruct a{ActionType::Check, 0};
            if (bd.currentBet > p.committed)
                a = {ActionType::Call, 0};
            g.applyAction(rng, a);
        }
        ++counter;
        if (!anyPlayersHaveSomeChips(g.players()))
        {
            break;
        }
        g.startNewHand(rng);
    }
    std::cerr << "Total hands played: " << counter << '\n';
    for (const Player &p : g.players())
    {
        std::cerr << "Player " << p.id << " has " << p.chips << " chips left.\n";
    }
    return 0;
}