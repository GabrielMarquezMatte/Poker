#include "../include/game/game.hpp"
#include "../include/game.hpp"
#include <iostream>
int main()
{
    omp::XoroShiro128Plus rng(std::random_device{}());
    Blinds blinds{50, 100};
    Game g(blinds);
    for (std::size_t i = 0; i < 3; ++i)
    {
        g.addPlayer(10000);
    }
    g.startNewHand(rng);
    BS::thread_pool<BS::tp::none> threadPool{8};
    while (g.state() != GameState::Finished)
    {
        const Player &p = g.currentPlayer();
        const BetData &bd = g.betData();
        const Deck &board = g.board();
        std::cerr << "Street: " << static_cast<int>(g.state()) << ", Player "
                  << p.id << " to act. Chips: " << p.chips << ", Committed: "
                  << p.committed << ", Pot: " << bd.pot << ", Current Bet: " 
                  << bd.currentBet << ", Min Raise: " << bd.minRaise 
                  << ", Probability of Winning: " << probabilityOfWinning(p.hole, board, 100'000, g.players().size(), threadPool) << '\n';
        ActionStruct a{ActionType::Check, 0};
        if (bd.currentBet > p.committed)
        {
            a = {ActionType::Call, 0};
        }
        if (g.applyAction(rng, a))
        {
            break;
        }
    }
    for (const Player &p : g.players())
    {
        std::cerr << "Player " << p.id << ": Chips: " << p.chips << ", Committed: " << p.committed << ", Deck: " << p.hole << '\n';
    }
    std::cerr << "Board: " << g.board() << '\n';
    return 0;
}