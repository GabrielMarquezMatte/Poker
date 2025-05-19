#include <iostream>
#include "../include/executioner.hpp"

int main()
{
    Deck fullDeck = Deck::createFullDeck();
    omp::XoroShiro128Plus randomEngine{std::random_device{}()};
    Deck playersCards = fullDeck.popRandomCards(randomEngine, 2 * 2);
    Deck playerCards = playersCards.popCards(2);
    Deck botCards = playersCards.popCards(2);
    Deck tableCards = fullDeck.popRandomCards(randomEngine, 5);
    Player player(playerCards, 1000.0f);
    Player bot(botCards, 1000.0f);
    Game game;
    game.addPlayer(player, ConsoleExecutioner());
    game.addPlayer(bot, RandomExecutioner(randomEngine));
    game.addCards(tableCards);
    while(game.playRound());
}