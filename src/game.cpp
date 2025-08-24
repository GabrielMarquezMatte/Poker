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
    game.addPlayer(std::make_unique<ConsoleExecutioner>(), &player);
    game.addPlayer(std::make_unique<RandomExecutioner>(randomEngine), &bot);
    game.addCards(tableCards);
    std::cout << "Player cards: " << player.getCards() << ". Table cards: " << tableCards << "\n";
    BS::thread_pool<BS::tp::none> threadPool(std::thread::hardware_concurrency());
    std::cout << "Probability of winning: " << probabilityOfWinning(playerCards, tableCards, 100'000, 2, threadPool) << "\n";
    static constexpr float smallBlind = 0.1f;
    static constexpr float bigBlind = 0.2f;
    while (game.playRound(smallBlind, bigBlind))
        ;
    std::cout << "Game over!\n";
    std::cout << "Player chips: " << player.getChips() << '\n';
    std::cout << "Bot chips: " << bot.getChips() << '\n';
}