#include "../include/deck.hpp"
#include "../include/game.hpp"
#include <random>
#include <thread>
#include <string>
#include <chrono>
#include <atomic>
#include <charconv>
inline constexpr bool checkUniqueCards(const Deck playerCards, const Deck tableCards)
{
    std::int64_t playerMask = playerCards.getMask();
    std::int64_t tableMask = tableCards.getMask();
    return (playerMask & tableMask) == 0;
}

bool getParameters(int argc, const char **argv, Deck &playerCards, Deck &tableCards, std::size_t &numPlayers, std::size_t &numSimulations)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <hand> <table> <num_players> [num_simulations]\n";
        return false;
    }
    std::string_view playerHand = argv[1];
    playerCards = Deck::parseHand(playerHand);
    if (playerCards.size() != 2)
    {
        std::cerr << "Invalid hand: " << playerHand << '\n';
        return false;
    }
    std::string_view tableCardsStr = argv[2];
    tableCards = Deck::parseHand(tableCardsStr);
    if (tableCards.size() > 5)
    {
        std::cerr << "Invalid table cards: " << tableCardsStr << '\n';
        return false;
    }
    if (!checkUniqueCards(playerCards, tableCards))
    {
        std::cerr << "Duplicate cards found in hand or table.\n";
        return false;
    }
    std::string_view numPlayersStr = argv[3];
    auto [ptr, err] = std::from_chars(numPlayersStr.data(), numPlayersStr.data() + numPlayersStr.size(), numPlayers);
    if (err != std::errc())
    {
        std::cerr << "Error parsing number of players: " << numPlayersStr << '\n';
        return false;
    }
    if (numPlayers < 2 || numPlayers > 10)
    {
        std::cerr << "Number of players must be between 2 and 10.\n";
        return false;
    }
    if (argc == 4)
    {
        numSimulations = 1'000'000;
        return true;
    }
    std::string_view numSimulationsStr = argv[4];
    auto [ptr2, err2] = std::from_chars(numSimulationsStr.data(), numSimulationsStr.data() + numSimulationsStr.size(), numSimulations);
    if (err2 != std::errc())
    {
        std::cerr << "Error parsing number of simulations: " << numSimulationsStr << '\n';
        return false;
    }
    if (numSimulations < 1 || numSimulations > 40'000'000)
    {
        std::cerr << "Number of simulations must be between 1 and 40,000,000.\n";
        return false;
    }
    return true;
}

int main(int argc, const char **argv)
{
    Deck playerDeck;
    Deck tableDeck;
    std::size_t numPlayers = 0;
    std::size_t simulations = 1'000'000;
    if (!getParameters(argc, argv, playerDeck, tableDeck, numPlayers, simulations))
    {
        return 1;
    }
    std::uint32_t threadCount = std::thread::hardware_concurrency();
    BS::thread_pool pool(threadCount);
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Probability of winning: " << probabilityOfWinning(playerDeck, tableDeck, simulations, numPlayers, pool) * 100 << "%\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count() << "ms\n";
    return 0;
}