#include "../include/deck.hpp"
#include "../include/game.hpp"
#include <random>
#include <thread>
#include <string>
#include <chrono>
#include <atomic>
#include <charconv>

bool playerWinsRandomGame(pcg64 &rng, const Deck playerCards, Deck tableCards, std::size_t numPlayers)
{
    Deck deck = Deck::createFullDeck();
    deck.removeCards(playerCards);
    deck.removeCards(tableCards);
    std::size_t numCardsToDeal = 5 - tableCards.size();
    for (std::size_t i = 0; i < numCardsToDeal; ++i)
    {
        tableCards.addCard(deck.popRandomCard(rng));
    }
    ClassificationResult mainResult = classifyPlayer(playerCards, tableCards);
    Classification mainClassification = mainResult.getClassification();
    switch (mainClassification)
    {
    case Classification::RoyalFlush:
    case Classification::StraightFlush:
        return true;
    default:
        break;
    }
    ClassificationResult boardResult = classifyPlayer(Deck::emptyDeck(), tableCards);
    if (boardResult.getClassification() >= mainClassification && boardResult.getRankFlag() == mainResult.getRankFlag())
    {
        return false;
    }
    for (std::size_t i = 0; i < numPlayers - 1; ++i)
    {
        Card opponentCard1 = deck.popRandomCard(rng);
        Card opponentCard2 = deck.popRandomCard(rng);
        ClassificationResult playerResult = classifyPlayer(Deck::createDeck({opponentCard1, opponentCard2}), tableCards);
        if (playerResult > mainResult)
        {
            return false;
        }
    }
    return true;
}
inline double probabilityOfWinning(pcg64 &rng, const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers)
{
    std::size_t wins = 0;
    for (std::size_t i = 0; i < numSimulations; ++i)
    {
        if (playerWinsRandomGame(rng, playerCards, tableCards, numPlayers))
        {
            ++wins;
        }
    }
    return static_cast<double>(wins) / numSimulations;
}
inline double probabilityOfWinning(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numThreads, std::size_t numPlayers)
{
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    std::atomic<std::size_t> wins = 0;
    std::size_t simulationsPerThread = numSimulations / numThreads;
    for (std::size_t i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([&, i]()
                             {
            pcg64 threadRng(std::random_device{}());
            std::size_t threadWins = 0;
            for (std::size_t j = 0; j < simulationsPerThread; ++j)
            {
                if (playerWinsRandomGame(threadRng, playerCards, tableCards, numPlayers))
                {
                    ++threadWins;
                }
            }
            wins += threadWins; });
    }
    for (auto &thread : threads)
    {
        thread.join();
    }
    return static_cast<double>(wins) / (simulationsPerThread * numThreads);
}
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
    std::optional<Deck> playerCardsOpt = Deck::parseHand(playerHand);
    if (!playerCardsOpt.has_value() || playerCardsOpt->size() != 2)
    {
        std::cerr << "Invalid hand: " << playerHand << '\n';
        return false;
    }
    playerCards = *playerCardsOpt;
    std::string_view tableCardsStr = argv[2];
    std::optional<Deck> tableCardsOpt = Deck::parseHand(tableCardsStr);
    if (!tableCardsOpt.has_value() || tableCardsOpt->size() > 5)
    {
        std::cerr << "Invalid table cards: " << tableCardsStr << '\n';
        return false;
    }
    tableCards = *tableCardsOpt;
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
    if (numSimulations < 1 || numSimulations > 10'000'000)
    {
        std::cerr << "Number of simulations must be between 1 and 10,000,000.\n";
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
    std::size_t threadCount = std::thread::hardware_concurrency();
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Probability of winning: " << probabilityOfWinning(playerDeck, tableDeck, simulations, threadCount, numPlayers) * 100 << "%\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count() << "ms\n";
    return 0;
}