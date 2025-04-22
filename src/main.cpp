#include "../include/deck.hpp"
#include "../include/game.hpp"
#include <random>
#include <thread>
#include <string>
#include <chrono>
#include <atomic>

bool playerWinsRandomGame(pcg64 &rng, const Deck playerCards, const Deck tableCards, std::size_t numPlayers)
{
    Deck deck = Deck::createFullDeck();
    deck.removeCards(playerCards);
    deck.removeCards(tableCards);
    Deck tableDeck = Deck::emptyDeck();
    for (const Card card : tableCards)
    {
        tableDeck.addCard(card);
    }
    for (std::size_t i = 0; i < 5 - tableCards.size(); ++i)
    {
        tableDeck.addCard(deck.popRandomCard(rng));
    }
    ClassificationResult mainResult = classifyPlayer(playerCards, tableDeck);
    if (mainResult.getClassification() == Classification::RoyalFlush)
    {
        return true;
    }
    for (std::size_t i = 0; i < numPlayers - 1; ++i)
    {
        Deck opponentDeck = Deck::emptyDeck();
        for (std::size_t j = 0; j < 2; ++j)
        {
            opponentDeck.addCard(deck.popRandomCard(rng));
        }
        ClassificationResult playerResult = classifyPlayer(opponentDeck, tableDeck);
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

int main(int argc, const char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <hand> <table> <num_players>\n";
        return 1;
    }
    std::string_view playerHand = argv[1];
    std::optional<Deck> playerCards = Deck::parseHand(playerHand);
    if (!playerCards.has_value() || playerCards->size() != 2)
    {
        std::cerr << "Invalid hand: " << playerHand << '\n';
        return 1;
    }
    Deck playerDeck = *playerCards;
    std::string_view tableCardsStr = argv[2];
    std::optional<Deck> tableCards = Deck::parseHand(tableCardsStr);
    if (!tableCards.has_value() || tableCards->size() > 5)
    {
        std::cerr << "Invalid table cards: " << tableCardsStr << '\n';
        return 1;
    }
    Deck tableDeck = *tableCards;
    if (!checkUniqueCards(playerDeck, tableDeck))
    {
        std::cerr << "Duplicate cards found in hand or table.\n";
        return 1;
    }
    std::cout << "Player cards: " << *playerDeck.at(0) << ", " << *playerDeck.at(1) << '\n';
    std::size_t threadCount = std::thread::hardware_concurrency();
    std::string numPlayersStr = argv[3];
    std::size_t numPlayers = std::stoul(numPlayersStr);
    if (numPlayers < 2 || numPlayers > 10)
    {
        std::cerr << "Number of players must be between 2 and 10.\n";
        return 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    static constexpr std::size_t simulations = 1'000'000;
    std::cout << "Probability of winning: " << probabilityOfWinning(playerDeck, tableDeck, simulations, threadCount, numPlayers) * 100 << "%\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count() << "ms\n";
    return 0;
}