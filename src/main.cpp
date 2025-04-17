#include "../include/deck.hpp"
#include "../include/game.hpp"
#include <random>
#include <thread>
#include <string>
#include <chrono>
#include <atomic>

template <std::size_t N>
bool playerWinsRandomGame(pcg64 &rng, const Deck playerCards, const Deck tableCards)
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
        auto card = deck.popRandomCard(rng);
        if (!card.has_value())
        {
            return false;
        }
        tableDeck.addCard(card.value());
    }
    ClassificationResult mainResult = classifyPlayer(playerCards, tableDeck);
    if (mainResult.getClassification() == Classification::RoyalFlush)
    {
        return true;
    }
    for (std::size_t i = 0; i < N - 1; ++i)
    {
        Deck oponentDeck = Deck::emptyDeck();
        for (std::size_t j = 0; j < 2; ++j)
        {
            auto card = deck.popRandomCard(rng);
            if (!card.has_value())
            {
                return false;
            }
            oponentDeck.addCard(card.value());
        }
        ClassificationResult playerResult = classifyPlayer(oponentDeck, tableDeck);
        if (playerResult > mainResult)
        {
            return false;
        }
    }
    return true;
}
template <std::size_t N>
inline double probabilityOfWinning(pcg64 &rng, const Deck playerCards, const Deck tableCards, std::size_t numSimulations)
{
    std::size_t wins = 0;
    for (std::size_t i = 0; i < numSimulations; ++i)
    {
        if (playerWinsRandomGame<N>(rng, playerCards, tableCards))
        {
            ++wins;
        }
    }
    return static_cast<double>(wins) / numSimulations;
}
template <std::size_t N>
inline double probabilityOfWinning(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numThreads)
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
                if (playerWinsRandomGame<N>(threadRng, playerCards, tableCards))
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

inline double runGame(const Deck playerDeck, const Deck tableDeck, std::size_t numPlayers, std::size_t threadCount)
{
    if (numPlayers < 2 || numPlayers > 10)
    {
        return 0.0;
    }
    static constexpr std::size_t simulations = 1'000'000;
    switch (numPlayers)
    {
    case 2:
        return probabilityOfWinning<2>(playerDeck, tableDeck, simulations, threadCount);
    case 3:
        return probabilityOfWinning<3>(playerDeck, tableDeck, simulations, threadCount);
    case 4:
        return probabilityOfWinning<4>(playerDeck, tableDeck, simulations, threadCount);
    case 5:
        return probabilityOfWinning<5>(playerDeck, tableDeck, simulations, threadCount);
    case 6:
        return probabilityOfWinning<6>(playerDeck, tableDeck, simulations, threadCount);
    case 7:
        return probabilityOfWinning<7>(playerDeck, tableDeck, simulations, threadCount);
    case 8:
        return probabilityOfWinning<8>(playerDeck, tableDeck, simulations, threadCount);
    case 9:
        return probabilityOfWinning<9>(playerDeck, tableDeck, simulations, threadCount);
    case 10:
        return probabilityOfWinning<10>(playerDeck, tableDeck, simulations, threadCount);
    default:
        return 0.0;
    }
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
    std::cout << "Probability of winning: " << runGame(playerDeck, tableDeck, numPlayers, threadCount) * 100 << "%\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count() << "ms\n";
    return 0;
}