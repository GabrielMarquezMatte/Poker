#include "../include/deck.hpp"
#include "../include/game.hpp"
#include <random>
#include <thread>
#include <string>
#include <chrono>

template <std::size_t N>
bool playerWinsRandomGame(pcg64 &rng, const std::span<const Card, 2> playerCards, const std::span<const Card> tableCards)
{
    Deck deck(rng);
    deck.removeCards(playerCards);
    deck.removeCards(tableCards);
    auto table = deck.dealCards(5 - tableCards.size());
    if (!table.has_value())
    {
        return false;
    }
    auto &tableValue = table.value();
    std::array<Card, 5> tableCardsCopy;
    std::copy(tableCards.begin(), tableCards.end(), tableCardsCopy.begin());
    std::copy(tableValue.begin(), tableValue.end(), tableCardsCopy.begin() + tableCards.size());
    ClassificationResult mainResult = classifyPlayer(playerCards, tableCardsCopy);
    if (mainResult.classification == Classification::RoyalFlush)
    {
        return true;
    }
    for (std::size_t i = 0; i < N - 1; ++i)
    {
        auto cards = deck.dealToPlayer();
        if (!cards.has_value())
        {
            return false;
        }
        auto &cardsValue = cards.value();
        ClassificationResult playerResult = classifyPlayer(cardsValue, tableCardsCopy);
        if (playerResult > mainResult)
        {
            return false;
        }
    }
    return true;
}
template <std::size_t N>
inline double probabilityOfWinning(pcg64 &rng, const std::span<const Card, 2> playerCards, const std::span<const Card> tableCards, std::size_t numSimulations)
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
inline double probabilityOfWinning(const std::span<const Card, 2> playerCards, const std::span<const Card> tableCards, std::size_t numSimulations, std::size_t numThreads)
{
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    std::vector<std::size_t> wins(numThreads, 0);
    std::size_t simulationsPerThread = numSimulations / numThreads;
    for (std::size_t i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([&, i]()
                             {
            pcg64 threadRng(std::random_device{}());
            for (std::size_t j = 0; j < simulationsPerThread; ++j)
            {
                if (playerWinsRandomGame<N>(threadRng, playerCards, tableCards))
                {
                    ++wins[i];
                }
            } });
    }
    for (auto &thread : threads)
    {
        thread.join();
    }
    std::size_t totalWins = 0;
    for (const auto &winCount : wins)
    {
        totalWins += winCount;
    }
    return static_cast<double>(totalWins) / (simulationsPerThread * numThreads);
}
inline constexpr bool checkUniqueCards(const std::span<const Card, 2> cardsPlayer, const std::span<const Card> cardsTable)
{
    for (std::size_t i = 0; i < cardsPlayer.size(); ++i)
    {
        for (std::size_t j = i + 1; j < cardsPlayer.size(); ++j)
        {
            if (cardsPlayer[i] == cardsPlayer[j])
            {
                return false;
            }
        }
    }
    for (std::size_t i = 0; i < cardsTable.size(); ++i)
    {
        for (std::size_t j = i + 1; j < cardsTable.size(); ++j)
        {
            if (cardsTable[i] == cardsTable[j])
            {
                return false;
            }
        }
    }
    for (const Card cardPlayer : cardsPlayer)
    {
        for (const Card cardTable : cardsTable)
        {
            if (cardPlayer == cardTable)
            {
                return false;
            }
        }
    }
    return true;
}

inline double runGame(std::span<const Card, 2> playerCardsSpan, std::span<const Card> tableCardsSpan, std::size_t numPlayers, std::size_t threadCount)
{
    if (numPlayers < 2 || numPlayers > 10)
    {
        return 0.0;
    }
    static constexpr std::size_t simulations = 1'000'000;
    switch (numPlayers)
    {
    case 2:
        return probabilityOfWinning<2>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    case 3:
        return probabilityOfWinning<3>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    case 4:
        return probabilityOfWinning<4>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    case 5:
        return probabilityOfWinning<5>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    case 6:
        return probabilityOfWinning<6>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    case 7:
        return probabilityOfWinning<7>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    case 8:
        return probabilityOfWinning<8>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    case 9:
        return probabilityOfWinning<9>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    case 10:
        return probabilityOfWinning<10>(playerCardsSpan, tableCardsSpan, simulations, threadCount);
    default:
        return 0.0;
    }
}

int main(int argc, const char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <hand> <table> <num_players>\n";
        return 1;
    }
    std::string_view playerHand = argv[1];
    std::optional<std::vector<Card>> playerCards = Card::parseHand(playerHand);
    if (!playerCards.has_value() || playerCards->size() != 2)
    {
        std::cerr << "Invalid hand: " << playerHand << '\n';
        return 1;
    }
    std::vector<Card>& playerCardsValue = *playerCards;
    std::span<const Card, 2> playerCardsSpan(playerCardsValue.data(), playerCardsValue.size());
    std::string_view tableCardsStr = argv[2];
    std::optional<std::vector<Card>> tableCards = Card::parseHand(tableCardsStr);
    if (!tableCards.has_value() || tableCards->size() > 5)
    {
        std::cerr << "Invalid table cards: " << tableCardsStr << '\n';
        return 1;
    }
    std::vector<Card>& tableCardsValue = *tableCards;
    std::span<const Card> tableCardsSpan(tableCardsValue.data(), tableCardsValue.size());
    if (!checkUniqueCards(playerCardsSpan, tableCardsSpan))
    {
        std::cerr << "Duplicate cards found in hand or table.\n";
        return 1;
    }
    pcg64 rng(std::random_device{}());
    Deck deck(rng);
    std::cout << "Player cards: " << playerCardsValue[0] << ", " << playerCardsValue[1] << '\n';
    std::size_t threadCount = std::thread::hardware_concurrency();
    std::string numPlayersStr = argv[3];
    std::size_t numPlayers = std::stoul(numPlayersStr);
    if (numPlayers < 2 || numPlayers > 10)
    {
        std::cerr << "Number of players must be between 2 and 10.\n";
        return 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Probability of winning: " << runGame(playerCardsSpan, tableCardsSpan, numPlayers, threadCount) * 100 << "%\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count() << "ms\n";
    return 0;
}