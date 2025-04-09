#include "../include/deck.hpp"
#include "../include/game.hpp"
#include <random>
#include <thread>

template <std::size_t N>
bool playerWinsRandomGame(pcg64 &rng, const std::span<const Card> playerCards, const std::span<const Card> tableCards)
{
    if (playerCards.size() != 2 || tableCards.size() > 5)
    {
        return false;
    }
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
    std::array<Player, N> players{};
    for (int i = 0; i < N - 1; ++i)
    {
        auto cards = deck.dealToPlayer();
        if (!cards.has_value())
        {
            return false;
        }
        auto &cardsValue = cards.value();
        players[i] = Player(cardsValue[0], cardsValue[1], tableCardsCopy);
    }
    players[N - 1] = Player(playerCards[0], playerCards[1], tableCardsCopy);
    Game<N> game(players, tableCardsCopy);
    auto results = game.classify();
    return results[0].first == players[N - 1];
}
template <std::size_t N>
inline double probabilityOfWinning(pcg64 &rng, const std::span<const Card> playerCards, const std::span<const Card> tableCards, std::size_t numSimulations)
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
inline double probabilityOfWinning(const std::span<const Card> playerCards, const std::span<const Card> tableCards, std::size_t numSimulations, std::size_t numThreads)
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
inline constexpr bool checkUniqueCards(const std::span<const Card> cardsPlayer, const std::span<const Card> cardsTable)
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
int main(int argc, const char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <hand> <table>\n";
        return 1;
    }
    static constexpr std::size_t numPlayers = 5;
    std::string_view playerHand = argv[1];
    std::optional<std::vector<Card>> playerCards = Card::parseHand(playerHand);
    if (!playerCards.has_value() || playerCards->size() != 2)
    {
        std::cerr << "Invalid hand: " << playerHand << '\n';
        return 1;
    }
    std::vector<Card>& playerCardsValue = *playerCards;
    std::array<Card, 2> playerCardsArray = {playerCardsValue[0], playerCardsValue[1]};
    std::string_view tableCardsStr = argv[2];
    std::optional<std::vector<Card>> tableCards = Card::parseHand(tableCardsStr);
    if (!tableCards.has_value() || tableCards->size() > 5)
    {
        std::cerr << "Invalid table cards: " << tableCardsStr << '\n';
        return 1;
    }
    std::vector<Card>& tableCardsValue = *tableCards;
    if (!checkUniqueCards(playerCardsValue, tableCardsValue))
    {
        std::cerr << "Duplicate cards found in hand or table.\n";
        return 1;
    }
    pcg64 rng(std::random_device{}());
    Deck deck(rng);
    std::cout << "Player cards: " << playerCardsValue[0] << ", " << playerCardsValue[1] << '\n';
    std::size_t threadCount = std::thread::hardware_concurrency();
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Probability of winning: " << probabilityOfWinning<numPlayers>(playerCardsArray, tableCardsValue, 1'000'000, threadCount) * 100 << "%\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count() << "ms\n";
    return 0;
}