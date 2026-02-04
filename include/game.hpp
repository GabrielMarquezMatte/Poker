#ifndef __POKER_GAME_HPP__
#define __POKER_GAME_HPP__
#include "classification_result.hpp"
#include "hand.hpp"
#include "deck.hpp"
#include <BS_thread_pool.hpp>
#include <span>
#include <thread>
enum class GameResult
{
    Win,
    Lose,
    Tie,
};
inline constexpr GameResult compareHands(const Deck playerCards, const Deck tableCards, const std::span<const Deck> opponents) noexcept
{
    ClassificationResult playerResult = Hand::classify(Deck::createDeck({playerCards, tableCards}));
    bool sawTie = false;
    for (const auto &opponent : opponents)
    {
        ClassificationResult opponentResult = Hand::classify(Deck::createDeck({opponent, tableCards}));
        if (opponentResult > playerResult)
        {
            return GameResult::Lose;
        }
        if (opponentResult == playerResult)
        {
            sawTie = true;
        }
    }
    if (sawTie)
    {
        return GameResult::Tie;
    }
    return GameResult::Win;
}
template <typename TRng>
inline bool playerWinsRandomGame(TRng &rng, const Deck playerCards, Deck tableCards, Deck deck, std::size_t numPlayers)
{
    std::size_t numCardsToDeal = 5 - tableCards.size();
    if (numCardsToDeal)
    {
        tableCards.addCards(deck.popRandomCards(rng, numCardsToDeal));
    }
    ClassificationResult mainResult = Hand::classify(Deck::createDeck({playerCards, tableCards}));
    for (std::size_t i = 0; i < numPlayers - 1; ++i)
    {
        Deck opp = deck.popPair(rng);
        const auto oppResult = Hand::classify(Deck::createDeck({opp, tableCards}));
        if (oppResult > mainResult)
        {
            return false;
        }
    }
    return true;
}
template <typename TRng>
inline constexpr double probabilityOfWinning(TRng &rng, const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers)
{
    std::size_t wins = 0;
    Deck deck = Deck::createFullDeck();
    deck.removeCards(playerCards);
    deck.removeCards(tableCards);
    for (std::size_t i = 0; i < numSimulations; ++i)
    {
        if (playerWinsRandomGame(rng, playerCards, tableCards, deck, numPlayers))
        {
            ++wins;
        }
    }
    return static_cast<double>(wins) / numSimulations;
}
inline double probabilityOfWinning(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers, BS::thread_pool<BS::tp::none> &threadPool)
{
    std::size_t numThreads = threadPool.get_thread_count();
    std::size_t simulationsPerThread = numSimulations / numThreads;
    std::vector<std::future<std::size_t>> threads;
    threads.reserve(numThreads);
    std::size_t remainingSimulations = numSimulations % numThreads;
    Deck deck = Deck::createFullDeck();
    deck.removeCards(playerCards);
    deck.removeCards(tableCards);
    for (std::size_t i = 0; i < numThreads; ++i)
    {
        threads.push_back(threadPool.submit_task([&, deck, i]()
                               {
            omp::XoroShiro128Plus threadRng(std::random_device{}());
            std::size_t threadWins = 0;
            Deck threadDeck = deck;
            for (std::size_t j = 0; j < simulationsPerThread; ++j)
            {
                if (playerWinsRandomGame(threadRng, playerCards, tableCards, threadDeck, numPlayers))
                {
                    ++threadWins;
                }
            }
            return threadWins; }));
    }
    std::size_t wins = 0;
    omp::XoroShiro128Plus threadRng(std::random_device{}());
    Deck threadDeck = deck;
    for (std::size_t i = 0; i < remainingSimulations; ++i)
    {
        if (playerWinsRandomGame(threadRng, playerCards, tableCards, threadDeck, numPlayers))
        {
            ++wins;
        }
    }
    for (auto &thread : threads)
    {
        wins += thread.get();
    }
    return static_cast<double>(wins) / numSimulations;
}
#endif // __POKER_GAME_HPP__