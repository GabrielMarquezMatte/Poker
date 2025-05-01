#ifndef __POKER_GAME_HPP__
#define __POKER_GAME_HPP__
#include "classification_result.hpp"
#include "hand.hpp"
#include "deck.hpp"
#include <BS_thread_pool.hpp>
#include <vector>
#include <thread>
enum class GameResult
{
    Win,
    Lose,
    Tie,
};
inline constexpr ClassificationResult classifyHand(const Deck allCards) noexcept
{
    ClassificationResult best = {Classification::HighCard, Rank::Two};
    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = i + 1; j < 7; ++j)
        {
            Deck hand = Deck::emptyDeck();
            for (std::size_t l = 0; l < 7; ++l)
            {
                if (l == i || l == j)
                {
                    continue;
                }
                auto cardDealt = allCards.at(l);
                if (!cardDealt.has_value())
                {
                    return {Classification::HighCard, Rank::Two};
                }
                hand.addCard(cardDealt.value());
            }
            ClassificationResult candidate = Hand::classify(hand);
            if (candidate > best)
            {
                best = candidate;
            }
        }
    }
    return best;
}
inline constexpr ClassificationResult classifyPlayer(const Deck playerCards, const Deck tableCards) noexcept
{
    Deck allCards = Deck::createDeck({playerCards, tableCards});
    return classifyHand(allCards);
}
inline constexpr GameResult compareHands(const Deck playerCards, const Deck tableCards, const std::span<const Deck> opponents) noexcept
{
    ClassificationResult playerResult = classifyPlayer(playerCards, tableCards);
    bool sawTie = false;
    for (const auto &opponent : opponents)
    {
        ClassificationResult opponentResult = classifyPlayer(opponent, tableCards);
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
bool playerWinsRandomGame(pcg64 &rng, const Deck playerCards, Deck tableCards, std::size_t numPlayers)
{
    Deck deck = Deck::createFullDeck();
    deck.removeCards(playerCards);
    deck.removeCards(tableCards);
    std::size_t numCardsToDeal = 5 - tableCards.size();
    Deck randomCards = deck.popRandomCards(rng, numCardsToDeal + 2 * (numPlayers - 1));
    for (std::size_t i = 0; i < numCardsToDeal; ++i)
    {
        tableCards.addCard(randomCards.popCard());
    }
    ClassificationResult mainResult = classifyPlayer(playerCards, tableCards);
    Classification mainClassification = mainResult.getClassification();
    if (mainClassification == Classification::RoyalFlush)
    {
        return true;
    }
    ClassificationResult boardResult = classifyPlayer(Deck::emptyDeck(), tableCards);
    if (boardResult.getClassification() >= mainClassification && boardResult.getRankFlag() == mainResult.getRankFlag())
    {
        return false;
    }
    for (std::size_t i = 0; i < numPlayers - 1; ++i)
    {
        Card opponentCard1 = deck.popCard();
        Card opponentCard2 = deck.popCard();
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
double probabilityOfWinning(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers, BS::thread_pool<BS::tp::none> &threadPool)
{
    std::size_t numThreads = threadPool.get_thread_count();
    std::size_t simulationsPerThread = numSimulations / numThreads;
    std::vector<std::future<std::size_t>> threads;
    threads.reserve(numThreads);
    std::size_t remainingSimulations = numSimulations % numThreads;
    for (std::size_t i = 0; i < numThreads; ++i)
    {
        threads.push_back(threadPool.submit_task([&, i]()
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
            return threadWins; }));
    }
    std::size_t wins = 0;
    pcg64 threadRng(std::random_device{}());
    for (std::size_t i = 0; i < remainingSimulations; ++i)
    {
        if (playerWinsRandomGame(threadRng, playerCards, tableCards, numPlayers))
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
    std::size_t remainingSimulations = numSimulations % numThreads;
    pcg64 threadRng(std::random_device{}());
    std::size_t threadWins = 0;
    for (std::size_t i = 0; i < remainingSimulations; ++i)
    {
        if (playerWinsRandomGame(threadRng, playerCards, tableCards, numPlayers))
        {
            ++threadWins;
        }
    }
    wins += threadWins;
    for (auto &thread : threads)
    {
        thread.join();
    }
    return static_cast<double>(wins) / numSimulations;
}
#endif // __POKER_GAME_HPP__