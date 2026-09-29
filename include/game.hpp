#ifndef __POKER_GAME_HPP__
#define __POKER_GAME_HPP__
#include "classification_result.hpp"
#include "hand.hpp"
#include "deck.hpp"
#include <BS_thread_pool.hpp>
#include <span>
#include <thread>
enum class GameResult : std::uint8_t
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
template<typename TRng>
inline constexpr GameResult compareRandomHands(TRng &rng, const Deck playerCards, Deck tableCards, Deck deck, std::size_t numPlayers) noexcept
{
    std::size_t numCardsToDeal = 5 - tableCards.size();
    if (numCardsToDeal)
    {
        tableCards.addCards(deck.popRandomCards(rng, numCardsToDeal));
    }
    ClassificationResult playerResult = Hand::classify(Deck::createDeck({playerCards, tableCards}));
    bool sawTie = false;
    for (std::size_t i = 0; i < numPlayers - 1; ++i)
    {
        Deck opponent = deck.popPair(rng);
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
struct GameStatistics
{
    std::size_t wins = 0;
    std::size_t losses = 0;
    std::size_t ties = 0;
    inline constexpr std::size_t totalGames() const noexcept
    {
        return wins + losses + ties;
    }
    inline constexpr void add(GameResult result) noexcept
    {
        switch (result)
        {
        case GameResult::Win:
            ++wins;
            break;
        case GameResult::Lose:
            ++losses;
            break;
        case GameResult::Tie:
            ++ties;
            break;
        }
    }
    inline constexpr GameStatistics &operator+=(const GameStatistics &other) noexcept
    {
        wins += other.wins;
        losses += other.losses;
        ties += other.ties;
        return *this;
    }
};
template <typename TRng>
inline GameStatistics computeRandomGameStatistics(TRng &rng, const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers)
{
    GameStatistics stats;
    Deck deck = Deck::createFullDeck();
    deck.removeCards(playerCards);
    deck.removeCards(tableCards);
    for (std::size_t i = 0; i < numSimulations; ++i)
    {
        stats.add(compareRandomHands(rng, playerCards, tableCards, deck, numPlayers));
    }
    return stats;
}

// Ordered ways to deal the missing board cards and then `opponents` hole pairs from `deckSize` cards.
inline constexpr double exactDealCount(std::size_t deckSize, std::size_t boardMissing, std::size_t opponents) noexcept
{
    double ways = 1;
    for (std::size_t i = 0; i < boardMissing; ++i)
    {
        ways = ways * static_cast<double>(deckSize - i) / static_cast<double>(i + 1);
    }
    deckSize -= boardMissing;
    for (std::size_t i = 0; i < opponents; ++i, deckSize -= 2)
    {
        ways *= static_cast<double>(deckSize * (deckSize - 1) / 2);
    }
    return ways;
}

// Calls f(subset) for every k-card subset of `cards`.
template <typename F>
inline constexpr void forEachCombination(std::uint64_t cards, std::size_t k, std::uint64_t chosen, F &&f)
{
    if (k == 0)
    {
        f(chosen);
        return;
    }
    for (; static_cast<std::size_t>(std::popcount(cards)) >= k; cards &= cards - 1)
    {
        const std::uint64_t lowest = cards & (~cards + 1);
        forEachCombination(cards & (cards - 1), k - 1, chosen | lowest, f);
    }
}

inline constexpr void enumerateOpponents(std::uint64_t deck, std::uint64_t board, ClassificationResult player, std::size_t opponents, bool sawTie, GameStatistics &stats)
{
    if (opponents == 0)
    {
        ++(sawTie ? stats.ties : stats.wins);
        return;
    }
    // A loss is decided as soon as one opponent wins; count every deal of the remaining opponents at once.
    const auto remainingDeals = static_cast<std::size_t>(exactDealCount(static_cast<std::size_t>(std::popcount(deck)) - 2, 0, opponents - 1));
    forEachCombination(deck, 2, 0, [&](std::uint64_t hole)
                       {
        const ClassificationResult result = Hand::classify(Deck::from_mask(hole | board));
        if (result > player)
        {
            stats.losses += remainingDeals;
            return;
        }
        enumerateOpponents(deck & ~hole, board, player, opponents - 1, sawTie || result == player, stats); });
}

// Every possible deal counted once (opponents in seat order), so the ratios are exact.
inline constexpr GameStatistics exactGameStatistics(const Deck playerCards, const Deck tableCards, std::size_t numPlayers)
{
    GameStatistics stats;
    const std::uint64_t deck = Deck::createFullDeck().getMask() & ~playerCards.getMask() & ~tableCards.getMask();
    forEachCombination(deck, 5 - tableCards.size(), 0, [&](std::uint64_t boardRest)
                       {
        const std::uint64_t board = tableCards.getMask() | boardRest;
        const ClassificationResult player = Hand::classify(Deck::from_mask(playerCards.getMask() | board));
        enumerateOpponents(deck & ~boardRest, board, player, numPlayers - 1, false, stats); });
    return stats;
}

inline GameStatistics computeRandomGameStatistics(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers, BS::thread_pool<BS::tp::none> &threadPool)
{
    const std::size_t numThreads = threadPool.get_thread_count();
    // Enumeration is exact and, up to this size, no slower than the parallel simulation.
    const std::size_t deckSize = 52 - playerCards.size() - tableCards.size();
    if (exactDealCount(deckSize, 5 - tableCards.size(), numPlayers - 1) <= static_cast<double>(numSimulations / numThreads))
    {
        return exactGameStatistics(playerCards, tableCards, numPlayers);
    }

    const std::size_t simulationsPerThread = numSimulations / numThreads;
    std::vector<std::future<GameStatistics>> threads;
    threads.reserve(numThreads);
    std::random_device rd{};
    for (std::size_t i = 0; i < numThreads; ++i)
    {
        threads.push_back(threadPool.submit_task([&, seed = rd()]()
                                                 {
            omp::XoroShiro128Plus threadRng(seed);
            return computeRandomGameStatistics(threadRng, playerCards, tableCards, simulationsPerThread, numPlayers); }));
    }
    omp::XoroShiro128Plus rng(rd());
    GameStatistics stats = computeRandomGameStatistics(rng, playerCards, tableCards, numSimulations % numThreads, numPlayers);
    for (auto &thread : threads)
    {
        stats += thread.get();
    }
    return stats;
}

// Probability of not losing: ties count as wins.
inline double probabilityOfWinning(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers, BS::thread_pool<BS::tp::none> &threadPool)
{
    const GameStatistics stats = computeRandomGameStatistics(playerCards, tableCards, numSimulations, numPlayers, threadPool);
    return static_cast<double>(stats.wins + stats.ties) / static_cast<double>(stats.totalGames());
}
#endif // __POKER_GAME_HPP__
