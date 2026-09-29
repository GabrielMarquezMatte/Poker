#ifndef __POKER_GAME_HPP__
#define __POKER_GAME_HPP__
#include "classification_result.hpp"
#include "hand.hpp"
#include "deck.hpp"
#include "preflop_table.hpp"
#include <BS_thread_pool.hpp>
#include <optional>
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
template <typename TRng>
inline constexpr std::size_t playRandomShowdown(TRng &rng, const Deck playerCards, Deck tableCards, Deck deck, std::size_t numPlayers) noexcept
{
    std::size_t numCardsToDeal = 5 - tableCards.size();
    if (numCardsToDeal)
    {
        tableCards.addCards(deck.popRandomCards(rng, numCardsToDeal));
    }
    ClassificationResult playerResult = Hand::classify(Deck::createDeck({playerCards, tableCards}));
    std::size_t winners = 1;
    for (std::size_t i = 0; i < numPlayers - 1; ++i)
    {
        Deck opponent = deck.popPair(rng);
        ClassificationResult opponentResult = Hand::classify(Deck::createDeck({opponent, tableCards}));
        if (opponentResult > playerResult)
        {
            return 0;
        }
        winners += opponentResult == playerResult;
    }
    return winners;
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
    static constexpr std::uint64_t fullPot = 2520;
    std::size_t wins = 0;
    std::size_t losses = 0;
    std::size_t ties = 0;
    std::uint64_t potShares = 0;
    inline constexpr std::size_t totalGames() const noexcept
    {
        return wins + losses + ties;
    }
    inline constexpr double notLosing() const noexcept
    {
        return static_cast<double>(wins + ties) / static_cast<double>(totalGames());
    }
    inline constexpr double equity() const noexcept
    {
        return static_cast<double>(potShares) / static_cast<double>(fullPot * totalGames());
    }
    inline constexpr void record(std::size_t winners) noexcept
    {
        if (winners == 0)
        {
            ++losses;
            return;
        }
        ++(winners == 1 ? wins : ties);
        potShares += fullPot / winners;
    }
    inline constexpr GameStatistics &operator+=(const GameStatistics &other) noexcept
    {
        wins += other.wins;
        losses += other.losses;
        ties += other.ties;
        potShares += other.potShares;
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
        stats.record(playRandomShowdown(rng, playerCards, tableCards, deck, numPlayers));
    }
    return stats;
}

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

inline constexpr double exactDealCount(const Deck playerCards, const Deck tableCards, std::size_t numPlayers) noexcept
{
    return exactDealCount(52 - playerCards.size() - tableCards.size(), 5 - tableCards.size(), numPlayers - 1);
}

template <typename F>
inline constexpr void forEachPair(std::uint64_t cards, F &&f)
{
    for (; cards; cards &= cards - 1)
    {
        const std::uint64_t lowest = cards & (~cards + 1);
        for (std::uint64_t rest = cards & (cards - 1); rest; rest &= rest - 1)
        {
            f(lowest | (rest & (~rest + 1)));
        }
    }
}

template <typename F>
inline constexpr void forEachCombination(std::uint64_t cards, std::size_t k, std::uint64_t chosen, F &&f)
{
    if (k == 2)
    {
        forEachPair(cards, [&](std::uint64_t pair)
                    { f(chosen | pair); });
        return;
    }
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

using TieCounts = std::array<std::uint64_t, 10>;

// Adds to counts[t] the unordered sets of k pairwise-disjoint hands from hands[start, size) that tie the player t times.
// Hands at index >= tieStart tie the player, the rest lose to it.
inline constexpr void countDisjointHands(const std::uint64_t *hands, std::size_t size, std::size_t tieStart, std::size_t start, std::size_t k, std::uint64_t used, std::size_t ties, TieCounts &counts) noexcept
{
    if (k == 0)
    {
        ++counts[ties];
        return;
    }
    if (k == 1)
    {
        std::uint64_t clear = 0;
        std::uint64_t clearTies = 0;
        for (std::size_t i = start; i < tieStart; ++i)
        {
            clear += (hands[i] & used) == 0;
        }
        for (std::size_t i = std::max(start, tieStart); i < size; ++i)
        {
            clearTies += (hands[i] & used) == 0;
        }
        counts[ties] += clear;
        counts[ties + 1] += clearTies;
        return;
    }
    for (std::size_t i = start; i < size; ++i)
    {
        if ((hands[i] & used) == 0)
        {
            countDisjointHands(hands, size, tieStart, i + 1, k - 1, used | hands[i], ties + (i >= tieStart), counts);
        }
    }
}

// With parts > 1 losses are left at 0: the caller derives them from the total once all parts are summed.
inline constexpr GameStatistics exactGameStatistics(const Deck playerCards, const Deck tableCards, std::size_t numPlayers, std::size_t part = 0, std::size_t parts = 1)
{
    GameStatistics stats;
    const std::uint64_t deck = Deck::createFullDeck().getMask() & ~playerCards.getMask() & ~tableCards.getMask();
    const std::size_t boardMissing = 5 - tableCards.size();
    const std::size_t opponents = numPlayers - 1;
    std::uint64_t orderings = 1;
    for (std::size_t i = 2; i <= opponents; ++i)
    {
        orderings *= i;
    }
    const bool splitBoards = exactDealCount(static_cast<std::size_t>(std::popcount(deck)), boardMissing, 0) >= 4.0 * static_cast<double>(parts);
    std::size_t owner = 0;
    const auto mine = [&]()
    {
        const bool result = owner == part;
        owner = owner + 1 == parts ? 0 : owner + 1;
        return result;
    };
    forEachCombination(deck, boardMissing, 0, [&](std::uint64_t boardRest)
                       {
        if (splitBoards && !mine())
        {
            return;
        }
        const std::uint64_t board = tableCards.getMask() | boardRest;
        const std::uint64_t rest = deck & ~boardRest;
        const ClassificationResult player = Hand::classify(Deck::from_mask(playerCards.getMask() | board));
        // Classify each opponent hand once per board; hands that beat the player only ever produce losses.
        constexpr std::size_t maxHands = 45 * 44 / 2;
        std::array<std::uint64_t, maxHands> hands;
        std::array<std::uint64_t, maxHands> tiedHands;
        std::size_t size = 0;
        std::size_t tieCount = 0;
        forEachPair(rest, [&](std::uint64_t hole)
                    {
            const ClassificationResult result = Hand::classify(Deck::from_mask(hole | board));
            if (result < player)
            {
                hands[size++] = hole;
            }
            else if (result == player)
            {
                tiedHands[tieCount++] = hole;
            } });
        const std::size_t tieStart = size;
        std::copy_n(tiedHands.begin(), tieCount, hands.begin() + size);
        size += tieCount;
        // Opponents are interchangeable, so count unordered sets and scale by the orderings.
        TieCounts counts{};
        for (std::size_t i = 0; i < size; ++i)
        {
            if (splitBoards || mine())
            {
                countDisjointHands(hands.data(), size, tieStart, i + 1, opponents - 1, hands[i], i >= tieStart, counts);
            }
        }
        for (std::size_t ties = 0; ties < counts.size(); ++ties)
        {
            const std::uint64_t games = counts[ties] * orderings;
            (ties == 0 ? stats.wins : stats.ties) += games;
            stats.potShares += games * (GameStatistics::fullPot / (ties + 1));
        } });
    if (parts == 1)
    {
        stats.losses = static_cast<std::size_t>(exactDealCount(static_cast<std::size_t>(std::popcount(deck)), boardMissing, opponents)) - stats.wins - stats.ties;
    }
    return stats;
}

// More tasks than threads so fast cores (P-cores on hybrid CPUs) pick up the slack instead of idling.
inline constexpr std::size_t tasksPerThread = 8;

inline GameStatistics exactGameStatistics(const Deck playerCards, const Deck tableCards, std::size_t numPlayers, BS::thread_pool<BS::tp::none> &threadPool)
{
    const std::size_t parts = threadPool.get_thread_count() * tasksPerThread;
    std::vector<std::future<GameStatistics>> results;
    results.reserve(parts);
    for (std::size_t part = 0; part < parts; ++part)
    {
        results.push_back(threadPool.submit_task([=]()
                                                 { return exactGameStatistics(playerCards, tableCards, numPlayers, part, parts); }));
    }
    GameStatistics stats;
    for (auto &result : results)
    {
        stats += result.get();
    }
    stats.losses = static_cast<std::size_t>(exactDealCount(playerCards, tableCards, numPlayers)) - stats.wins - stats.ties;
    return stats;
}

inline constexpr std::size_t preflopClassIndex(const Deck holeCards) noexcept
{
    const std::uint64_t mask = holeCards.getMask();
    const int first = std::countr_zero(mask);
    const int second = 63 - std::countl_zero(mask);
    const int high = std::max(first % 13, second % 13);
    const int low = std::min(first % 13, second % 13);
    const bool suited = first / 13 == second / 13;
    return static_cast<std::size_t>(suited ? high * 13 + low : low * 13 + high);
}

inline constexpr GameStatistics preflopStatistics(const Deck holeCards, std::size_t numPlayers) noexcept
{
    if (holeCards.size() != 2 || numPlayers < 2 || numPlayers > preflopTable.size() + 1)
    {
        return {};
    }
    const PreflopEntry entry = preflopTable[numPlayers - 2][preflopClassIndex(holeCards)];
    return {entry.wins, entry.losses, entry.ties, entry.potShares};
}

inline GameStatistics simulateGameStatistics(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers, BS::thread_pool<BS::tp::none> &threadPool)
{
    constexpr std::size_t minSimulationsPerTask = 4096;
    const std::size_t numThreads = threadPool.get_thread_count();
    const std::size_t numTasks = std::max(numThreads, std::min(numThreads * tasksPerThread, numSimulations / minSimulationsPerTask));
    std::vector<std::future<GameStatistics>> tasks;
    tasks.reserve(numTasks);
    std::random_device rd{};
    for (std::size_t i = 0; i < numTasks; ++i)
    {
        const std::size_t simulations = numSimulations / numTasks + (i < numSimulations % numTasks);
        tasks.push_back(threadPool.submit_task([&, simulations, seed = rd()]()
                                               {
            omp::XoroShiro128Plus taskRng(seed);
            return computeRandomGameStatistics(taskRng, playerCards, tableCards, simulations, numPlayers); }));
    }
    GameStatistics stats;
    for (auto &task : tasks)
    {
        stats += task.get();
    }
    return stats;
}

inline constexpr bool preferExact(double deals, std::size_t numSimulations) noexcept
{
    return deals <= 1.5 * static_cast<double>(numSimulations);
}

inline std::optional<GameStatistics> quickGameStatistics(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers)
{
    constexpr double inlineWork = 50'000;
    if (tableCards.size() == 0)
    {
        const GameStatistics stats = preflopStatistics(playerCards, numPlayers);
        if (stats.totalGames() >= numSimulations)
        {
            return stats;
        }
    }
    const double deals = exactDealCount(playerCards, tableCards, numPlayers);
    if (preferExact(deals, numSimulations))
    {
        if (deals <= inlineWork)
        {
            return exactGameStatistics(playerCards, tableCards, numPlayers);
        }
        return std::nullopt;
    }
    if (static_cast<double>(numSimulations) <= inlineWork)
    {
        omp::XoroShiro128Plus rng(std::random_device{}());
        return computeRandomGameStatistics(rng, playerCards, tableCards, numSimulations, numPlayers);
    }
    return std::nullopt;
}

inline GameStatistics computeRandomGameStatistics(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers, BS::thread_pool<BS::tp::none> &threadPool)
{
    if (const auto quick = quickGameStatistics(playerCards, tableCards, numSimulations, numPlayers))
    {
        return *quick;
    }
    if (preferExact(exactDealCount(playerCards, tableCards, numPlayers), numSimulations))
    {
        return exactGameStatistics(playerCards, tableCards, numPlayers, threadPool);
    }
    return simulateGameStatistics(playerCards, tableCards, numSimulations, numPlayers, threadPool);
}

inline double probabilityOfWinning(const Deck playerCards, const Deck tableCards, std::size_t numSimulations, std::size_t numPlayers, BS::thread_pool<BS::tp::none> &threadPool)
{
    return computeRandomGameStatistics(playerCards, tableCards, numSimulations, numPlayers, threadPool).notLosing();
}
#endif // __POKER_GAME_HPP__
