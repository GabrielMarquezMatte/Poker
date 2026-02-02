#include <benchmark/benchmark.h>
#include "../include/game.hpp"

// ============================================================================
// Deck Creation and Card Operations
// ============================================================================

static void BM_CreateRandom7Cards(benchmark::State &state)
{
    omp::XoroShiro128Plus rng(state.thread_index() + state.iterations());
    for (auto _ : state)
    {
        Deck deck = Deck::createFullDeck();
        Deck hand = deck.popRandomCards(rng, 7);
        benchmark::DoNotOptimize(hand);
    }
}
BENCHMARK(BM_CreateRandom7Cards);

static void BM_CreateRandom7CardsSequential(benchmark::State &state)
{
    omp::XoroShiro128Plus rng(state.thread_index() + state.iterations());
    for (auto _ : state)
    {
        Deck deck = Deck::createFullDeck();
        Deck hand = Deck::emptyDeck();
        for (std::size_t i = 0; i < 7; ++i)
        {
            hand.addCard(deck.popRandomCard(rng));
        }
        benchmark::DoNotOptimize(hand);
    }
}
BENCHMARK(BM_CreateRandom7CardsSequential);

static void BM_CreateFullDeck(benchmark::State &state)
{
    for (auto _ : state)
    {
        Deck deck = Deck::createFullDeck();
        benchmark::DoNotOptimize(deck);
    }
}
BENCHMARK(BM_CreateFullDeck);

static void BM_PopRandomCards(benchmark::State &state)
{
    omp::XoroShiro128Plus rng(state.thread_index() + state.iterations());
    std::size_t numCards = state.range(0);
    for (auto _ : state)
    {
        Deck deck = Deck::createFullDeck();
        Deck cards = deck.popRandomCards(rng, numCards);
        benchmark::DoNotOptimize(cards);
        benchmark::DoNotOptimize(deck);
    }
}
BENCHMARK(BM_PopRandomCards)->DenseRange(1, 10, 1);

static void BM_DeckIteration(benchmark::State &state)
{
    Deck deck = Deck::createFullDeck();
    for (auto _ : state)
    {
        std::size_t count = 0;
        for (const auto &card : deck)
        {
            benchmark::DoNotOptimize(card);
            ++count;
        }
        benchmark::DoNotOptimize(count);
    }
}
BENCHMARK(BM_DeckIteration);

static void BM_ParseHand(benchmark::State &state)
{
    for (auto _ : state)
    {
        Deck deck = Deck::parseHand("As Kh Qd Jc Ts 9h 8d");
        benchmark::DoNotOptimize(deck);
    }
}
BENCHMARK(BM_ParseHand);

// ============================================================================
// Hand Classification Benchmarks
// ============================================================================

static void BM_Classification(benchmark::State &state)
{
    omp::XoroShiro128Plus rng(state.thread_index() + state.iterations());
    Deck deck = Deck::createFullDeck();
    Deck cards = deck.popRandomCards(rng, 7);
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(cards);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Classification);

static void BM_ClassificationVaryingHands(benchmark::State &state)
{
    omp::XoroShiro128Plus rng(state.thread_index() + state.iterations());
    for (auto _ : state)
    {
        Deck deck = Deck::createFullDeck();
        Deck cards = deck.popRandomCards(rng, 7);
        ClassificationResult result = Hand::classify(cards);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassificationVaryingHands);

static void BM_ClassifyRoyalFlush(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Ks Qs Js Ts 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyRoyalFlush);

static void BM_ClassifyStraightFlush(benchmark::State &state)
{
    Deck deck = Deck::parseHand("9s 8s 7s 6s 5s 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyStraightFlush);

static void BM_ClassifyFourOfAKind(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Ah Ad Ac Ks 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyFourOfAKind);

static void BM_ClassifyFullHouse(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Ah Ad Ks Kh 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyFullHouse);

static void BM_ClassifyFlush(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Ks Qs Js 9s 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyFlush);

static void BM_ClassifyStraight(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Kh Qd Jc Ts 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyStraight);

static void BM_ClassifyThreeOfAKind(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Ah Ad Ks Qh 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyThreeOfAKind);

static void BM_ClassifyTwoPair(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Ah Ks Kh Qd 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyTwoPair);

static void BM_ClassifyOnePair(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Ah Ks Qh Jd 2h 3d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyOnePair);

static void BM_ClassifyHighCard(benchmark::State &state)
{
    Deck deck = Deck::parseHand("As Kh Qd Jc 9s 2h 4d");
    for (auto _ : state)
    {
        ClassificationResult result = Hand::classify(deck);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ClassifyHighCard);

// ============================================================================
// Hand Comparison Benchmarks
// ============================================================================

static void BM_CompareHands(benchmark::State &state)
{
    Deck player = Deck::parseHand("As Ah");
    Deck opp = Deck::parseHand("Ks Kh");
    Deck board = Deck::parseHand("Qd Jc Ts 2h 3d");
    std::array<Deck, 1> opps{opp};
    for (auto _ : state)
    {
        GameResult result = compareHands(player, board, opps);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CompareHands);

static void BM_CompareHandsMultipleOpponents(benchmark::State &state)
{
    Deck player = Deck::parseHand("As Ah");
    std::array<Deck, 5> opps = {
        Deck::parseHand("Ks Kh"),
        Deck::parseHand("Qs Qh"),
        Deck::parseHand("Js Jh"),
        Deck::parseHand("Ts Th"),
        Deck::parseHand("9s 9h")};
    Deck board = Deck::parseHand("2d 3c 4s 5h 7d");
    for (auto _ : state)
    {
        GameResult result = compareHands(player, board, opps);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CompareHandsMultipleOpponents);

// ============================================================================
// Game Simulation Benchmarks
// ============================================================================

static void BM_PlayerWinsRandomGame(benchmark::State &st)
{
    omp::XoroShiro128Plus rng(st.thread_index() + st.iterations());
    Deck deck = Deck::createFullDeck();
    Deck allCards = deck.popRandomCards(rng, 7);
    Deck playerCards = allCards.popCards(2);
    Deck tableCards = allCards.popCards(5);
    std::size_t numPlayers = st.range(0);
    Deck deckForGame = Deck::createFullDeck();
    deckForGame.removeCards(playerCards);
    deckForGame.removeCards(tableCards);
    for (auto _ : st)
    {
        bool result = playerWinsRandomGame(rng, playerCards, tableCards, deckForGame, numPlayers);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_PlayerWinsRandomGame)->DenseRange(2, 10, 1);

static void BM_PlayerWinsRandomGamePreflop(benchmark::State &st)
{
    omp::XoroShiro128Plus rng(st.thread_index() + st.iterations());
    Deck deck = Deck::createFullDeck();
    Deck playerCards = deck.popRandomCards(rng, 2);
    Deck tableCards = Deck::emptyDeck(); // No community cards yet
    std::size_t numPlayers = st.range(0);
    Deck deckForGame = Deck::createFullDeck();
    deckForGame.removeCards(playerCards);
    for (auto _ : st)
    {
        bool result = playerWinsRandomGame(rng, playerCards, tableCards, deckForGame, numPlayers);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_PlayerWinsRandomGamePreflop)->DenseRange(2, 10, 1);

static void BM_PlayerWinsRandomGameFlop(benchmark::State &st)
{
    omp::XoroShiro128Plus rng(st.thread_index() + st.iterations());
    Deck deck = Deck::createFullDeck();
    Deck playerCards = deck.popRandomCards(rng, 2);
    Deck tableCards = deck.popRandomCards(rng, 3); // Flop only
    std::size_t numPlayers = st.range(0);
    Deck deckForGame = Deck::createFullDeck();
    deckForGame.removeCards(playerCards);
    deckForGame.removeCards(tableCards);
    for (auto _ : st)
    {
        bool result = playerWinsRandomGame(rng, playerCards, tableCards, deckForGame, numPlayers);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_PlayerWinsRandomGameFlop)->DenseRange(2, 10, 1);

static void BM_PlayerWinsRandomGameTurn(benchmark::State &st)
{
    omp::XoroShiro128Plus rng(st.thread_index() + st.iterations());
    Deck deck = Deck::createFullDeck();
    Deck playerCards = deck.popRandomCards(rng, 2);
    Deck tableCards = deck.popRandomCards(rng, 4); // Flop + Turn
    std::size_t numPlayers = st.range(0);
    Deck deckForGame = Deck::createFullDeck();
    deckForGame.removeCards(playerCards);
    deckForGame.removeCards(tableCards);
    for (auto _ : st)
    {
        bool result = playerWinsRandomGame(rng, playerCards, tableCards, deckForGame, numPlayers);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_PlayerWinsRandomGameTurn)->DenseRange(2, 10, 1);

// ============================================================================
// Probability of Winning Benchmarks (Sequential)
// ============================================================================

static void BM_ProbabilityOfWinningSequential(benchmark::State &st)
{
    omp::XoroShiro128Plus rng(st.thread_index() + st.iterations());
    Deck deck = Deck::createFullDeck();
    Deck allCards = deck.popRandomCards(rng, 7);
    Deck playerCards = allCards.popCards(2);
    Deck tableCards = allCards.popCards(5);
    std::size_t numPlayers = st.range(0);
    std::size_t numSimulations = st.range(1);
    for (auto _ : st)
    {
        double probability = probabilityOfWinning(rng, playerCards, tableCards, numSimulations, numPlayers);
        benchmark::DoNotOptimize(probability);
    }
}
BENCHMARK(BM_ProbabilityOfWinningSequential)->Ranges({{2, 8}, {10'000, 1'000'000}})->Unit(benchmark::kMillisecond);

static void BM_ProbabilityOfWinningPreflop(benchmark::State &st)
{
    omp::XoroShiro128Plus rng(st.thread_index() + st.iterations());
    Deck deck = Deck::createFullDeck();
    Deck playerCards = deck.popRandomCards(rng, 2);
    Deck tableCards = Deck::emptyDeck();
    std::size_t numPlayers = st.range(0);
    std::size_t numSimulations = 10'000;
    for (auto _ : st)
    {
        double probability = probabilityOfWinning(rng, playerCards, tableCards, numSimulations, numPlayers);
        benchmark::DoNotOptimize(probability);
    }
}
BENCHMARK(BM_ProbabilityOfWinningPreflop)->DenseRange(2, 10, 1)->Unit(benchmark::kMillisecond);

// ============================================================================
// Probability of Winning Benchmarks (Parallel)
// ============================================================================

static void BM_ProbabilityOfWinningParallel(benchmark::State &st)
{
    omp::XoroShiro128Plus rng(st.thread_index() + st.iterations());
    Deck deck = Deck::createFullDeck();
    Deck allCards = deck.popRandomCards(rng, 7);
    Deck playerCards = allCards.popCards(2);
    Deck tableCards = allCards.popCards(5);
    std::size_t numPlayers = st.range(0);
    std::size_t numSimulations = st.range(1);
    BS::thread_pool<BS::tp::none> threadPool(std::thread::hardware_concurrency());
    for (auto _ : st)
    {
        double probability = probabilityOfWinning(playerCards, tableCards, numSimulations, numPlayers, threadPool);
        benchmark::DoNotOptimize(probability);
    }
}
BENCHMARK(BM_ProbabilityOfWinningParallel)->Ranges({{2, 8}, {10'000, 1'000'000}})->Unit(benchmark::kMillisecond);

static void BM_ProbabilityOfWinningParallelScaling(benchmark::State &st)
{
    omp::XoroShiro128Plus rng(42);
    Deck deck = Deck::createFullDeck();
    Deck allCards = deck.popRandomCards(rng, 7);
    Deck playerCards = allCards.popCards(2);
    Deck tableCards = allCards.popCards(5);
    std::size_t numThreads = st.range(0);
    std::size_t numSimulations = 100'000;
    std::size_t numPlayers = 6;
    BS::thread_pool<BS::tp::none> threadPool(numThreads);
    for (auto _ : st)
    {
        double probability = probabilityOfWinning(playerCards, tableCards, numSimulations, numPlayers, threadPool);
        benchmark::DoNotOptimize(probability);
    }
    st.SetItemsProcessed(st.iterations() * numSimulations);
}
BENCHMARK(BM_ProbabilityOfWinningParallelScaling)->DenseRange(1, 16, 1)->Unit(benchmark::kMillisecond);

// ============================================================================
// Throughput Benchmarks
// ============================================================================

static void BM_ClassificationThroughput(benchmark::State &state)
{
    omp::XoroShiro128Plus rng(42);
    constexpr std::size_t batchSize = 1000;
    std::vector<Deck> hands;
    hands.reserve(batchSize);
    for (std::size_t i = 0; i < batchSize; ++i)
    {
        Deck deck = Deck::createFullDeck();
        hands.push_back(deck.popRandomCards(rng, 7));
    }

    for (auto _ : state)
    {
        for (const auto &hand : hands)
        {
            ClassificationResult result = Hand::classify(hand);
            benchmark::DoNotOptimize(result);
        }
    }
    state.SetItemsProcessed(state.iterations() * batchSize);
}
BENCHMARK(BM_ClassificationThroughput);

static void BM_SimulationThroughput(benchmark::State &state)
{
    omp::XoroShiro128Plus rng(state.thread_index() + state.iterations());
    Deck playerCards = Deck::parseHand("As Ah");
    Deck tableCards = Deck::emptyDeck();
    Deck deck = Deck::createFullDeck();
    deck.removeCards(playerCards);
    std::size_t numPlayers = 6;
    std::size_t simCount = 0;

    for (auto _ : state)
    {
        bool result = playerWinsRandomGame(rng, playerCards, tableCards, deck, numPlayers);
        benchmark::DoNotOptimize(result);
        ++simCount;
    }
    state.SetItemsProcessed(simCount);
}
BENCHMARK(BM_SimulationThroughput);

BENCHMARK_MAIN();