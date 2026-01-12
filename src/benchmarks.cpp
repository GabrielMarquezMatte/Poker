#include <benchmark/benchmark.h>
#include "../include/game.hpp"

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

static void BM_PlayerWinsRandomGame(benchmark::State& st)
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

static void BM_ProbabilityOfWinningSequential(benchmark::State& st)
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

BENCHMARK_MAIN();