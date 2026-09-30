#include "../include/cfr/hand_indexer.hpp"
#include "../include/random.hpp"
#include <gtest/gtest.h>

static HandIndexer makeIndexer(std::initializer_list<std::uint8_t> sizes)
{
    return HandIndexer(std::span<const std::uint8_t>(sizes.begin(), sizes.size()));
}

static std::uint64_t permuteSuits(std::uint64_t mask, const std::array<int, 4> &perm)
{
    std::uint64_t out = 0;
    for (std::size_t s = 0; s < 4; ++s)
    {
        out |= ((mask >> (13 * s)) & 0x1FFF) << (13 * perm[s]);
    }
    return out;
}

// Random disjoint rounds of the given sizes.
static HandIndexer::Rounds randomRounds(std::initializer_list<std::uint8_t> sizes, omp::XoroShiro128Plus &rng)
{
    HandIndexer::Rounds rounds{};
    std::uint64_t used = 0;
    std::size_t r = 0;
    for (const std::uint8_t size : sizes)
    {
        for (std::uint8_t c = 0; c < size;)
        {
            const std::uint64_t card = 1ull << (rng() % 52);
            if ((used & card) == 0)
            {
                used |= card;
                rounds[r] |= card;
                ++c;
            }
        }
        ++r;
    }
    return rounds;
}

TEST(HandIndexer, SizesMatchKnownIsomorphismCounts)
{
    EXPECT_EQ(makeIndexer({2}).size(), 169u);
    EXPECT_EQ(makeIndexer({2, 3}).size(), 1'286'792u);
    EXPECT_EQ(makeIndexer({2, 3, 1}).size(), 55'190'538u);
    EXPECT_EQ(makeIndexer({2, 3, 1, 1}).size(), 2'428'287'420u);
    // Board as one round: card order on the board is irrelevant to hand strength.
    EXPECT_EQ(makeIndexer({2, 4}).size(), 13'960'050u);
    EXPECT_EQ(makeIndexer({2, 5}).size(), 123'156'254u);
}

TEST(HandIndexer, FlopRoundTripsEveryIndex)
{
    const auto indexer = makeIndexer({2, 3});
    for (std::uint64_t i = 0; i < indexer.size(); ++i)
    {
        const auto rounds = indexer.unindex(i);
        ASSERT_EQ(std::popcount(rounds[0]), 2) << i;
        ASSERT_EQ(std::popcount(rounds[1]), 3) << i;
        ASSERT_EQ(rounds[0] & rounds[1], 0u) << i;
        ASSERT_EQ(indexer.index(rounds), i);
    }
}

TEST(HandIndexer, RiverRoundTripsSampledIndices)
{
    const auto indexer = makeIndexer({2, 3, 1, 1});
    omp::XoroShiro128Plus rng{9};
    for (int n = 0; n < 200'000; ++n)
    {
        const std::uint64_t i = rng() % indexer.size();
        ASSERT_EQ(indexer.index(indexer.unindex(i)), i);
    }
}

TEST(HandIndexer, InvariantUnderSuitPermutation)
{
    const auto indexer = makeIndexer({2, 3, 1, 1});
    omp::XoroShiro128Plus rng{3};
    std::array<int, 4> perm{0, 1, 2, 3};
    for (int n = 0; n < 100'000; ++n)
    {
        const auto rounds = randomRounds({2, 3, 1, 1}, rng);
        const std::uint64_t idx = indexer.index(rounds);
        ASSERT_LT(idx, indexer.size());
        std::next_permutation(perm.begin(), perm.end());
        HandIndexer::Rounds permuted{};
        for (std::size_t r = 0; r < 4; ++r)
        {
            permuted[r] = permuteSuits(rounds[r], perm);
        }
        ASSERT_EQ(indexer.index(permuted), idx);
    }
}

TEST(HandIndexer, DistinguishesRoundsAndSuitedness)
{
    const auto flop = makeIndexer({2, 3});
    const std::uint64_t aceSpades = 1ull << 12, aceHearts = 1ull << 25, kingSpades = 1ull << 11, kingHearts = 1ull << 24;
    // Suited vs offsuit hole cards.
    EXPECT_NE(flop.index({aceSpades | kingSpades, 7}), flop.index({aceSpades | kingHearts, 7}));
    // Same 5 cards, different split between hole and board.
    EXPECT_NE(flop.index({aceSpades | aceHearts, kingSpades | 3}), flop.index({aceSpades | kingSpades, aceHearts | 3}));
}
