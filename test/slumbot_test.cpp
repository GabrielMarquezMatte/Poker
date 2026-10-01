#include "../include/cfr/blueprint.hpp"
#include "../include/cfr/slumbot.hpp"
#include <gtest/gtest.h>
#include <numeric>

TEST(Slumbot, ParsesActionStrings)
{
    const auto actions = parseSlumbotActions("b200c/kb400c/kk/b1000f");
    ASSERT_TRUE(actions.has_value());
    ASSERT_EQ(actions->size(), 9u);
    EXPECT_EQ((*actions)[0].type, 'b');
    EXPECT_EQ((*actions)[0].size, 200u);
    EXPECT_EQ((*actions)[3].street, 1);
    EXPECT_EQ((*actions)[3].size, 400u);
    EXPECT_EQ((*actions)[8].street, 3);
    EXPECT_EQ((*actions)[8].type, 'f');
    EXPECT_TRUE(parseSlumbotActions("b20000c///").has_value());
    EXPECT_FALSE(parseSlumbotActions("bx").has_value());
    EXPECT_FALSE(parseSlumbotActions("r200").has_value());
}

TEST(Slumbot, LegalBetsFollowItsRules)
{
    EXPECT_EQ(slumbotLegalBet(500, 250, 150, 250), 500u);     // enough
    EXPECT_EQ(slumbotLegalBet(380, 250, 150, 250), 400u);     // raised to the last increment
    EXPECT_EQ(slumbotLegalBet(50, 0, 0, 300), 100u);          // opening bets are at least a big blind
    EXPECT_EQ(slumbotLegalBet(20000, 0, 0, 300), 19700u);     // at most all-in
    EXPECT_EQ(slumbotLegalBet(100, 19800, 400, 19900), 19900u); // a short all-in is always allowed
}

// The copy answers like the blueprint on preflop keys, and covers every preflop node and hand class.
TEST(Slumbot, PreflopStrategyCopiesTheBlueprint)
{
    auto blueprint = std::make_unique<Mccfr<Hunl<Hunl100bbConfig>>>(1 << 20);
    CfrRng rng{3};
    blueprint->train(300, rng);
    const PreflopStrategy<Hunl<Hunl100bbConfig>> preflop(*blueprint);
    std::array<std::uint64_t, 4> nodes{};
    countPublicNodes<Hunl<Hunl100bbConfig>>(Hunl<Hunl100bbConfig>::initial(), nodes, rng);
    EXPECT_EQ(preflop.size(), nodes[0] * 169);
    using G = Hunl<Hunl100bbConfig>;
    const auto s = G::apply(G::initial(), 3); // a preflop raise
    for (std::uint16_t bucket = 0; bucket < 169; ++bucket)
    {
        const auto key = G::infosetKeyWithBucket(s, bucket);
        EXPECT_EQ(preflop.averageStrategy(key, G::numActions(s)), blueprint->averageStrategy(key, G::numActions(s)));
    }
}

namespace
{
// Slumbot's rules (its sample_api.py): tracks a hand's action string and rejects illegal increments.
// pos 1 is the small blind (first preflop), 0 the big blind (first postflop).
struct Referee
{
    std::string action;
    int street = 0, pos = 1;
    std::uint32_t streetBetTo = 100, totalBetTo = 100, lastBetSize = 50;
    bool endsStreet = false, done = false;

    bool apply(const std::string &incr)
    {
        const char c = incr.at(0);
        bool slash = false;
        if (c == 'f')
        {
            if (lastBetSize == 0)
            {
                return false;
            }
            done = true;
        }
        else if (c == 'k' || c == 'c')
        {
            if ((c == 'k') != (lastBetSize == 0))
            {
                return false;
            }
            if (c == 'c' && totalBetTo == 20000)
            {
                done = true; // all-in called: the board runs out
            }
            else if (endsStreet)
            {
                done = street == 3;
                slash = !done;
                street += done ? 0 : 1;
                pos = 0;
                streetBetTo = 0;
                endsStreet = false;
            }
            else
            {
                pos = 1 - pos;
                endsStreet = true;
            }
            lastBetSize = 0;
        }
        else if (c == 'b')
        {
            const auto to = static_cast<std::uint32_t>(std::stoul(incr.substr(1)));
            if (to <= streetBetTo)
            {
                return false;
            }
            const std::uint32_t size = to - streetBetTo, remaining = 20000 - totalBetTo;
            const std::uint32_t minimum = std::min(std::max<std::uint32_t>(lastBetSize, 100), remaining);
            if (size < minimum || size > remaining)
            {
                return false;
            }
            lastBetSize = size;
            streetBetTo = to;
            totalBetTo += size;
            pos = 1 - pos;
            endsStreet = true;
        }
        else
        {
            return false;
        }
        action += incr;
        action += slash ? "/" : "";
        return true;
    }

    // A random legal action, as a stand-in for Slumbot.
    std::string random(CfrRng &rng) const
    {
        const std::uint32_t remaining = 20000 - totalBetTo;
        const std::uint64_t r = rng() % 10;
        if (lastBetSize > 0 && r == 0)
        {
            return "f";
        }
        if (remaining == 0 || r < 5)
        {
            return lastBetSize > 0 ? "c" : "k";
        }
        const std::uint32_t minimum = std::min(std::max<std::uint32_t>(lastBetSize, 100), remaining);
        const std::uint32_t size = minimum + static_cast<std::uint32_t>(rng() % (remaining - minimum + 1)) / (r == 9 ? 1 : 8);
        return "b" + std::to_string(streetBetTo + size);
    }
};
} // namespace

// Against random legal play (with long raise wars), the bot always answers when it is to act, its
// increments are legal and Slumbot's range never empties.
TEST(Slumbot, PlaysLegallyAgainstRandomActions)
{
    const auto blueprint = std::make_unique<Mccfr<Blueprint200>>(1 << 16); // untrained: uniform preflop
    const PreflopStrategy<Blueprint200> preflop(*blueprint);
    SlumbotBot<Blueprint200Config> bot(preflop, SlumbotResolves{1, 1, 2, 2}, 7);
    CfrRng rng{11};
    std::array<int, 4> decisions{};
    for (int hand = 0; hand < 40; ++hand)
    {
        Deck deck = Deck::createFullDeck();
        const std::uint64_t hole = deck.popPair(rng).getMask();
        deck.popPair(rng); // Slumbot's
        std::vector<std::uint64_t> board;
        for (int i = 0; i < 5; ++i)
        {
            board.push_back(deck.popRandomCards(rng, 1).getMask());
        }
        const int clientPos = static_cast<int>(rng() % 2);
        bot.newHand(clientPos == 0 ? 1 : 0, hole);
        Referee referee;
        while (!referee.done)
        {
            if (referee.pos == clientPos)
            {
                const auto actions = parseSlumbotActions(referee.action);
                ASSERT_TRUE(actions.has_value()) << referee.action;
                const std::vector<std::uint64_t> seen(board.begin(), board.begin() + std::array<int, 4>{0, 3, 4, 5}[referee.street]);
                const auto incr = bot.update(*actions, seen);
                ASSERT_TRUE(incr.has_value()) << "hand " << hand << ": no answer after " << referee.action;
                ASSERT_TRUE(referee.apply(*incr)) << "hand " << hand << ": illegal " << *incr << " after " << referee.action;
                ++decisions[static_cast<std::size_t>(bot.state().street)];
                const auto &range = bot.range();
                ASSERT_GT(std::accumulate(range.begin(), range.end(), 0.0), 0.0) << "hand " << hand << ": Slumbot's range emptied after " << referee.action;
            }
            else
            {
                ASSERT_TRUE(referee.apply(referee.random(rng)));
            }
        }
    }
    for (const int count : decisions)
    {
        EXPECT_GT(count, 0); // every street was played
    }
}
