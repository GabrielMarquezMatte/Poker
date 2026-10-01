#include "../include/cfr/blueprint.hpp"
#include "../include/cfr/slumbot.hpp"
#include "../include/cfr/slumbot_referee.hpp"
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

// The 200bb blueprint keeps the average strategy for exactly the infosets the bot reads.
TEST(Slumbot, BlueprintAveragesWhatTheBotReads)
{
    static_assert(PartlyAveraged<Blueprint200> && !PartlyAveraged<Blueprint>);
    const auto blueprint = std::make_unique<Mccfr<Blueprint200>>(1 << 10);
    EXPECT_EQ(Blueprint200::averagedKeys().size(), PreflopStrategy<Blueprint200>(*blueprint).size());
}

namespace
{
// A random legal action, as a stand-in for Slumbot.
std::string randomAction(const SlumbotReferee &referee, CfrRng &rng)
{
    const std::uint32_t remaining = 20000 - referee.totalBetTo;
    const std::uint64_t r = rng() % 10;
    if (referee.lastBetSize > 0 && r == 0)
    {
        return "f";
    }
    if (remaining == 0 || r < 5)
    {
        return referee.lastBetSize > 0 ? "c" : "k";
    }
    const std::uint32_t minimum = std::min(std::max<std::uint32_t>(referee.lastBetSize, 100), remaining);
    const std::uint32_t size = minimum + static_cast<std::uint32_t>(rng() % (remaining - minimum + 1)) / (r == 9 ? 1 : 8);
    return "b" + std::to_string(referee.streetBetTo + size);
}
} // namespace

TEST(SlumbotReferee, ScoresFoldsShowdownsAndAllIns)
{
    const std::array<std::uint64_t, 2> hands{Deck::parseHand("Kc Kd").getMask(), Deck::parseHand("Ac Ad").getMask()}; // pos 0, pos 1
    std::vector<std::uint64_t> order;
    for (const char *c : {"2h", "7s", "9d", "Js", "3c"})
    {
        order.push_back(Deck::parseHand(c).getMask());
    }
    const std::uint64_t board = order[0] | order[1] | order[2] | order[3] | order[4];

    SlumbotReferee fold; // small blind raises, big blind folds
    ASSERT_TRUE(fold.apply("b300"));
    ASSERT_TRUE(fold.apply("f"));
    EXPECT_EQ(fold.winnings(1, hands, board, order, false), 100.0);
    EXPECT_EQ(fold.winnings(0, hands, board, order, false), -100.0);

    SlumbotReferee showdown; // limp, check down
    for (const char *incr : {"c", "k", "k", "k", "k", "k", "b200", "c"})
    {
        ASSERT_TRUE(showdown.apply(incr)) << incr;
    }
    EXPECT_TRUE(showdown.done);
    EXPECT_EQ(showdown.action, "ck/kk/kk/b200c");
    EXPECT_EQ(showdown.winnings(1, hands, board, order, false), 300.0); // aces win 100 + 200

    SlumbotReferee allIn; // all-in preflop, called
    ASSERT_TRUE(allIn.apply("b20000"));
    ASSERT_TRUE(allIn.apply("c"));
    EXPECT_EQ(allIn.allInStreet, 0);
    EXPECT_EQ(allIn.winnings(1, hands, board, order, false), 20000.0);
    const double equity = SlumbotReferee::equity(hands[1], hands[0], 0);
    EXPECT_NEAR(equity, 0.826, 0.003); // aces against kings of the same suits (0.828 +/- 0.005 by sampling)
    EXPECT_NEAR(allIn.winnings(1, hands, board, order, true), (2.0 * equity - 1.0) * 20000.0, 1e-6);
    EXPECT_FALSE(SlumbotReferee{}.apply("k")); // the small blind faces the big blind
}


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
        SlumbotReferee referee;
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
                ASSERT_TRUE(referee.apply(randomAction(referee, rng)));
            }
        }
    }
    for (const int count : decisions)
    {
        EXPECT_GT(count, 0); // every street was played
    }
}

// Resolves allow a fourth raise per postflop street, all-in only; preflop keeps the blueprint's raises.
TEST(Slumbot, FourthPostflopRaiseIsAllIn)
{
    using R = Hunl<PostflopRaises<Blueprint200Config>>;
    auto s = R::initial();
    s.dealt = true;
    s.street = 1;
    s.invested = {4, 4};
    s.toAct = 1;
    for (int raise = 0; raise < 3; ++raise)
    {
        const auto legal = R::legalActions(s);
        ASSERT_GT(legal.size, 3u); // sized raises and all-in
        s = R::apply(s, legal.size - 2); // the largest sized raise
    }
    const auto legal = R::legalActions(s);
    ASSERT_EQ(legal.size, 3u); // fold, call, all-in
    EXPECT_EQ(legal.to[2], Blueprint200Config::stack);
    EXPECT_EQ(R::legalActions(R::initial()).size, Blueprint200::legalActions(Blueprint200::initial()).size);
}

// Two bots with jittered bets (so each faces bets its tree lacks) play legally against each other, and
// neither one's range for the other ever empties.
TEST(Slumbot, JitteredBotsPlayEachOtherLegally)
{
    const auto blueprint = std::make_unique<Mccfr<Blueprint200>>(1 << 16);
    const PreflopStrategy<Blueprint200> preflop(*blueprint);
    const SlumbotResolves resolves{.flop = 1, .turn = 1, .river = 2, .flopSamples = 2, .allInSamples = 2, .nested = true, .jitter = 0.3};
    SlumbotBot<Blueprint200Config> small(preflop, resolves, 3), big(preflop, resolves, 4);
    CfrRng rng{13};
    std::array<int, 4> decisions{};
    for (int hand = 0; hand < 40; ++hand)
    {
        Deck deck = Deck::createFullDeck();
        const std::array<std::uint64_t, 2> hands{deck.popPair(rng).getMask(), deck.popPair(rng).getMask()};
        std::vector<std::uint64_t> board;
        for (int i = 0; i < 5; ++i)
        {
            board.push_back(deck.popRandomCards(rng, 1).getMask());
        }
        small.newHand(0, hands[1]);
        big.newHand(1, hands[0]);
        SlumbotReferee referee;
        while (!referee.done)
        {
            auto &bot = referee.pos == 1 ? small : big;
            const std::vector<std::uint64_t> seen(board.begin(), board.begin() + std::array<int, 4>{0, 3, 4, 5}[referee.street]);
            const auto incr = bot.update(*parseSlumbotActions(referee.action), seen);
            ASSERT_TRUE(incr.has_value()) << "hand " << hand << ": no answer after " << referee.action;
            ASSERT_TRUE(referee.apply(*incr)) << "hand " << hand << ": illegal " << *incr << " after " << referee.action;
            ++decisions[static_cast<std::size_t>(bot.state().street)];
            ASSERT_GT(std::accumulate(bot.range().begin(), bot.range().end(), 0.0), 0.0) << "hand " << hand << " after " << referee.action;
        }
    }
    EXPECT_GT(decisions[1], 0); // postflop, where jittered bets fall off the tree
    EXPECT_GT(decisions[2], 0);
}
