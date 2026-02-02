#include <gtest/gtest.h>
#include "../include/hand.hpp"
#include "../include/deck.hpp"
#include "../include/card.hpp"
#include "../include/game.hpp"

TEST(DeckTest, ParsingHand)
{
    static constexpr Deck parsedDeck = Deck::parseHand("2H 3D 4S 5C 6H");
    static constexpr Deck expectedDeck = Deck::createDeck({Card(Suit::Hearts, Rank::Two), Card(Suit::Diamonds, Rank::Three),
                                                           Card(Suit::Spades, Rank::Four), Card(Suit::Clubs, Rank::Five),
                                                           Card(Suit::Hearts, Rank::Six)});
    static_assert(parsedDeck.size() == 5, "Expected deck size of 5 after parsing");
    static_assert(parsedDeck.getMask() == expectedDeck.getMask(), "Parsed deck does not match expected deck");
}
TEST(DeckTest, ClassifyRoyalFlush)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Hearts, Rank::Queen),
                                                   Card(Suit::Hearts, Rank::Jack),
                                                   Card(Suit::Hearts, Rank::Ten)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::RoyalFlush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten), "Expected Royal Flush classification");
}

TEST(DeckTest, ClassifyStraightFlush)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Nine),
                                                   Card(Suit::Hearts, Rank::Eight),
                                                   Card(Suit::Hearts, Rank::Seven),
                                                   Card(Suit::Hearts, Rank::Six),
                                                   Card(Suit::Hearts, Rank::Five)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::StraightFlush, Rank::Nine), "Expected Straight Flush classification");
}
TEST(DeckTest, ClassifyStraight)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Hearts, Rank::Queen),
                                                   Card(Suit::Hearts, Rank::Jack),
                                                   Card(Suit::Diamonds, Rank::Ten)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::Straight, Rank::Ace), "Expected Flush classification");
}
TEST(DeckTest, ClassifyFlush)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Hearts, Rank::Queen),
                                                   Card(Suit::Hearts, Rank::Jack),
                                                   Card(Suit::Hearts, Rank::Two)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::Flush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Two), "Expected Flush classification");
}
TEST(DeckTest, ClassifyFullHouse)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Diamonds, Rank::King)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::FullHouse, Rank::Ace | Rank::King), "Expected Full House classification");
}
TEST(DeckTest, ClassifyFourOfAKind)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::Ace),
                                                   Card(Suit::Spades, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::FourOfAKind, Rank::Ace | Rank::King), "Expected Four of a Kind classification");
}
TEST(DeckTest, ClassifyThreeOfAKind)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Diamonds, Rank::Queen)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::ThreeOfAKind, Rank::Ace | Rank::King | Rank::Queen), "Expected Three of a Kind classification");
}
TEST(DeckTest, ClassifyTwoPair)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::King),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Diamonds, Rank::Queen)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::TwoPair, Rank::Ace | Rank::King | Rank::Queen), "Expected Two Pair classification");
}
TEST(DeckTest, ClassifyOnePair)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::King),
                                                   Card(Suit::Hearts, Rank::Queen),
                                                   Card(Suit::Diamonds, Rank::Jack)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result.getClassification() == Classification::Pair, "Expected Pair classification");
    static constexpr Deck lowerPairDeck = Deck::createDeck({Card(Suit::Hearts, Rank::King),
                                                            Card(Suit::Diamonds, Rank::King),
                                                            Card(Suit::Clubs, Rank::Ace),
                                                            Card(Suit::Hearts, Rank::Queen),
                                                            Card(Suit::Diamonds, Rank::Jack)});
    static constexpr ClassificationResult lowerResult = Hand::classify(lowerPairDeck);
    static_assert(result > lowerResult, "Pair of Aces should beat Pair of Kings");
}
TEST(DeckTest, ClassifyHighCard)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Two),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::Four),
                                                   Card(Suit::Hearts, Rank::Seven),
                                                   Card(Suit::Diamonds, Rank::Six)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::HighCard, Rank::Ace | Rank::Seven | Rank::Six | Rank::Four | Rank::Two), "Expected High Card classification");
}

//
// 1) Wheel (A-2-3-4-5) Straight / Straight‐Flush
//
TEST(ClassificationTest, ClassifyWheelStraight)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Clubs, Rank::Five),
                                                   Card(Suit::Hearts, Rank::Four),
                                                   Card(Suit::Diamonds, Rank::Three),
                                                   Card(Suit::Spades, Rank::Two),
                                                   Card(Suit::Clubs, Rank::Ace)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::Straight, Rank::Five), "Expected Low-Ace Straight");
}

TEST(ClassificationTest, ClassifyWheelStraightFlush)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Five),
                                                   Card(Suit::Hearts, Rank::Four),
                                                   Card(Suit::Hearts, Rank::Three),
                                                   Card(Suit::Hearts, Rank::Two),
                                                   Card(Suit::Hearts, Rank::Ace)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::StraightFlush, Rank::Five), "Expected Low-Ace Straight Flush");
}
TEST(ClassificationTest, StraightComparisonByTopCard)
{
    static constexpr ClassificationResult straight5 = ClassificationResult(Classification::Straight, Rank::Five);
    static constexpr ClassificationResult straight9 = ClassificationResult(Classification::Straight, Rank::Nine);
    static_assert(straight9 > straight5, "9-high Straight should beat 5-high Wheel");
}
TEST(ClassificationTest, classifyPlayerBestOfSeven)
{
    // hole + 5-card board that makes a Flush, but board alone is only a Pair
    static constexpr Deck hole = Deck::createDeck({Card(Suit::Spades, Rank::King),
                                                   Card(Suit::Spades, Rank::Queen)});
    static constexpr Deck board = Deck::createDeck({Card(Suit::Spades, Rank::Two),
                                                    Card(Suit::Spades, Rank::Three),
                                                    Card(Suit::Spades, Rank::Four),
                                                    Card(Suit::Hearts, Rank::Ace),
                                                    Card(Suit::Hearts, Rank::King)});
    // best 5 of 7 is a Spade flush King-high
    static constexpr ClassificationResult res = Hand::classify(Deck::createDeck({hole, board}));
    static constexpr ClassificationResult expected = ClassificationResult(Classification::Flush, Rank::King | Rank::Queen | Rank::Four | Rank::Three | Rank::Two);
    static_assert(res == expected, "Expected best hand to be a King-high Flush");
}
TEST(ClassificationTest, TestTwoPairBreakdown)
{
    static constexpr Deck hand = Deck::parseHand("qd tc");
    static constexpr Deck board = Deck::parseHand("9d 9c 8h Kh Qh");
    static constexpr Deck firstOpponent = Deck::parseHand("kd 2c");
    static constexpr Deck secondOpponent = Deck::parseHand("qc 2d");
    static constexpr Deck thirdOpponent = Deck::parseHand("ac qc");
    static constexpr Deck fourthOpponent = Deck::parseHand("jd tc");
    static constexpr Deck fifthOpponent = Deck::parseHand("8c 8d");
    static constexpr Deck sixthOpponent = Deck::parseHand("ad ac");
    static constexpr ClassificationResult mainClassification = Hand::classify(Deck::createDeck({hand, board}));
    static constexpr ClassificationResult res1 = Hand::classify(Deck::createDeck({firstOpponent, board}));
    static constexpr ClassificationResult res2 = Hand::classify(Deck::createDeck({secondOpponent, board}));
    static constexpr ClassificationResult res3 = Hand::classify(Deck::createDeck({thirdOpponent, board}));
    static constexpr ClassificationResult res4 = Hand::classify(Deck::createDeck({fourthOpponent, board}));
    static constexpr ClassificationResult res5 = Hand::classify(Deck::createDeck({fifthOpponent, board}));
    static constexpr ClassificationResult res6 = Hand::classify(Deck::createDeck({sixthOpponent, board}));
    static_assert(mainClassification == res1, "Expected tie with first opponent");
    static_assert(mainClassification == res2, "Expected tie with second opponent");
    static_assert(mainClassification < res3, "Expected tie with third opponent");
    static_assert(mainClassification < res4, "Expected tie with fourth opponent");
    static_assert(mainClassification < res5, "Expected tie with fifth opponent");
    static_assert(mainClassification < res6, "Expected tie with sixth opponent");
}
TEST(DeckTest, ParseInvalidFormat)
{
    static_assert(Deck::parseHand("ZZ XX 11").size() == 0, "Expected empty deck from invalid hand format");
    static_assert(Deck::parseHand("2H3D").size() == 0, "Expected empty deck from invalid hand format");
}
TEST(DeckTest, ParseDuplicateCards)
{
    static_assert(Deck::parseHand("AH AH 2D 3C 4S").size() == 4, "Expected 4 unique cards in deck");
}
TEST(ClassificationTest, EqualityInequality)
{
    static constexpr auto a = ClassificationResult(Classification::Pair, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack);
    static constexpr auto b = ClassificationResult(Classification::Pair, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack);
    static constexpr auto c = ClassificationResult(Classification::Pair, Rank::Ace | Rank::King | Rank::Queen | Rank::Ten);
    static_assert(a == b, "Expected equality of identical classification results");
    static_assert(a != c, "Expected inequality of different classification results");
    static_assert(b != c, "Expected inequality of different classification results");
}
TEST(ClassificationTest, OnePairKickerComparison)
{
    static constexpr ClassificationResult pairHighJack = ClassificationResult(Classification::Pair, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack);
    static constexpr ClassificationResult pairHighTen = ClassificationResult(Classification::Pair, Rank::Ace | Rank::King | Rank::Queen | Rank::Ten);
    static_assert(pairHighJack > pairHighTen, "Pair with Jack kicker should beat Pair with Ten kicker");
}

// 7) Two-Pair kicker comparison: AA QQ K beats AA JJ K
TEST(ClassificationTest, TwoPairKickerComparison)
{
    static constexpr ClassificationResult twoPairQHigh = ClassificationResult(Classification::TwoPair, Rank::Ace | Rank::Queen | Rank::King);
    static constexpr ClassificationResult twoPairJHigh = ClassificationResult(Classification::TwoPair, Rank::Ace | Rank::Jack | Rank::King);
    static_assert(twoPairQHigh > twoPairJHigh, "Two Pair with Queen kicker should beat Two Pair with Jack kicker");
}

// 8) Flush kicker comparison: A K Q J 10 ♠ beats A K Q J 9 ♠
TEST(ClassificationTest, FlushKickerComparison)
{
    static constexpr ClassificationResult flushWithTen = ClassificationResult(Classification::Flush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten);
    static constexpr ClassificationResult flushWithNine = ClassificationResult(Classification::Flush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Nine);
    static_assert(flushWithTen > flushWithNine, "Flush with Ten kicker should beat Flush with Nine kicker");
}

// 9) Streaming operator: verify human-readable output
TEST(StreamingTest, ClassificationResultToString)
{
    static constexpr ClassificationResult cr(Classification::Pair, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack);
    std::ostringstream oss;
    oss << cr;
    EXPECT_EQ(oss.str(), "Pair: J Q K A");
}

// 11) checkUniqueCards
TEST(GameTest, CheckUniqueCards)
{
    static constexpr Deck p = Deck::createDeck({Card(Suit::Hearts, Rank::Ace)});
    static constexpr Deck t1 = Deck::createDeck({Card(Suit::Hearts, Rank::Ace)});
    static constexpr Deck t2 = Deck::createDeck({Card(Suit::Hearts, Rank::King)});
    static constexpr auto checkUniqueCards = [](const Deck &playerCards, const Deck &tableCards)
    {
        return (playerCards.getMask() & tableCards.getMask()) == 0;
    };
    static_assert(checkUniqueCards(p, t1) == false, "Expected duplicate cards in hand and table");
    static_assert(checkUniqueCards(p, t2) == true, "Expected no duplicate cards in hand and table");
}

TEST(GameCompareTest, PlayerWinsBeatsLowerOpponent)
{
    static constexpr Deck player = Deck::createDeck({Card(Suit::Hearts, Rank::Ace), Card(Suit::Clubs, Rank::Ace)});
    static constexpr Deck opp = Deck::createDeck({Card(Suit::Hearts, Rank::King), Card(Suit::Clubs, Rank::King)});
    static constexpr Deck board = Deck::createDeck({Card(Suit::Spades, Rank::Two), Card(Suit::Diamonds, Rank::Three),
                                                    Card(Suit::Spades, Rank::Four), Card(Suit::Diamonds, Rank::Five),
                                                    Card(Suit::Clubs, Rank::Nine)});
    static constexpr std::array<Deck, 1> opps{opp};
    static constexpr auto res = compareHands(player, board, opps);
    static_assert(res == GameResult::Win);
}

TEST(GameCompareTest, PlayerLosesToHigherOpponent)
{
    static constexpr Deck player = Deck::createDeck({Card(Suit::Hearts, Rank::King), Card(Suit::Clubs, Rank::King)});
    static constexpr Deck opp = Deck::createDeck({Card(Suit::Hearts, Rank::Ace), Card(Suit::Clubs, Rank::Ace)});
    static constexpr Deck board = Deck::createDeck({Card(Suit::Spades, Rank::Two), Card(Suit::Diamonds, Rank::Three),
                                                    Card(Suit::Spades, Rank::Four), Card(Suit::Diamonds, Rank::Five),
                                                    Card(Suit::Clubs, Rank::Nine)});
    static constexpr std::array<Deck, 1> opps{opp};
    static constexpr auto res = compareHands(player, board, opps);
    static_assert(res == GameResult::Lose);
}

TEST(GameCompareTest, PlayerTiesWithSameBestHand)
{
    // Both have same hole cards ranks but different suits; board makes the same two pair from board + kickers
    static constexpr Deck player = Deck::createDeck({Card(Suit::Hearts, Rank::Ace), Card(Suit::Spades, Rank::King)});
    static constexpr Deck opp = Deck::createDeck({Card(Suit::Diamonds, Rank::Ace), Card(Suit::Clubs, Rank::King)});
    // Board pairs A and K with equal kickers such that both best 5 are identical
    static constexpr Deck board = Deck::createDeck({Card(Suit::Hearts, Rank::Ace), Card(Suit::Diamonds, Rank::King),
                                                    Card(Suit::Clubs, Rank::Queen), Card(Suit::Spades, Rank::Jack),
                                                    Card(Suit::Hearts, Rank::Ten)});
    static constexpr std::array<Deck, 1> opps{opp};
    static constexpr auto res = compareHands(player, board, opps);
    static_assert(res == GameResult::Tie);
}

TEST(DeckTest, RemoveAndAddCards)
{
    static constexpr Deck full = Deck::createFullDeck();
    static_assert(full.size() == 52, "Expected full deck size of 52");
    static constexpr Deck firstTestDeck = []
    {
        Deck deck = Deck::createFullDeck();
        deck.removeCard(Card(Suit::Spades, Rank::Ten));
        return deck;
    }();
    static_assert(firstTestDeck.size() == 51, "Expected deck size of 51 after removing one card");
    static constexpr Deck secondTestDeck = []
    {
        Deck deck = Deck::createFullDeck();
        deck.removeCard(Card(Suit::Spades, Rank::Ten));
        deck.removeCard(Card(Suit::Clubs, Rank::Five));
        return deck;
    }();
    static_assert(secondTestDeck.size() == 50, "Expected deck size of 50 after removing two cards");
    static constexpr Deck thirdTestDeck = []
    {
        Deck deck = Deck::createFullDeck();
        deck.removeCard(Card(Suit::Spades, Rank::Ten));
        deck.removeCard(Card(Suit::Clubs, Rank::Five));
        deck.addCard(Card(Suit::Spades, Rank::Ten));
        return deck;
    }();
    static_assert(thirdTestDeck.size() == 51, "Expected deck size of 51 after removing two cards and adding one card");
    static constexpr Deck fourthTestDeck = []
    {
        Deck deck = Deck::createFullDeck();
        deck.removeCard(Card(Suit::Spades, Rank::Ten));
        deck.removeCard(Card(Suit::Clubs, Rank::Five));
        deck.addCard(Card(Suit::Spades, Rank::Ten));
        deck.addCard(Card(Suit::Clubs, Rank::Five));
        return deck;
    }();
    static_assert(fourthTestDeck.size() == 52, "Expected deck size of 52 after removing two cards and adding two cards");
}

TEST(DeckTest, PopRandomCards)
{
    static constexpr Deck fullDeck = Deck::createFullDeck();
    static constexpr Deck emptyDeck = Deck::emptyDeck();
    {
        static constexpr Deck popped = []
        {
            Deck deck = fullDeck;
            omp::XoroShiro128Plus rng{124};
            return deck.popRandomCards(rng, 5);
        }();
        static constexpr std::size_t remaining = []
        {
            Deck deck = fullDeck;
            omp::XoroShiro128Plus rng{124};
            deck.popRandomCards(rng, 5);
            return deck.size();
        }();
        static constexpr Deck expectedDeck = Deck::parseHand("8d tc Jc 2s Js");
        static_assert(popped.size() == 5, "Expected 5 cards to be popped");
        static_assert(remaining == 47, "Expected 47 cards to remain in the deck");
        static_assert(popped == expectedDeck);
    }
    {
        static constexpr Deck popped = []
        {
            Deck deck = emptyDeck;
            omp::XoroShiro128Plus rng{123};
            return deck.popRandomCards(rng, 5);
        }();
        static constexpr std::size_t remaining = []
        {
            Deck deck = emptyDeck;
            omp::XoroShiro128Plus rng{123};
            deck.popRandomCards(rng, 5);
            return deck.size();
        }();
        static_assert(popped.size() == 0, "Expected 0 cards to be popped");
        static_assert(remaining == 0, "Expected 0 cards to remain in the deck");
    }
}

TEST(BugReproduction, StaticAssert_WheelStraightWithKickers)
{
    // Bug: Wheel Straight (A-2-3-4-5) com Kickers
    // Se a lógica falhar, o compilador emitirá erro aqui.
    static constexpr Deck hand = Deck::parseHand("2c 3d 4h 5s ac 9d kd");
    static constexpr ClassificationResult result = Hand::classify(hand);
    
    static_assert(result == ClassificationResult(Classification::Straight, Rank::Five), 
        "FALHA DE COMPILACAO: Nao detetou Wheel Straight (A-5) quando existem kickers (9, K).");
}

TEST(BugReproduction, StaticAssert_RoyalFlushWithExtraSuitedCard)
{
    // Bug: Royal Flush com carta extra do mesmo naipe (9s)
    // Se a lógica falhar (classificar como Straight Flush), a compilação para.
    static constexpr Deck hand = Deck::parseHand("as ks qs js ts 9s 2c");
    static constexpr ClassificationResult result = Hand::classify(hand);
    
    // Verifica se é estritamente Royal Flush
    static_assert(result.getClassification() == Classification::RoyalFlush, 
        "FALHA DE COMPILACAO: Classificou incorretamente Royal Flush 'sujo' como Straight Flush.");
}

TEST(EdgeCases, StaticAssert_Complex7CardHands)
{
    // Caso: 3 Pares no Board -> Deve escolher os 2 maiores (AA, KK) com kicker Q
    static constexpr Deck threePairHand = Deck::parseHand("as ac ks kc qs qc 2h");
    static_assert(Hand::classify(threePairHand) == ClassificationResult(Classification::TwoPair, Rank::Ace | Rank::King | Rank::Queen),
        "FALHA DE COMPILACAO: Erro ao escolher os melhores 2 pares de 3 possiveis.");

    // Caso: Wheel Straight Flush com Kicker do mesmo naipe
    static constexpr Deck wheelSFHand = Deck::parseHand("5h 4h 3h 2h ah 9h kd");
    static_assert(Hand::classify(wheelSFHand) == ClassificationResult(Classification::StraightFlush, Rank::Five),
        "FALHA DE COMPILACAO: Erro ao detetar Wheel Straight Flush com kicker suited.");

    // Caso: Quads vs Full House (Prioridade)
    // Board: 5555, Mão: AA. Resultado deve ser Quads (5555 A), nunca Full House.
    static constexpr Deck quadsHand = Deck::parseHand("5c 5d 5h 5s as ac 2d");
    static_assert(Hand::classify(quadsHand).getClassification() == Classification::FourOfAKind,
        "FALHA DE COMPILACAO: Full House foi priorizado indevidamente sobre Four of a Kind.");
}

TEST(BugReproduction, PairEquality_DifferentPairs)
{
    static constexpr Deck board = Deck::parseHand("Ah Kd Qc Js 2h");
    static constexpr Deck p1 = Deck::parseHand("As 3s");
    static constexpr Deck p2 = Deck::parseHand("Ks 3d");
    static constexpr ClassificationResult r1 = Hand::classify(Deck::createDeck({p1, board}));
    static constexpr ClassificationResult r2 = Hand::classify(Deck::createDeck({p2, board}));

    // Now that we fix the implementation, r1 > r2 because Pair of Aces (Main=A) >
    // Pair of Kings (Main=K).
    static_assert(r1 > r2, "Pair of Aces should beat Pair of Kings despite having same kickers/ranks.");
}