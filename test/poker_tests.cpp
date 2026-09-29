#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
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
    static_assert(result == ClassificationResult(Classification::FullHouse, Rank::Ace, Rank::King), "Expected Full House classification");
}
TEST(DeckTest, ClassifyFourOfAKind)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::Ace),
                                                   Card(Suit::Spades, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::FourOfAKind, Rank::Ace, Rank::King), "Expected Four of a Kind classification");
}
TEST(DeckTest, ClassifyThreeOfAKind)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Diamonds, Rank::Queen)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::ThreeOfAKind, Rank::Ace, Rank::King | Rank::Queen), "Expected Three of a Kind classification");
}
TEST(DeckTest, ClassifyTwoPair)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Diamonds, Rank::Ace),
                                                   Card(Suit::Clubs, Rank::King),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Diamonds, Rank::Queen)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::TwoPair, Rank::Ace | Rank::King, Rank::Queen), "Expected Two Pair classification");
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
    static_assert(result == ClassificationResult(Classification::HighCard, Rank{}, Rank::Ace | Rank::Seven | Rank::Six | Rank::Four | Rank::Two), "Expected High Card classification");
}

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
    static constexpr Deck hole = Deck::createDeck({Card(Suit::Spades, Rank::King),
                                                   Card(Suit::Spades, Rank::Queen)});
    static constexpr Deck board = Deck::createDeck({Card(Suit::Spades, Rank::Two),
                                                    Card(Suit::Spades, Rank::Three),
                                                    Card(Suit::Spades, Rank::Four),
                                                    Card(Suit::Hearts, Rank::Ace),
                                                    Card(Suit::Hearts, Rank::King)});
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
    static_assert(mainClassification < res1, "KK99Q must beat QQ99K");
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

TEST(ClassificationTest, TwoPairKickerComparison)
{
    static constexpr ClassificationResult twoPairQHigh = ClassificationResult(Classification::TwoPair, Rank::Ace | Rank::Queen, Rank::King);
    static constexpr ClassificationResult twoPairJHigh = ClassificationResult(Classification::TwoPair, Rank::Ace | Rank::Jack, Rank::King);
    static_assert(twoPairQHigh > twoPairJHigh, "Two Pair with Queen kicker should beat Two Pair with Jack kicker");
}

TEST(ClassificationTest, FlushKickerComparison)
{
    static constexpr ClassificationResult flushWithTen = ClassificationResult(Classification::Flush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten);
    static constexpr ClassificationResult flushWithNine = ClassificationResult(Classification::Flush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Nine);
    static_assert(flushWithTen > flushWithNine, "Flush with Ten kicker should beat Flush with Nine kicker");
}

TEST(StreamingTest, ClassificationResultToString)
{
    static constexpr ClassificationResult cr(Classification::Pair, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack);
    std::ostringstream oss;
    oss << cr;
    EXPECT_EQ(oss.str(), "Pair: J Q K A");
}

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
    static constexpr Deck player = Deck::createDeck({Card(Suit::Hearts, Rank::Ace), Card(Suit::Spades, Rank::King)});
    static constexpr Deck opp = Deck::createDeck({Card(Suit::Diamonds, Rank::Ace), Card(Suit::Clubs, Rank::King)});
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
    static constexpr Deck hand = Deck::parseHand("2c 3d 4h 5s ac 9d kd");
    static constexpr ClassificationResult result = Hand::classify(hand);
    
    static_assert(result == ClassificationResult(Classification::Straight, Rank::Five), 
        "FALHA DE COMPILACAO: Nao detetou Wheel Straight (A-5) quando existem kickers (9, K).");
}

TEST(BugReproduction, StaticAssert_RoyalFlushWithExtraSuitedCard)
{
    static constexpr Deck hand = Deck::parseHand("as ks qs js ts 9s 2c");
    static constexpr ClassificationResult result = Hand::classify(hand);
    
    static_assert(result.getClassification() == Classification::RoyalFlush, 
        "FALHA DE COMPILACAO: Classificou incorretamente Royal Flush 'sujo' como Straight Flush.");
}

TEST(EdgeCases, StaticAssert_Complex7CardHands)
{
    static constexpr Deck threePairHand = Deck::parseHand("as ac ks kc qs qc 2h");
    static_assert(Hand::classify(threePairHand) == ClassificationResult(Classification::TwoPair, Rank::Ace | Rank::King, Rank::Queen),
        "FALHA DE COMPILACAO: Erro ao escolher os melhores 2 pares de 3 possiveis.");

    static constexpr Deck wheelSFHand = Deck::parseHand("5h 4h 3h 2h ah 9h kd");
    static_assert(Hand::classify(wheelSFHand) == ClassificationResult(Classification::StraightFlush, Rank::Five),
        "FALHA DE COMPILACAO: Erro ao detetar Wheel Straight Flush com kicker suited.");

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

    static_assert(r1 > r2, "Pair of Aces should beat Pair of Kings despite having same kickers/ranks.");
}


TEST(BugFix, FullHouse_TripsRankDominates)
{
    static constexpr Deck aaakk = Deck::parseHand("Ah Ac Ad Kh Kd");
    static constexpr Deck kkkaa = Deck::parseHand("Kh Kc Ks Ah Ad");
    static constexpr ClassificationResult r1 = Hand::classify(aaakk);
    static constexpr ClassificationResult r2 = Hand::classify(kkkaa);
    static_assert(r1.getClassification() == Classification::FullHouse, "AAAKK must be FullHouse");
    static_assert(r2.getClassification() == Classification::FullHouse, "KKKAA must be FullHouse");
    static_assert(r1 > r2, "AAAKK must beat KKKAA");
}

TEST(BugFix, FullHouse_PairRankBreaksTie)
{
    static constexpr Deck aaakk = Deck::parseHand("Ah Ac Ad Kh Kd");
    static constexpr Deck aaaqq = Deck::parseHand("Ah Ac Ad Qh Qd");
    static constexpr ClassificationResult r1 = Hand::classify(aaakk);
    static constexpr ClassificationResult r2 = Hand::classify(aaaqq);
    static_assert(r1 > r2, "AAAKK must beat AAAQQ (same trips, K > Q pair)");
}

TEST(BugFix, DoubleTrips_ClassifiesAsFullHouse)
{
    static constexpr Deck hand = Deck::parseHand("Ah Ad Ac Kh Kd Ks 2c");
    static constexpr ClassificationResult result = Hand::classify(hand);
    static_assert(result.getClassification() == Classification::FullHouse,
        "3A + 3K + x must be FullHouse, not ThreeOfAKind");
}

TEST(BugFix, DoubleTrips_HigherTripsWins)
{
    static constexpr Deck hand1 = Deck::parseHand("Ah Ad Ac Kh Kd Ks 2c"); 
    static constexpr Deck hand2 = Deck::parseHand("Kh Kd Kc Qh Qd Qs 2c"); 
    static constexpr ClassificationResult r1 = Hand::classify(hand1);
    static constexpr ClassificationResult r2 = Hand::classify(hand2);
    static_assert(r1 > r2, "AAAKK must beat KKKQQ (from double-trips hands)");
}

TEST(BugFix, HighCard_7Card_KickerNormalization)
{
    static constexpr Deck hand1 = Deck::parseHand("8s 2h Ac Kd Qh Js 9d");
    static constexpr Deck hand2 = Deck::parseHand("7d 2c Ac Kd Qh Js 9d");
    static constexpr ClassificationResult r1 = Hand::classify(hand1);
    static constexpr ClassificationResult r2 = Hand::classify(hand2);
    static_assert(r1.getClassification() == Classification::HighCard, "Must be HighCard");
    static_assert(r1 == r2, "Both share best 5 (AKQJ9); irrelevant 6th card must not break tie");
}

TEST(BugFix, ThreeOfAKind_7Card_KickerNormalization)
{
    static constexpr Deck hand1 = Deck::parseHand("5c 6d Ah Ad As 8c 7h");
    static constexpr Deck hand2 = Deck::parseHand("4c 6d Ah Ad As 8c 7h");
    static constexpr ClassificationResult r1 = Hand::classify(hand1);
    static constexpr ClassificationResult r2 = Hand::classify(hand2);
    static_assert(r1.getClassification() == Classification::ThreeOfAKind, "Must be ThreeOfAKind");
    static_assert(r1 == r2, "Both have AAA87 as best 5; irrelevant 4th kicker must not break tie");
}

TEST(BugFix, FourOfAKind_7Card_KickerNormalization)
{
    static constexpr Deck hand1 = Deck::parseHand("Ah Ac Ad As Kh Jd 2c");
    static constexpr Deck hand2 = Deck::parseHand("Ah Ac Ad As Kh Td 2c");
    static constexpr ClassificationResult r1 = Hand::classify(hand1);
    static constexpr ClassificationResult r2 = Hand::classify(hand2);
    static_assert(r1.getClassification() == Classification::FourOfAKind, "Must be FourOfAKind");
    static_assert(r1 == r2, "Both have AAAA K as best 5; irrelevant 3rd card must not break tie");
}

TEST(BugFix, Flush_6Cards_KickerNormalization)
{
    static constexpr Deck hand1 = Deck::parseHand("Ah Kh Qh Jh 9h 8h 2d");
    static constexpr Deck hand2 = Deck::parseHand("Ah Kh Qh Jh 9h 7h 2d");
    static constexpr ClassificationResult r1 = Hand::classify(hand1);
    static constexpr ClassificationResult r2 = Hand::classify(hand2);
    static_assert(r1.getClassification() == Classification::Flush, "Must be Flush");
    static_assert(r1 == r2, "Both have AhKhQhJh9h as best flush; irrelevant 6th heart must not break tie");
}

TEST(BugFix, Flush_6Cards_WheelStraightFlushStillDetected)
{
    static constexpr Deck hand = Deck::parseHand("5h 4h 3h 2h Ah 9h Kd");
    static constexpr ClassificationResult result = Hand::classify(hand);
    static_assert(result.getClassification() == Classification::StraightFlush,
        "Wheel straight flush must be detected even with a 6th suited card present");
    static_assert(result == ClassificationResult(Classification::StraightFlush, Rank::Five),
        "Must be 5-high straight flush (wheel)");
}

static constexpr ClassificationResult classifyWith(std::string_view board, std::string_view hole)
{
    return Hand::classify(Deck::createDeck({Deck::parseHand(board), Deck::parseHand(hole)}));
}

TEST(BugFix, Trips_HigherTripsWinsWithSameRankSet)
{
    static_assert(classifyWith("Kh Qd Jc 4s 2h", "Ks Kd") > classifyWith("Kh Qd Jc 4s 2h", "Qs Qc"));
}

TEST(BugFix, TwoPair_HigherSecondPairWinsWithSameRankSet)
{
    static_assert(classifyWith("Kh Ks 5h 5s Qd", "Qc 2d") > classifyWith("Kh Ks 5h 5s Qd", "3c 4d"));
}

TEST(BugFix, Pair_LowKickerBreaksTie)
{
    static_assert(classifyWith("9h Kd Qc 3s 2h", "9s 5d") > classifyWith("9h Kd Qc 3s 2h", "9c 4d"));
}

TEST(BugFix, Quads_HigherQuadsWinsWithSameRankSet)
{
    static_assert(classifyWith("Kh Kd 5h 5d 3c", "Ks Kc") > classifyWith("Kh Kd 5h 5d 3c", "5s 5c"));
}

TEST(StreamingTest, PairPrintsPairThenKickers)
{
    std::ostringstream oss;
    oss << classifyWith("9h Kd Qc 3s 2h", "9s 5d");
    EXPECT_EQ(oss.str(), "Pair: 9 + 5 Q K");
}

namespace reference
{
    static std::vector<int> score5(const std::array<int, 5> &cards)
    {
        std::array<int, 13> count{};
        bool flush = true;
        for (int card : cards)
        {
            ++count[card % 13];
            flush &= card / 13 == cards[0] / 13;
        }
        std::vector<std::pair<int, int>> groups; 
        for (int rank = 12; rank >= 0; --rank)
        {
            if (count[rank])
            {
                groups.push_back({count[rank], rank});
            }
        }
        std::stable_sort(groups.begin(), groups.end(), [](auto a, auto b) { return a.first > b.first; });
        int straightHigh = -1;
        if (groups.size() == 5)
        {
            if (groups[0].second - groups[4].second == 4)
            {
                straightHigh = groups[0].second;
            }
            else if (groups[0].second == 12 && groups[1].second == 3)
            {
                straightHigh = 3; 
            }
        }
        int category = 0;
        if (straightHigh >= 0 && flush) { category = 8; }
        else if (groups[0].first == 4) { category = 7; }
        else if (groups[0].first == 3 && groups[1].first == 2) { category = 6; }
        else if (flush) { category = 5; }
        else if (straightHigh >= 0) { category = 4; }
        else if (groups[0].first == 3) { category = 3; }
        else if (groups[0].first == 2 && groups[1].first == 2) { category = 2; }
        else if (groups[0].first == 2) { category = 1; }
        std::vector<int> score{category};
        if (straightHigh >= 0)
        {
            score.push_back(straightHigh);
        }
        else
        {
            for (auto [c, rank] : groups)
            {
                score.push_back(rank);
            }
        }
        return score;
    }

    static std::vector<int> score7(const std::array<int, 7> &cards)
    {
        std::vector<int> best;
        for (int skipA = 0; skipA < 7; ++skipA)
        {
            for (int skipB = skipA + 1; skipB < 7; ++skipB)
            {
                std::array<int, 5> hand{};
                int k = 0;
                for (int i = 0; i < 7; ++i)
                {
                    if (i != skipA && i != skipB)
                    {
                        hand[k++] = cards[i];
                    }
                }
                best = std::max(best, score5(hand));
            }
        }
        return best;
    }

    static Deck toDeck(const std::array<int, 7> &cards)
    {
        std::uint64_t mask = 0;
        for (int card : cards)
        {
            mask |= 1ull << card;
        }
        return Deck::from_mask(mask);
    }
}

TEST(DifferentialFuzz, ClassifyOrdersHandsLikeBruteForce)
{
    std::mt19937_64 rng(7);
    std::array<int, 52> deck{};
    std::iota(deck.begin(), deck.end(), 0);
    int mismatches = 0;
    for (int trial = 0; trial < 100'000 && mismatches < 5; ++trial)
    {
        std::shuffle(deck.begin(), deck.end(), rng);
        if (trial & 1)
        {
            std::stable_partition(deck.begin(), deck.end(), [](int card) { return card % 13 < 5; });
        }
        const std::array<int, 7> a{deck[0], deck[1], deck[2], deck[3], deck[4], deck[5], deck[6]};
        const std::array<int, 7> b{deck[0], deck[1], deck[2], deck[3], deck[4], deck[7], deck[8]}; 
        const auto refA = reference::score7(a);
        const auto refB = reference::score7(b);
        const ClassificationResult engA = Hand::classify(reference::toDeck(a));
        const ClassificationResult engB = Hand::classify(reference::toDeck(b));
        const int expected = (refA > refB) - (refA < refB);
        const int actual = (engA > engB) - (engA < engB);
        const bool royal = refA[0] == 8 && refA[1] == 12;
        const auto expectedCategory = static_cast<Classification>(1u << (refA[0] + (royal ? 1 : 0)));
        if (expected != actual || engA.getClassification() != expectedCategory)
        {
            ++mismatches;
            ADD_FAILURE() << engA << " vs " << engB << ": expected " << expected << ", got " << actual;
        }
    }
    EXPECT_EQ(mismatches, 0);
}
