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
    static_assert(result == ClassificationResult(Classification::Pair, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack), "Expected Pair classification");
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
    static constexpr ClassificationResult twoPairQHigh = ClassificationResult(Classification::TwoPair,Rank::Ace | Rank::Queen | Rank::King);
    static constexpr ClassificationResult twoPairJHigh = ClassificationResult(Classification::TwoPair,Rank::Ace | Rank::Jack | Rank::King);
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
    static constexpr auto checkUniqueCards = [](const Deck &playerCards, const Deck &tableCards) {
        return (playerCards.getMask() & tableCards.getMask()) == 0;
    };
    static_assert(checkUniqueCards(p, t1) == false, "Expected duplicate cards in hand and table");
    static_assert(checkUniqueCards(p, t2) == true, "Expected no duplicate cards in hand and table");
}

TEST(DeckTest, RemoveAndAddCards) {
    static constexpr Deck full = Deck::createFullDeck();
    static_assert(full.size() == 52, "Expected full deck size of 52");
    static constexpr Deck firstTestDeck = [] {
        Deck deck = Deck::createFullDeck();
        deck.removeCard(Card(Suit::Spades, Rank::Ten));
        return deck;
    }();
    static_assert(firstTestDeck.size() == 51, "Expected deck size of 51 after removing one card");
    static constexpr Deck secondTestDeck = [] {
        Deck deck = Deck::createFullDeck();
        deck.removeCard(Card(Suit::Spades, Rank::Ten));
        deck.removeCard(Card(Suit::Clubs, Rank::Five));
        return deck;
    }();
    static_assert(secondTestDeck.size() == 50, "Expected deck size of 50 after removing two cards");
    static constexpr Deck thirdTestDeck = [] {
        Deck deck = Deck::createFullDeck();
        deck.removeCard(Card(Suit::Spades, Rank::Ten));
        deck.removeCard(Card(Suit::Clubs, Rank::Five));
        deck.addCard(Card(Suit::Spades, Rank::Ten));
        return deck;
    }();
    static_assert(thirdTestDeck.size() == 51, "Expected deck size of 51 after removing two cards and adding one card");
    static constexpr Deck fourthTestDeck = [] {
        Deck deck = Deck::createFullDeck();
        deck.removeCard(Card(Suit::Spades, Rank::Ten));
        deck.removeCard(Card(Suit::Clubs, Rank::Five));
        deck.addCard(Card(Suit::Spades, Rank::Ten));
        deck.addCard(Card(Suit::Clubs, Rank::Five));
        return deck;
    }();
    static_assert(fourthTestDeck.size() == 52, "Expected deck size of 52 after removing two cards and adding two cards");
}