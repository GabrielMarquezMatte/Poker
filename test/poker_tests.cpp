#include <gtest/gtest.h>
#include "../include/hand.hpp"
#include "../include/deck.hpp"
#include "../include/card.hpp"

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
    static_assert(result == ClassificationResult(Classification::StraightFlush, Rank::Nine | Rank::Eight | Rank::Seven | Rank::Six | Rank::Five), "Expected Straight Flush classification");
}
TEST(DeckTest, ClassifyStraight)
{
    static constexpr Deck deck = Deck::createDeck({Card(Suit::Hearts, Rank::Ace),
                                                   Card(Suit::Hearts, Rank::King),
                                                   Card(Suit::Hearts, Rank::Queen),
                                                   Card(Suit::Hearts, Rank::Jack),
                                                   Card(Suit::Diamonds, Rank::Ten)});
    static constexpr ClassificationResult result = Hand::classify(deck);
    static_assert(result == ClassificationResult(Classification::Straight, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten), "Expected Flush classification");
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