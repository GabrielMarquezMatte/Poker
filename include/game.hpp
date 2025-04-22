#ifndef __POKER_GAME_HPP__
#define __POKER_GAME_HPP__
#include "classification_result.hpp"
#include "hand.hpp"
#include "deck.hpp"
inline constexpr ClassificationResult classifyPlayer(const Deck playerCards, const Deck tableCards)
{
    Deck allCards = Deck::createDeck({playerCards, tableCards});
    ClassificationResult best = {Classification::HighCard, Rank::Two};
    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = i + 1; j < 7; ++j)
        {
            Deck hand = Deck::emptyDeck();
            for (std::size_t l = 0; l < 7; ++l)
            {
                if (l != i && l != j)
                {
                    auto cardDealt = allCards.at(l);
                    if (!cardDealt.has_value())
                    {
                        return {Classification::HighCard, Rank::Two};
                    }
                    hand.addCard(cardDealt.value());
                }
            }
            ClassificationResult candidate = Hand::classify(hand);
            if (candidate > best)
            {
                best = candidate;
            }
        }
    }
    return best;
}
#endif // __POKER_GAME_HPP__