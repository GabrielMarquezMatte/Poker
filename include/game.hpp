#ifndef __POKER_GAME_HPP__
#define __POKER_GAME_HPP__
#include "classification_result.hpp"
#include "hand.hpp"
inline constexpr ClassificationResult classifyPlayer(const std::span<const Card, 2> playerCards, const std::span<const Card, 5> tableCards)
{
    std::array<Card, 7> cards;
    std::copy(tableCards.begin(), tableCards.end(), cards.begin());
    std::copy(playerCards.begin(), playerCards.end(), cards.begin() + 5);
    ClassificationResult best = {Classification::HighCard, Rank::Two};
    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = i + 1; j < 7; ++j)
        {
            std::array<Card, 5> hand;
            std::size_t k = 0;
            for (std::size_t l = 0; l < 7; ++l)
            {
                if (l != i && l != j)
                {
                    hand[k++] = cards[l];
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