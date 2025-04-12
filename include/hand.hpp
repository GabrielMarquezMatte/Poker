#ifndef __POKER_HAND_HPP__
#define __POKER_HAND_HPP__
#include <array>
#include <span>
#include <algorithm>
#include "card.hpp"
struct Hand
{
private:
    static inline constexpr std::tuple<Rank, bool, std::array<int, 13>> processCards(const std::span<const Card, 5> cards)
    {
        Rank rankMask = Rank::Two; // ou 0, dependendo de como o Rank é definido
        bool flush = true;
        std::array<int, 13> counts{}; // Inicializa com 0
        // Assume que o primeiro naipe seja o de referência para verificar flush.
        auto firstSuit = cards[0].getSuit();
        for (const Card card : cards)
        {
            Rank cardRank = card.getRank();
            // Atualiza o bitmask dos ranks (talvez o operador | já esteja definido para Rank).
            rankMask = rankMask | cardRank;
            // Atualiza contagem. Supondo que getRankIndex retorne um índice de 0 a 12.
            counts[getRankIndex(cardRank)]++;
            // Checa se todos os naipes são iguais.
            if (card.getSuit() != firstSuit)
            {
                flush = false;
            }
        }
        return {rankMask, flush, counts};
    }
    static inline constexpr std::pair<bool, Rank> getStraight(const Rank rankMask)
    {
        constexpr std::array<uint32_t, 9> straightMasks = {0x1Fu << 0, 0x1Fu << 1, 0x1Fu << 2, 0x1Fu << 3,
                                                           0x1Fu << 4, 0x1Fu << 5, 0x1Fu << 6, 0x1Fu << 7,
                                                           0x1Fu << 8};
        // Special Ace-low case: 2,3,4,5,A
        if (rankMask == Rank::LowStraight)
        {
            return {true, Rank::Five};
        }
        for (int i = 0; i <= 8; ++i)
        {
            // Cria uma máscara com 5 bits consecutivos iniciando em i: (0x1F = 0b11111)
            if ((static_cast<uint32_t>(rankMask) & straightMasks[i]) == straightMasks[i])
            {
                // O rank mais alto da sequência será i+4.
                return {true, static_cast<Rank>(1 << (i + 4))};
            }
        }
        int highestBit = std::bit_width(static_cast<uint32_t>(rankMask)) - 1;
        return {false, static_cast<Rank>(1 << highestBit)};
    }

public:
    static inline constexpr ClassificationResult classify(const std::span<const Card, 5> cards)
    {
        auto [rankMask, flush, counts] = processCards(cards);
        // Check for a straight.
        auto [straight, straightHigh] = getStraight(rankMask);
        if (flush && straight)
        {
            // For a royal flush the high card is Ace.
            if (straightHigh == Rank::Ace)
            {
                return {Classification::RoyalFlush, Rank::Ace | Rank::King | Rank::Queen | Rank::Jack | Rank::Ten};
            }
            return {Classification::StraightFlush, straightHigh};
        }
        // Build a frequency table for ranks.
        int maxCount = 0;
        int secondMaxCount = 0;
        for (int count : counts)
        {
            if (count > maxCount)
            {
                secondMaxCount = maxCount;
                maxCount = count;
                continue;
            }
            if (count > secondMaxCount)
            {
                secondMaxCount = count;
            }
        }

        if (maxCount == 4)
        {
            return {Classification::FourOfAKind, rankMask};
        }
        if (maxCount == 3 && secondMaxCount >= 2)
        {
            return {Classification::FullHouse, rankMask};
        }
        if (flush)
        {
            return {Classification::Flush, rankMask};
        }
        if (straight)
        {
            return {Classification::Straight, straightHigh};
        }
        if (maxCount == 3)
        {
            return {Classification::ThreeOfAKind, rankMask};
        }
        if (maxCount != 2)
        {
            return {Classification::HighCard, rankMask};
        }
        if (secondMaxCount == 2)
        {
            return {Classification::TwoPair, rankMask};
        }
        return {Classification::Pair, rankMask};
    }
};
#endif // __POKER_HAND_HPP__