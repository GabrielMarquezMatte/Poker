#ifndef __POKER_TABLE_HPP__
#define __POKER_TABLE_HPP__
#include "deck.hpp"
#include "player.hpp"
#include <array>
#include <cassert>
class Table
{
private:
    std::size_t m_activePlayers = 0;
    Deck m_tableCards;
    float m_pot = 0.0f;
    float m_currentBet = 0.0f;
    float m_minBet = 0.0f;

public:
    constexpr Table() noexcept : m_tableCards(Deck::emptyDeck()) {}
    inline constexpr Deck getTableCards() const noexcept { return m_tableCards; }
    inline constexpr float getPot() const noexcept { return m_pot; }
    inline constexpr float getCurrentBet() const noexcept { return m_currentBet; }
    inline constexpr float getMinBet() const noexcept { return m_minBet; }
    inline constexpr void addCards(const Deck cards) noexcept
    {
        assert(m_tableCards.size() + cards.size() <= 5 && "Final size must be lower or equal to 5");
        m_tableCards.addCards(cards);
    }
    inline constexpr void addToPot(float amount) noexcept
    {
        assert(amount >= 0.0f && "Amount to add to pot must be non-negative");
        assert(amount >= m_minBet && "Amount to add to pot must be greater than or equal to min bet");
        m_pot += amount;
        m_currentBet += amount;
        m_minBet = m_currentBet;
    }
    inline constexpr void resetTable() noexcept
    {
        m_tableCards = Deck::emptyDeck();
        m_pot = 0.0f;
        m_currentBet = 0.0f;
        m_minBet = 0.0f;
    }
};
#endif // __POKER_TABLE_HPP__