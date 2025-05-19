#ifndef __POKER_TABLE_HPP__
#define __POKER_TABLE_HPP__
#include "deck.hpp"
#include "player.hpp"
#include <array>
#include <cassert>
class Table
{
private:
    Deck m_tableCards;
    float m_pot = 0.0f;

public:
    constexpr Table() noexcept : m_tableCards(Deck::emptyDeck()) {}
    inline constexpr Deck getTableCards() const noexcept { return m_tableCards; }
    inline constexpr float getPot() const noexcept { return m_pot; }
    inline constexpr void addCards(const Deck cards) noexcept
    {
        assert(m_tableCards.size() + cards.size() <= 5 && "Final size must be lower or equal to 5");
        m_tableCards.addCards(cards);
    }
    inline constexpr void addToPot(float amount) noexcept
    {
        assert(amount >= 0.0f && "Amount to add to pot must be non-negative");
        m_pot += amount;
    }
    inline constexpr void resetTable() noexcept
    {
        m_tableCards = Deck::emptyDeck();
        m_pot = 0.0f;
    }
};
#endif // __POKER_TABLE_HPP__