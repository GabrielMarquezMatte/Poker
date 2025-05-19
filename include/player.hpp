#ifndef __POKER_PLAYER_HPP__
#define __POKER_PLAYER_HPP__
#include "deck.hpp"
#include <cassert>
class Player
{
private:
    Deck m_cards;
    float m_chips;
    float m_contribution = 0.0f;

public:
    constexpr Player() noexcept : m_cards(Deck::emptyDeck()), m_chips(0.0f) {}
    constexpr Player(const Deck cards, const float chips) noexcept : m_cards(cards), m_chips(chips) {}
    inline constexpr float getChips() const noexcept { return m_chips; }
    inline constexpr void setChips(const float chips) noexcept { m_chips = chips; }
    inline constexpr Deck getCards() const noexcept { return m_cards; }
    inline constexpr void addChips(const float amount) noexcept { m_chips += amount; m_contribution -= amount; }
    inline constexpr void removeChips(const float amount) noexcept { m_chips -= amount; m_contribution += amount; }
    inline constexpr float getContribution() const noexcept { return m_contribution; }
    inline constexpr void setContribution(const float contribution) noexcept { m_contribution = contribution; }
    inline constexpr void addCards(const Deck cards) noexcept
    {
        assert(m_cards.size() + cards.size() <= 2 && "Player can only have 2 cards");
        m_cards.addCards(cards);
    }
    inline constexpr void clearCards() noexcept
    {
        m_cards = Deck::emptyDeck();
    }
};
#endif // __POKER_PLAYER_HPP__