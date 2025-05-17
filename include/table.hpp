#ifndef __POKER_TABLE_HPP__
#define __POKER_TABLE_HPP__
#include "deck.hpp"
#include "player.hpp"
#include <array>
#include <cassert>
template <std::size_t NPlayers>
class Table
{
private:
    std::array<Player, NPlayers> m_players;
    std::size_t m_activePlayers = 0;
    Deck m_tableCards;
    float m_pot = 0.0f;

public:
    constexpr Table() noexcept : m_tableCards(Deck::emptyDeck()) {}
    inline constexpr void addPlayer(const Player &player) noexcept
    {
        assert(m_activePlayers < NPlayers && "Cannot add more players than the table can hold");
        assert(player.getCards().size() == 2 && "Player must have exactly 2 cards");
        m_players[m_activePlayers++] = player;
    }
    inline constexpr void removePlayer(std::size_t index) noexcept
    {
        assert(index < m_activePlayers);
        std::swap(m_players[m_activePlayers - 1], m_players[index]);
        m_activePlayers--;
    }
    inline constexpr void addTableCards(const Deck cards) noexcept
    {
        assert(m_tableCards.size() + cards.size() <= 5 && "Final size must be lower or equal to 5");
        m_tableCards.addCards(cards);
    }
    inline constexpr Deck getTableCards() const noexcept { return m_tableCards; }
    inline constexpr float getPot() const noexcept { return m_pot; }
    inline constexpr void addToPot(float amount) noexcept
    {
        assert(amount >= 0.0f && "Amount to add to pot must be non-negative");
        m_pot += amount;
    }
    inline constexpr void removeFromPot(float amount) noexcept
    {
        assert(amount >= 0.0f && "Amount to remove from pot must be non-negative");
        assert(amount <= m_pot && "Cannot remove more than the current pot amount");
        m_pot -= amount;
    }
    inline constexpr void clearPot() noexcept { m_pot = 0.0f; }
};
#endif // __POKER_TABLE_HPP__