#ifndef __POKER_TABLE_HPP__
#define __POKER_TABLE_HPP__
#include "deck.hpp"
#include "player.hpp"
#include <array>
#include <cassert>
#include <vector>
#include <span>
class Pot
{
    float m_amount;
    std::vector<Player*> m_players;
public:
    constexpr Pot(float amount, std::vector<Player*> players) : m_amount(amount), m_players(players) {}
    inline constexpr float getAmount() const noexcept { return m_amount; }
    inline constexpr void addAmount(float amount) noexcept { m_amount += amount; }
    inline constexpr void addPlayer(Player* player) noexcept { m_players.push_back(player); }
    inline constexpr void removePlayer(Player* player) noexcept
    {
        auto it = std::remove(m_players.begin(), m_players.end(), player);
        if (it != m_players.end())
        {
            std::iter_swap(it, m_players.end() - 1);
            m_players.pop_back();
        }
    }
    inline constexpr std::span<Player*> getPlayers() noexcept
    {
        return m_players;
    }
};
class Table
{
private:
    Deck m_tableCards;
    std::vector<Pot> m_pots;

public:
    constexpr Table() noexcept : m_tableCards(Deck::emptyDeck()), m_pots() {}
    inline constexpr Deck getTableCards() const noexcept { return m_tableCards; }
    inline constexpr std::span<Pot> getPots() noexcept { return m_pots; }
    inline constexpr void addCards(const Deck cards) noexcept
    {
        assert(m_tableCards.size() + cards.size() <= 5 && "Final size must be lower or equal to 5");
        m_tableCards.addCards(cards);
    }
    inline constexpr void addPot(float amount, std::vector<Player*> players) noexcept
    {
        m_pots.emplace_back(amount, players);
    }
    inline constexpr void resetTable() noexcept
    {
        m_tableCards = Deck::emptyDeck();
        m_pots.clear();
    }
};
#endif // __POKER_TABLE_HPP__