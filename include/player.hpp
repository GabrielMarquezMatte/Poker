#ifndef __POKER_PLAYER_HPP__
#define __POKER_PLAYER_HPP__
#include "deck.hpp"
class Player
{
private:
    Deck m_cards;
    float m_chips;
public:
    constexpr Player() noexcept : m_cards(Deck::emptyDeck()), m_chips(0.0f) {}
    constexpr Player(const Deck cards, const float chips) noexcept : m_cards(cards), m_chips(chips) {}
    inline constexpr float getChips() const noexcept { return m_chips; }
    inline constexpr void setChips(const float chips) noexcept { m_chips = chips; }
    inline constexpr Deck getCards() const noexcept { return m_cards; }
    inline constexpr void addChips(const float amount) noexcept { m_chips += amount; }
    inline constexpr void removeChips(const float amount) noexcept { m_chips -= amount; }
    template<typename RNG>
    inline float bet(RNG &rng, const float minBet, const float maxBet) noexcept
    {
        if (m_chips <= 0.0f)
        {
            return 0.0f;
        }
        std::uniform_real_distribution<float> dist(minBet, std::min(maxBet, m_chips));
        return dist(rng);
    }
    inline constexpr void addCards(const Deck cards) noexcept { m_cards.addCards(cards); }
};
#endif // __POKER_PLAYER_HPP__