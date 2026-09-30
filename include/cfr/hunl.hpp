#ifndef __POKER_CFR_HUNL_HPP__
#define __POKER_CFR_HUNL_HPP__
#include "mccfr.hpp"
#include "../game.hpp"
#include <array>
#include <bit>
#include <cstdint>

// 100bb heads-up with a 0.5/1/2 pot raise abstraction. Chips: small blind 1, big blind 2.
// Postflop buckets here are the made-hand category (9 buckets): enough for tests and smoke runs.
// Real training overrides postflopBucket with CardAbstraction::bucket.
struct Hunl100bbConfig
{
    static constexpr std::uint32_t stack = 200;
    static constexpr std::uint32_t smallBlind = 1;
    static constexpr std::uint32_t bigBlind = 2;
    static constexpr std::array<double, 3> raiseFractions{0.5, 1.0, 2.0};
    static constexpr std::uint8_t maxRaisesPerStreet = 3;
    static constexpr bool allowLimp = true;
    static std::uint16_t postflopBucket(std::uint64_t hole, std::uint64_t board)
    {
        const ClassificationResult r = Hand::classify(Deck::from_mask(hole | board));
        return static_cast<std::uint16_t>(getClassificationIndex(r.getClassification()));
    }
};

// Heads-up no-limit hold'em for CfrGame. Player 0 is the small blind/button: acts first preflop, last postflop.
// Actions are indices into legalActions(s): [fold] [check/call] [pot-fraction raises...] [all-in].
template <typename C>
struct Hunl
{
    static constexpr std::size_t numPlayers = 2;
    static constexpr std::size_t maxActions = 3 + C::raiseFractions.size();
    static constexpr std::uint32_t fold = ~0u;
    static constexpr std::uint8_t nobody = 2;
    static constexpr std::uint8_t showdown = 4;

    struct State
    {
        std::array<std::uint64_t, 2> hole{};
        std::uint64_t board = 0;
        std::uint64_t history = 0; // hash of the public action sequence
        std::array<std::uint32_t, 2> invested{C::smallBlind, C::bigBlind};
        std::uint32_t lastRaise = C::bigBlind;
        std::array<std::uint16_t, 2> bucket{};
        std::uint8_t street = 0; // 0 preflop .. 3 river, 4 showdown (board runs out)
        std::uint8_t toAct = 0;
        std::uint8_t actions = 0; // this street
        std::uint8_t raises = 0;  // this street
        std::uint8_t folder = nobody;
        bool dealt = false;
    };

    // Target total investment of the actor per legal action (fold = sentinel).
    struct Actions
    {
        std::array<std::uint32_t, maxActions> to{};
        std::size_t size = 0;
    };

    static State initial() { return {}; }

    static State withHoles(const State &s, std::uint64_t hole0, std::uint64_t hole1)
    {
        State next = s;
        next.hole = {hole0, hole1};
        next.dealt = true;
        for (std::size_t p = 0; p < 2; ++p)
        {
            next.bucket[p] = static_cast<std::uint16_t>(preflopClassIndex(Deck::from_mask(next.hole[p])));
        }
        return next;
    }

    static bool isTerminal(const State &s)
    {
        return s.folder != nobody || (s.street == showdown && std::popcount(s.board) == 5);
    }

    static bool isChance(const State &s)
    {
        return !s.dealt || std::popcount(s.board) < boardSize[s.street];
    }

    static State sampleChance(const State &s, CfrRng &rng)
    {
        Deck deck = Deck::from_mask(Deck::createFullDeck().getMask() & ~(s.hole[0] | s.hole[1] | s.board));
        if (!s.dealt)
        {
            const std::uint64_t hole0 = deck.popPair(rng).getMask();
            return withHoles(s, hole0, deck.popPair(rng).getMask());
        }
        State next = s;
        next.board |= deck.popRandomCards(rng, static_cast<std::size_t>(boardSize[s.street] - std::popcount(s.board))).getMask();
        if (next.street != showdown)
        {
            for (std::size_t p = 0; p < 2; ++p)
            {
                next.bucket[p] = C::postflopBucket(next.hole[p], next.board);
            }
        }
        return next;
    }

    static double utility(const State &s, std::size_t player)
    {
        const std::size_t opp = 1 - player;
        if (s.folder != nobody)
        {
            return s.folder == player ? -static_cast<double>(s.invested[player]) : static_cast<double>(s.invested[opp]);
        }
        const ClassificationResult mine = Hand::classify(Deck::from_mask(s.hole[player] | s.board));
        const ClassificationResult theirs = Hand::classify(Deck::from_mask(s.hole[opp] | s.board));
        if (mine == theirs)
        {
            return 0.0;
        }
        return mine > theirs ? static_cast<double>(s.invested[opp]) : -static_cast<double>(s.invested[player]);
    }

    static std::size_t currentPlayer(const State &s) { return s.toAct; }
    static std::size_t numActions(const State &s) { return legalActions(s).size; }

    static std::uint64_t infosetKey(const State &s) { return mix(s.history, s.bucket[s.toAct]); }

    static Actions legalActions(const State &s)
    {
        Actions out;
        const std::uint32_t facing = s.invested[1 - s.toAct];
        const std::uint32_t mine = s.invested[s.toAct];
        if (facing > mine)
        {
            out.to[out.size++] = fold;
        }
        if (C::allowLimp || s.street != 0 || s.actions != 0)
        {
            out.to[out.size++] = facing;
        }
        if (s.raises >= C::maxRaisesPerStreet || facing >= C::stack)
        {
            return out;
        }
        const std::uint32_t potAfterCall = 2 * facing;
        std::uint32_t last = facing + s.lastRaise - 1; // below this is not a legal raise
        for (const double fraction : C::raiseFractions)
        {
            const std::uint32_t to = facing + static_cast<std::uint32_t>(fraction * potAfterCall);
            if (to > last && to < C::stack)
            {
                out.to[out.size++] = to;
                last = to;
            }
        }
        out.to[out.size++] = C::stack;
        return out;
    }

    static State apply(const State &s, std::size_t action)
    {
        const std::uint32_t to = legalActions(s).to[action];
        const std::uint8_t player = s.toAct;
        const std::uint8_t opp = static_cast<std::uint8_t>(1 - player);
        State next = s;
        next.history = mix(s.history, action);
        ++next.actions;
        if (to == fold)
        {
            next.folder = player;
            return next;
        }
        next.invested[player] = to;
        if (to > s.invested[opp])
        {
            // ponytail: short all-ins never reopen raising; they are always the last option anyway.
            next.lastRaise = std::max(s.lastRaise, to - s.invested[opp]);
            ++next.raises;
            next.toAct = opp;
            return next;
        }
        if (next.actions < 2)
        {
            next.toAct = opp; // opening check, or small blind limp
            return next;
        }
        next.street = next.invested[0] == C::stack ? showdown : static_cast<std::uint8_t>(s.street + 1);
        next.toAct = 1;
        next.actions = 0;
        next.raises = 0;
        next.lastRaise = C::bigBlind;
        return next;
    }

private:
    static constexpr std::array<int, 5> boardSize{0, 3, 4, 5, 5};

    static std::uint64_t mix(std::uint64_t h, std::uint64_t v)
    {
        std::uint64_t x = h ^ (v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2));
        return omp::splitmix64(x);
    }
};
#endif // __POKER_CFR_HUNL_HPP__
