#ifndef __POKER_CFR_HUNL_HPP__
#define __POKER_CFR_HUNL_HPP__
#include "mccfr.hpp"
#include "../game.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

// 100bb heads-up with a 0.5/1/2 pot raise abstraction on every street. Chips: small blind 1, big blind 2.
// Postflop buckets here are the made-hand category (9 buckets): enough for tests and smoke runs.
// Real training overrides postflopBucket with CardAbstraction::bucket.
struct Hunl100bbConfig
{
    static constexpr std::uint32_t stack = 200;
    static constexpr std::uint32_t smallBlind = 1;
    static constexpr std::uint32_t bigBlind = 2;
    // Per street (preflop, flop, turn, river); 0 marks an unused slot.
    static constexpr std::array<std::array<double, 3>, 4> raiseFractions{{{0.5, 1.0, 2.0}, {0.5, 1.0, 2.0}, {0.5, 1.0, 2.0}, {0.5, 1.0, 2.0}}};
    static constexpr std::array<std::uint8_t, 4> maxRaises{3, 3, 3, 3};
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
    static constexpr std::size_t maxActions = 3 + C::raiseFractions[0].size();
    static constexpr std::uint32_t fold = ~0u;
    static constexpr std::uint8_t nobody = 2;
    static constexpr std::uint8_t showdown = 4;

    struct State
    {
        std::array<std::uint64_t, 2> hole{};
        std::array<std::uint64_t, 3> runout{}; // flop, turn, river: dealt up front, revealed street by street
        std::uint64_t board = 0;
        std::uint64_t history = 0; // hash of the public action sequence
        std::array<std::uint32_t, 2> invested{C::smallBlind, C::bigBlind};
        std::uint32_t lastRaise = C::bigBlind;
        std::array<std::array<std::uint16_t, 2>, 4> bucket{}; // [street][player]
        std::uint8_t street = 0; // 0 preflop .. 3 river, 4 showdown (board runs out)
        std::uint8_t toAct = 0;
        std::uint8_t actions = 0; // this street
        std::uint8_t raises = 0;  // this street
        std::uint8_t folder = nobody;
        std::int8_t winner = 0; // at showdown: 1 player 0 wins, -1 player 1 wins, 0 split
        bool dealt = false;
    };

    // Target total investment of the actor per legal action (fold = sentinel).
    struct Actions
    {
        std::array<std::uint32_t, maxActions> to{};
        std::size_t size = 0;
    };

    static State initial() { return {}; }

    // All cards are dealt at once so buckets and the showdown are computed once per deal, not per visit.
    static State deal(const State &s, std::uint64_t hole0, std::uint64_t hole1, std::uint64_t flop, std::uint64_t turn, std::uint64_t river)
    {
        State next = s;
        next.hole = {hole0, hole1};
        next.runout = {flop, turn, river};
        next.dealt = true;
        for (std::size_t p = 0; p < 2; ++p)
        {
            next.bucket[0][p] = static_cast<std::uint16_t>(preflopClassIndex(Deck::from_mask(next.hole[p])));
            std::uint64_t board = 0;
            for (std::size_t street = 1; street < 4; ++street)
            {
                board |= next.runout[street - 1];
                next.bucket[street][p] = C::postflopBucket(next.hole[p], board);
            }
        }
        const std::uint64_t board = flop | turn | river;
        const ClassificationResult first = Hand::classify(Deck::from_mask(hole0 | board));
        const ClassificationResult second = Hand::classify(Deck::from_mask(hole1 | board));
        next.winner = static_cast<std::int8_t>(first > second ? 1 : (first < second ? -1 : 0));
        return next;
    }

    static bool isTerminal(const State &s)
    {
        return s.folder != nobody || (s.street == showdown && std::popcount(s.board) == 5);
    }

    static bool isChance(const State &s)
    {
        return !s.dealt || std::popcount(s.board) < cardsOnBoard[s.street];
    }

    static State sampleChance(const State &s, CfrRng &rng)
    {
        if (!s.dealt)
        {
            Deck deck = Deck::createFullDeck();
            const std::uint64_t hole0 = deck.popPair(rng).getMask();
            const std::uint64_t hole1 = deck.popPair(rng).getMask();
            const std::uint64_t flop = deck.popRandomCards(rng, 3).getMask();
            const std::uint64_t turn = deck.popRandomCards(rng, 1).getMask();
            return deal(s, hole0, hole1, flop, turn, deck.popRandomCards(rng, 1).getMask());
        }
        State next = s;
        for (std::size_t i = 0; i < std::min<std::size_t>(s.street, 3); ++i)
        {
            next.board |= s.runout[i];
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
        const int result = player == 0 ? s.winner : -s.winner;
        if (result == 0)
        {
            return 0.0;
        }
        return result > 0 ? static_cast<double>(s.invested[opp]) : -static_cast<double>(s.invested[player]);
    }

    static std::size_t currentPlayer(const State &s) { return s.toAct; }
    static std::size_t numActions(const State &s) { return legalActions(s).size; }

    static std::uint64_t infosetKey(const State &s) { return mix(s.history, s.bucket[s.street][s.toAct]); }

    // For range tracking: the bucket and infoset key of the player to act if they held `hole`.
    static std::uint16_t bucketFor(const State &s, std::uint64_t hole)
    {
        return s.street == 0 ? static_cast<std::uint16_t>(preflopClassIndex(Deck::from_mask(hole))) : C::postflopBucket(hole, s.board);
    }
    static std::uint64_t infosetKeyWithBucket(const State &s, std::uint16_t bucket) { return mix(s.history, bucket); }

    // With C::preflopAverageOnly the solver keeps the average strategy preflop only (Mccfr's PartlyAveraged):
    // for blueprints whose later streets are re-solved at play time.
    static bool averaged(const State &s)
        requires requires { C::preflopAverageOnly; }
    {
        return s.street == 0;
    }
    static std::vector<std::uint64_t> averagedKeys()
        requires requires { C::preflopAverageOnly; }
    {
        std::vector<std::uint64_t> keys;
        preflopKeys(initial(), keys);
        return keys;
    }

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
        if (s.raises >= C::maxRaises[s.street] || facing >= C::stack)
        {
            return out;
        }
        const std::uint32_t potAfterCall = 2 * facing;
        std::uint32_t last = facing + s.lastRaise - 1; // below this is not a legal raise; also skips 0 slots
        // Optionally, a street's last allowed raise is all-in only (C::allInOnlyRaises from that raise on).
        if constexpr (requires { C::allInOnlyRaises; })
        {
            if (s.raises + 1 >= C::allInOnlyRaises[s.street])
            {
                out.to[out.size++] = C::stack;
                return out;
            }
        }
        for (const double fraction : C::raiseFractions[s.street])
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

    static State apply(const State &s, std::size_t action) { return applyTo(s, legalActions(s).to[action], action); }
    static State apply(const State &s, const Actions &legal, std::size_t action) { return applyTo(s, legal.to[action], action); }

    // The actor puts `to` chips in total (fold = sentinel), which need not be one of legalActions: real
    // opponents bet any size. `token` extends the history hash (apply passes the action index).
    static State applyTo(const State &s, std::uint32_t to, std::uint64_t token)
    {
        const std::uint8_t player = s.toAct;
        const std::uint8_t opp = static_cast<std::uint8_t>(1 - player);
        State next = s;
        next.history = mix(s.history, token);
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
    static void preflopKeys(const State &s, std::vector<std::uint64_t> &keys)
    {
        if (isTerminal(s) || s.street != 0)
        {
            return;
        }
        for (std::uint16_t bucket = 0; bucket < 169; ++bucket)
        {
            keys.push_back(infosetKeyWithBucket(s, bucket));
        }
        for (std::size_t a = 0; a < numActions(s); ++a)
        {
            preflopKeys(apply(s, a), keys);
        }
    }

    static std::uint64_t mix(std::uint64_t h, std::uint64_t v)
    {
        std::uint64_t x = h ^ (v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2));
        return omp::splitmix64(x);
    }
};
#endif // __POKER_CFR_HUNL_HPP__
