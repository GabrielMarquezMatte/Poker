#ifndef __POKER_CFR_PUSH_FOLD_HPP__
#define __POKER_CFR_PUSH_FOLD_HPP__
#include "hand_indexer.hpp"
#include "hunl.hpp"
#include <BS_thread_pool.hpp>
#include <fstream>
#include <string>
#include <vector>

// 20bb push/fold: small blind shoves or folds, big blind calls or folds. Preflop buckets are lossless
// here, so the solver's exploitability can be measured exactly (pushFoldExploitability).
struct PushFold20bb : Hunl100bbConfig
{
    static constexpr std::uint32_t stack = 40;
    static constexpr std::array<std::array<double, 0>, 4> raiseFractions{};
    static constexpr std::array<std::uint8_t, 4> maxRaises{1, 1, 1, 1};
    static constexpr bool allowLimp = false;
};
using PushFold = Hunl<PushFold20bb>;

constexpr std::size_t holeCombos = 1326;

// Index in [0, 1326) of a two-card mask (colex of the card bits), and back.
inline constexpr std::size_t holeIndex(std::uint64_t hole) noexcept
{
    const std::size_t low = static_cast<std::size_t>(std::countr_zero(hole));
    const std::size_t high = static_cast<std::size_t>(63 - std::countl_zero(hole));
    return high * (high - 1) / 2 + low;
}

inline constexpr std::array<std::uint64_t, holeCombos> holes = []
{
    std::array<std::uint64_t, holeCombos> out{};
    for (std::size_t high = 1; high < 52; ++high)
    {
        for (std::size_t low = 0; low < high; ++low)
        {
            out[high * (high - 1) / 2 + low] = (1ull << high) | (1ull << low);
        }
    }
    return out;
}();

// Exact all-in equity of every hole pair: equity(i, j) = P(i wins) + P(tie) / 2 over all 5-card boards.
class PreflopEquity
{
public:
    // Evaluates each suit-isomorphic (hero | villain) class over all C(48, 5) boards.
    static PreflopEquity compute(BS::thread_pool<BS::tp::none> &pool)
    {
        static constexpr std::array<std::uint8_t, 2> rounds{2, 2};
        const HandIndexer indexer(rounds);
        std::vector<float> classEquity(indexer.size());
        pool.detach_blocks<std::uint64_t>(0, indexer.size(), [&](std::uint64_t begin, std::uint64_t end)
                                          {
            for (std::uint64_t c = begin; c < end; ++c)
            {
                const auto hands = indexer.unindex(c);
                classEquity[c] = allBoardsEquity(hands[0], hands[1]);
            } }, 1024);
        pool.wait();
        PreflopEquity out;
        out.m_equity.assign(holeCombos * holeCombos, 0.0f);
        for (std::size_t i = 0; i < holeCombos; ++i)
        {
            for (std::size_t j = 0; j < holeCombos; ++j)
            {
                if ((holes[i] & holes[j]) == 0)
                {
                    out.m_equity[i * holeCombos + j] = classEquity[indexer.index({holes[i], holes[j]})];
                }
            }
        }
        return out;
    }

    inline float operator()(std::size_t i, std::size_t j) const noexcept { return m_equity[i * holeCombos + j]; }

    bool save(const std::string &path) const
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(m_equity.data()), static_cast<std::streamsize>(m_equity.size() * sizeof(float)));
        return static_cast<bool>(out);
    }

    bool load(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        m_equity.resize(holeCombos * holeCombos);
        in.read(reinterpret_cast<char *>(m_equity.data()), static_cast<std::streamsize>(m_equity.size() * sizeof(float)));
        return static_cast<bool>(in) && in.peek() == std::ifstream::traits_type::eof();
    }

private:
    std::vector<float> m_equity;

    static float allBoardsEquity(std::uint64_t hero, std::uint64_t villain)
    {
        const std::uint64_t rest = ((1ull << 52) - 1) & ~(hero | villain);
        std::uint64_t points = 0; // 2 per win, 1 per tie
        std::uint64_t boards = 0;
        // Gosper's hack over 5-of-48 subsets, deposited into the remaining cards.
        for (std::uint64_t v = 0b11111; v < (1ull << 48);)
        {
            const std::uint64_t board = pdep(v, rest);
            const ClassificationResult h = Hand::classify(Deck::from_mask(hero | board));
            const ClassificationResult w = Hand::classify(Deck::from_mask(villain | board));
            points += h > w ? 2 : (h == w ? 1 : 0);
            ++boards;
            const std::uint64_t t = v | (v - 1);
            v = (t + 1) | (((~t & (t + 1)) - 1) >> (std::countr_zero(v) + 1));
        }
        return static_cast<float>(static_cast<double>(points) / static_cast<double>(2 * boards));
    }
};

// Push probability per preflop class for the small blind, call probability for the big blind.
struct PushFoldStrategy
{
    std::array<double, 169> push{};
    std::array<double, 169> call{};
};

inline PushFoldStrategy pushFoldStrategy(const Mccfr<PushFold> &solver)
{
    // Infoset keys depend only on the actor's class and the history; the other cards just complete a deal.
    const auto dealWith = [](std::uint64_t sb, std::uint64_t bb)
    {
        Deck rest = Deck::from_mask(((1ull << 52) - 1) & ~(sb | bb));
        const std::uint64_t flop = rest.popCards(3).getMask();
        const std::uint64_t turn = rest.popCards(1).getMask();
        return PushFold::deal(PushFold::initial(), sb, bb, flop, turn, rest.popCards(1).getMask());
    };
    PushFoldStrategy out;
    for (const std::uint64_t hole : holes)
    {
        const std::size_t c = preflopClassIndex(Deck::from_mask(hole));
        const std::uint64_t opponent = *std::find_if(holes.begin(), holes.end(), [hole](std::uint64_t h)
                                                     { return (h & hole) == 0; });
        out.push[c] = solver.averageStrategy(PushFold::infosetKey(dealWith(hole, opponent)), 2)[1];
        out.call[c] = solver.averageStrategy(PushFold::infosetKey(PushFold::apply(dealWith(opponent, hole), 1)), 2)[1];
    }
    return out;
}

struct PushFoldExploitability
{
    double smallBlindBestResponse; // chips per hand won by a best-responding small blind
    double bigBlindBestResponse;   // chips per hand won by a best-responding big blind
    // Average gain of a best response over both seats; 0 exactly at equilibrium.
    inline double mbbPerHand() const noexcept { return (smallBlindBestResponse + bigBlindBestResponse) / 2.0 / PushFold20bb::bigBlind * 1000.0; }
};

inline PushFoldExploitability pushFoldExploitability(const PreflopEquity &equity, const PushFoldStrategy &strategy)
{
    constexpr double stack = PushFold20bb::stack, small = PushFold20bb::smallBlind, big = PushFold20bb::bigBlind;
    std::array<std::size_t, holeCombos> cls{};
    for (std::size_t i = 0; i < holeCombos; ++i)
    {
        cls[i] = preflopClassIndex(Deck::from_mask(holes[i]));
    }
    double smallBlindTotal = 0.0; // small blind best-responds to strategy.call
    double bigBlindTotal = 0.0;   // big blind best-responds to strategy.push
    for (std::size_t i = 0; i < holeCombos; ++i)
    {
        double pushValue = 0.0, foldValue = 0.0; // small blind holding i
        double callValue = 0.0, passValue = 0.0; // big blind holding i
        for (std::size_t j = 0; j < holeCombos; ++j)
        {
            if ((holes[i] & holes[j]) != 0)
            {
                continue;
            }
            const double allIn = stack * (2.0 * equity(i, j) - 1.0);
            const double q = strategy.call[cls[j]];
            pushValue += q * allIn + (1.0 - q) * big;
            foldValue -= small;
            const double p = strategy.push[cls[j]];
            callValue += (1.0 - p) * small + p * allIn;
            passValue += (1.0 - p) * small - p * big;
        }
        smallBlindTotal += std::max(pushValue, foldValue);
        bigBlindTotal += std::max(callValue, passValue);
    }
    constexpr double deals = holeCombos * 1225.0;
    return {smallBlindTotal / deals, bigBlindTotal / deals};
}
#endif // __POKER_CFR_PUSH_FOLD_HPP__
