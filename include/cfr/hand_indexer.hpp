#ifndef __POKER_CFR_HAND_INDEXER_HPP__
#define __POKER_CFR_HAND_INDEXER_HPP__
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <span>
#include <vector>

// Dense index of hands modulo suit isomorphism (Waugh 2013). Rounds are card groups kept apart
// (hole | flop | turn | river); a round is a Deck mask (bit = rank + 13 * suit).
class HandIndexer
{
public:
    static constexpr std::size_t maxRounds = 4;
    using Rounds = std::array<std::uint64_t, maxRounds>;

    explicit HandIndexer(std::span<const std::uint8_t> roundSizes) : m_rounds(roundSizes.size())
    {
        std::copy(roundSizes.begin(), roundSizes.end(), m_sizes.begin());
        std::vector<std::uint64_t> keys;
        std::array<std::array<std::uint8_t, maxRounds>, 4> counts{};
        collectPatterns(0, 0, 0, counts, keys);
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        for (const std::uint64_t key : keys)
        {
            m_patterns.push_back({key, m_size});
            m_size += patternSize(key);
        }
    }

    inline std::uint64_t size() const noexcept { return m_size; }

    std::uint64_t index(const Rounds &rounds) const noexcept
    {
        std::array<Suit, 4> suits{};
        for (std::size_t s = 0; s < 4; ++s)
        {
            std::uint32_t used = 0;
            std::uint64_t multiplier = 1;
            for (std::size_t r = 0; r < m_rounds; ++r)
            {
                const std::uint32_t ranks = static_cast<std::uint32_t>(rounds[r] >> (13 * s)) & 0x1FFF;
                const std::uint32_t count = static_cast<std::uint32_t>(std::popcount(ranks));
                const std::uint32_t available = 13 - static_cast<std::uint32_t>(std::popcount(used));
                suits[s].config = static_cast<std::uint16_t>(suits[s].config | (count << (3 * (maxRounds - 1 - r))));
                suits[s].value += multiplier * colexIndex(_pext_u32(ranks, ~used & 0x1FFF));
                multiplier *= choose(available, count);
                used |= ranks;
            }
        }
        std::sort(suits.begin(), suits.end(), [](const Suit &a, const Suit &b)
                  { return a.config != b.config ? a.config > b.config : a.value < b.value; });
        std::uint64_t key = 0;
        for (const Suit &suit : suits)
        {
            key = (key << 16) | suit.config;
        }
        const auto pattern = std::lower_bound(m_patterns.begin(), m_patterns.end(), key, [](const Pattern &p, std::uint64_t k)
                                              { return p.key < k; });
        std::uint64_t idx = 0;
        std::uint64_t multiplier = 1;
        for (std::size_t begin = 0; begin < 4;)
        {
            const std::size_t end = groupEnd(suits, begin);
            const std::uint64_t m = end - begin;
            std::uint64_t groupIndex = 0;
            for (std::uint64_t i = 1; i <= m; ++i)
            {
                groupIndex += choose(suits[begin + i - 1].value + i - 1, i);
            }
            idx += groupIndex * multiplier;
            multiplier *= choose(valuesFor(suits[begin].config) + m - 1, m);
            begin = end;
        }
        return pattern->offset + idx;
    }

    // Canonical representative of an index.
    Rounds unindex(std::uint64_t idx) const noexcept
    {
        const auto pattern = std::prev(std::upper_bound(m_patterns.begin(), m_patterns.end(), idx, [](std::uint64_t i, const Pattern &p)
                                                        { return i < p.offset; }));
        std::uint64_t rest = idx - pattern->offset;
        std::array<Suit, 4> suits{};
        for (std::size_t s = 0; s < 4; ++s)
        {
            suits[s].config = static_cast<std::uint16_t>(pattern->key >> (16 * (3 - s)));
        }
        Rounds rounds{};
        for (std::size_t begin = 0; begin < 4;)
        {
            const std::size_t end = groupEnd(suits, begin);
            const std::uint64_t m = end - begin;
            const std::uint64_t n = valuesFor(suits[begin].config);
            const std::uint64_t groupSize = choose(n + m - 1, m);
            std::uint64_t groupIndex = rest % groupSize;
            rest /= groupSize;
            for (std::uint64_t i = m; i >= 1; --i)
            {
                const std::uint64_t w = largestWithChooseAtMost(groupIndex, i, n + m - 1);
                groupIndex -= choose(w, i);
                decodeSuit(suits[begin].config, w - (i - 1), begin + i - 1, rounds);
            }
            begin = end;
        }
        return rounds;
    }

private:
    struct Suit
    {
        std::uint16_t config = 0; // per-round card counts, 3 bits each, round 0 most significant
        std::uint64_t value = 0;  // index of the per-round rank sets given config
    };
    struct Pattern
    {
        std::uint64_t key;    // 4 suit configs, sorted descending, 16 bits each
        std::uint64_t offset; // patterns sorted by key; offsets grow in the same order
    };

    std::size_t m_rounds;
    std::array<std::uint8_t, maxRounds> m_sizes{};
    std::vector<Pattern> m_patterns;
    std::uint64_t m_size = 0;

    static constexpr std::uint64_t choose(std::uint64_t n, std::uint64_t k) noexcept
    {
        if (k > n)
        {
            return 0;
        }
        std::uint64_t result = 1;
        for (std::uint64_t i = 0; i < k; ++i)
        {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    static std::uint64_t colexIndex(std::uint32_t set) noexcept
    {
        std::uint64_t idx = 0;
        for (std::uint64_t i = 1; set != 0; ++i, set &= set - 1)
        {
            idx += choose(static_cast<std::uint64_t>(std::countr_zero(set)), i);
        }
        return idx;
    }

    static std::uint32_t colexUnindex(std::uint64_t idx, std::uint64_t k) noexcept
    {
        std::uint32_t set = 0;
        for (std::uint64_t i = k; i >= 1; --i)
        {
            const std::uint64_t p = largestWithChooseAtMost(idx, i, 12);
            idx -= choose(p, i);
            set |= 1u << p;
        }
        return set;
    }

    // Largest w in [i - 1, hi] with choose(w, i) <= x.
    static std::uint64_t largestWithChooseAtMost(std::uint64_t x, std::uint64_t i, std::uint64_t hi) noexcept
    {
        std::uint64_t lo = i - 1;
        while (lo < hi)
        {
            const std::uint64_t mid = lo + (hi - lo + 1) / 2;
            if (choose(mid, i) <= x)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }
        return lo;
    }

    std::uint32_t countOf(std::uint16_t config, std::size_t round) const noexcept
    {
        return (config >> (3 * (maxRounds - 1 - round))) & 7u;
    }

    std::uint64_t valuesFor(std::uint16_t config) const noexcept
    {
        std::uint64_t n = 1;
        std::uint32_t used = 0;
        for (std::size_t r = 0; r < m_rounds; ++r)
        {
            n *= choose(13 - used, countOf(config, r));
            used += countOf(config, r);
        }
        return n;
    }

    static std::size_t groupEnd(const std::array<Suit, 4> &suits, std::size_t begin) noexcept
    {
        std::size_t end = begin + 1;
        while (end < 4 && suits[end].config == suits[begin].config)
        {
            ++end;
        }
        return end;
    }

    void decodeSuit(std::uint16_t config, std::uint64_t value, std::size_t suit, Rounds &rounds) const noexcept
    {
        std::uint32_t used = 0;
        for (std::size_t r = 0; r < m_rounds; ++r)
        {
            const std::uint32_t count = countOf(config, r);
            const std::uint64_t options = choose(13 - static_cast<std::uint64_t>(std::popcount(used)), count);
            const std::uint32_t ranks = _pdep_u32(colexUnindex(value % options, count), ~used & 0x1FFF);
            value /= options;
            rounds[r] |= static_cast<std::uint64_t>(ranks) << (13 * suit);
            used |= ranks;
        }
    }

    std::uint64_t patternSize(std::uint64_t key) const noexcept
    {
        std::array<Suit, 4> suits{};
        for (std::size_t s = 0; s < 4; ++s)
        {
            suits[s].config = static_cast<std::uint16_t>(key >> (16 * (3 - s)));
        }
        std::uint64_t size = 1;
        for (std::size_t begin = 0; begin < 4;)
        {
            const std::size_t end = groupEnd(suits, begin);
            size *= choose(valuesFor(suits[begin].config) + (end - begin) - 1, end - begin);
            begin = end;
        }
        return size;
    }

    // Every way to split each round's cards across the 4 suits, reduced to sorted config keys.
    void collectPatterns(std::size_t round, std::size_t suit, std::uint32_t placed,
                         std::array<std::array<std::uint8_t, maxRounds>, 4> &counts, std::vector<std::uint64_t> &keys) const
    {
        if (round == m_rounds)
        {
            std::array<std::uint16_t, 4> configs{};
            for (std::size_t s = 0; s < 4; ++s)
            {
                for (std::size_t r = 0; r < m_rounds; ++r)
                {
                    configs[s] = static_cast<std::uint16_t>(configs[s] | (counts[s][r] << (3 * (maxRounds - 1 - r))));
                }
            }
            std::sort(configs.begin(), configs.end(), std::greater<>{});
            keys.push_back((std::uint64_t{configs[0]} << 48) | (std::uint64_t{configs[1]} << 32) | (std::uint64_t{configs[2]} << 16) | configs[3]);
            return;
        }
        if (suit == 3)
        {
            counts[3][round] = static_cast<std::uint8_t>(m_sizes[round] - placed);
            collectPatterns(round + 1, 0, 0, counts, keys);
            return;
        }
        for (std::uint32_t c = 0; c + placed <= m_sizes[round]; ++c)
        {
            counts[suit][round] = static_cast<std::uint8_t>(c);
            collectPatterns(round, suit + 1, placed + c, counts, keys);
        }
    }
};
#endif // __POKER_CFR_HAND_INDEXER_HPP__
