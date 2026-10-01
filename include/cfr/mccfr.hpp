#ifndef __POKER_CFR_MCCFR_HPP__
#define __POKER_CFR_MCCFR_HPP__
#include "../random.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <immintrin.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using CfrRng = omp::XoroShiro128Plus;

// Game contract for the solver. State is a cheap value type; actions are indices in [0, numActions(s)).
// utility(s, p) is the payoff for player p at a terminal state.
template <typename G>
concept CfrGame = requires(const typename G::State &s, std::size_t action, std::size_t player, CfrRng &rng) {
    { G::numPlayers } -> std::convertible_to<std::size_t>;
    { G::maxActions } -> std::convertible_to<std::size_t>;
    { G::initial() } -> std::same_as<typename G::State>;
    { G::isTerminal(s) } -> std::same_as<bool>;
    { G::utility(s, player) } -> std::same_as<double>;
    { G::isChance(s) } -> std::same_as<bool>;
    { G::sampleChance(s, rng) } -> std::same_as<typename G::State>;
    { G::currentPlayer(s) } -> std::same_as<std::size_t>;
    { G::numActions(s) } -> std::same_as<std::size_t>;
    { G::apply(s, action) } -> std::same_as<typename G::State>;
    { G::infosetKey(s) } -> std::same_as<std::uint64_t>;
};

// A game may keep the average strategy for some of its infosets only: averaged(s) says which and
// averagedKeys() lists their keys. A blueprint whose later streets are re-solved at play time needs the
// average preflop only, and the averages are half of its memory.
template <typename G>
concept PartlyAveraged = requires(const typename G::State &s) {
    { G::averaged(s) } -> std::same_as<bool>;
    { G::averagedKeys() } -> std::same_as<std::vector<std::uint64_t>>;
};

// External-sampling MCCFR over fixed-capacity open-addressing tables of float regrets and average
// strategies. train() may run concurrently from several threads: keys are claimed with CAS and values
// are updated with relaxed loads/stores, so racing updates can be lost but never tear (as in Pluribus).
// Linear CFR is applied in blocks: call discount(t / (t + 1)) after block t, with no training running.
// Optional regret-based pruning (Pluribus MCCFR-P): see enablePruning.
template <CfrGame G>
class Mccfr
{
public:
    using Strategy = std::array<double, G::maxActions>;

    explicit Mccfr(std::size_t capacity) : m_regrets(capacity), m_averages(averageCapacity(capacity)) {}

    void train(std::uint64_t iterations, CfrRng &rng)
    {
        for (std::uint64_t i = 0; i < iterations; ++i)
        {
            for (std::size_t p = 0; p < G::numPlayers; ++p)
            {
                const bool prune = m_pruneThreshold < 0.0f && uniform(rng) < m_pruneProbability;
                traverse(G::initial(), p, rng, prune);
            }
        }
        m_iterations.fetch_add(iterations, std::memory_order_relaxed);
    }

    // In `probability` of traversals the traverser skips actions whose regret is below `threshold` (< 0);
    // the rest explore everything so pruned actions can recover. Regrets are then floored slightly below
    // the threshold (Pluribus: threshold -300M, floor -310M). Not safe while training.
    void enablePruning(float threshold, double probability = 0.95) noexcept
    {
        m_pruneThreshold = threshold;
        m_regretFloor = threshold * (310.0f / 300.0f);
        m_pruneProbability = probability;
    }

    void discount(float factor) noexcept
    {
        ++m_discounts;
        for (Table *table : {&m_regrets, &m_averages})
        {
            for (std::size_t i = 0; i <= table->mask; ++i)
            {
                for (std::size_t a = 0; a < G::maxActions; ++a)
                {
                    table->slots[i].values[a] *= factor;
                }
            }
        }
    }

    // Average strategy; the current one for infosets whose average the game does not keep; uniform for
    // infosets never visited.
    Strategy averageStrategy(std::uint64_t key, std::size_t numActions) const noexcept
    {
        const std::uint64_t stored = storedKey(key);
        const Slot *slot = m_averages.find(stored);
        if constexpr (PartlyAveraged<G>)
        {
            if (slot == nullptr)
            {
                slot = m_regrets.find(stored);
            }
        }
        return normalize(slot == nullptr ? Values{} : load(slot->values), numActions);
    }

    inline std::size_t numInfosets() const noexcept { return m_regrets.used.load(std::memory_order_relaxed); }
    inline std::size_t capacity() const noexcept { return m_regrets.mask + 1; }
    inline std::size_t bytes() const noexcept { return (m_regrets.mask + 1 + m_averages.mask + 1) * sizeof(Slot); }
    inline std::uint64_t iterations() const noexcept { return m_iterations.load(std::memory_order_relaxed); }
    inline std::uint64_t discounts() const noexcept { return m_discounts; }

    // Occupied slots only: the regrets, then the averages. Not safe while training.
    bool save(const std::string &path) const
    {
        std::ofstream out(path, std::ios::binary);
        const std::uint64_t header[5] = {separateTables | G::maxActions, numInfosets(), iterations(), m_discounts,
                                         m_averages.used.load(std::memory_order_relaxed)};
        out.write(reinterpret_cast<const char *>(header), sizeof(header));
        for (const Table *table : {&m_regrets, &m_averages})
        {
            for (std::size_t i = 0; i <= table->mask; ++i)
            {
                const std::uint64_t key = table->slots[i].key.load(std::memory_order_relaxed);
                if (key != empty)
                {
                    out.write(reinterpret_cast<const char *>(&key), sizeof(key));
                    out.write(reinterpret_cast<const char *>(table->slots[i].values.data()), sizeof(Values));
                }
            }
        }
        return static_cast<bool>(out);
    }

    // Also reads files from before the averages had their own table (one record per infoset: key,
    // regrets, average), dropping the averages the game does not keep.
    bool load(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        std::uint64_t header[4]{};
        in.read(reinterpret_cast<char *>(header), sizeof(header));
        const bool together = header[0] == G::maxActions;
        if (!in || (!together && header[0] != (separateTables | G::maxActions)))
        {
            return false;
        }
        const auto readKey = [&]
        {
            std::uint64_t key = 0;
            in.read(reinterpret_cast<char *>(&key), sizeof(key));
            return key;
        };
        const auto readValues = [&](Values &values) { in.read(reinterpret_cast<char *>(values.data()), sizeof(Values)); };
        if (together)
        {
            std::unordered_set<std::uint64_t> averaged;
            if constexpr (PartlyAveraged<G>)
            {
                for (const std::uint64_t key : G::averagedKeys())
                {
                    averaged.insert(storedKey(key));
                }
            }
            const auto keepsAverage = [&](std::uint64_t key)
            {
                if constexpr (PartlyAveraged<G>)
                {
                    return averaged.contains(key);
                }
                else
                {
                    return true;
                }
            };
            for (std::uint64_t n = 0; n < header[1] && in; ++n)
            {
                const std::uint64_t key = readKey();
                readValues(m_regrets.findOrInsert(key).values);
                Values average{};
                readValues(average);
                if (keepsAverage(key))
                {
                    m_averages.findOrInsert(key).values = average;
                }
            }
        }
        else
        {
            std::uint64_t averages = 0;
            in.read(reinterpret_cast<char *>(&averages), sizeof(averages));
            for (const auto &[table, records] : {std::pair{&m_regrets, header[1]}, std::pair{&m_averages, averages}})
            {
                for (std::uint64_t n = 0; n < records && in; ++n)
                {
                    const std::uint64_t key = readKey();
                    readValues(table->findOrInsert(key).values);
                }
            }
        }
        m_iterations.store(header[2], std::memory_order_relaxed);
        m_discounts = header[3];
        return static_cast<bool>(in);
    }

private:
    static constexpr std::uint64_t empty = 0;
    static constexpr std::uint64_t separateTables = std::uint64_t{1} << 32; // file format flag, with maxActions
    using Values = std::array<float, G::maxActions>;
    // Slots whose size is a power of two never straddle a cache line (six actions: two slots per line).
    static constexpr std::size_t slotSize = sizeof(std::uint64_t) + sizeof(Values);
    struct alignas(std::has_single_bit(slotSize) ? std::min<std::size_t>(slotSize, 64) : alignof(std::uint64_t)) Slot
    {
        std::atomic<std::uint64_t> key{empty};
        Values values{};
    };

    // Open addressing over stored keys (see storedKey).
    struct Table
    {
        std::size_t mask;
        std::unique_ptr<Slot[]> slots;
        std::atomic<std::size_t> used{0};

        explicit Table(std::size_t capacity) : mask(std::bit_ceil(capacity) - 1), slots(std::make_unique<Slot[]>(mask + 1)) {}

        const Slot *find(std::uint64_t key) const noexcept
        {
            for (std::size_t i = key & mask, probes = 0; probes <= mask; i = (i + 1) & mask, ++probes)
            {
                const std::uint64_t k = slots[i].key.load(std::memory_order_acquire);
                if (k == key)
                {
                    return &slots[i];
                }
                if (k == empty)
                {
                    return nullptr;
                }
            }
            return nullptr;
        }

        Slot &findOrInsert(std::uint64_t key)
        {
            for (std::size_t i = key & mask, probes = 0; probes <= mask; i = (i + 1) & mask, ++probes)
            {
                std::uint64_t k = slots[i].key.load(std::memory_order_acquire);
                if (k == empty && slots[i].key.compare_exchange_strong(k, key, std::memory_order_acq_rel))
                {
                    used.fetch_add(1, std::memory_order_relaxed);
                    return slots[i];
                }
                if (k == key)
                {
                    return slots[i];
                }
            }
            std::fputs("Mccfr: infoset table full, raise capacity\n", stderr);
            std::abort();
        }
    };

    Table m_regrets;
    Table m_averages; // only the infosets the game averages, and of those only the ones an opponent reached
    std::atomic<std::uint64_t> m_iterations{0};
    std::uint64_t m_discounts = 0;
    float m_pruneThreshold = 0.0f; // 0 disables pruning
    float m_regretFloor = -std::numeric_limits<float>::infinity();
    double m_pruneProbability = 0.0;

    static std::size_t averageCapacity(std::size_t capacity)
    {
        if constexpr (PartlyAveraged<G>)
        {
            return 2 * G::averagedKeys().size();
        }
        else
        {
            return capacity;
        }
    }

    static inline bool averaged(const typename G::State &s) noexcept
    {
        if constexpr (PartlyAveraged<G>)
        {
            return G::averaged(s);
        }
        else
        {
            return true;
        }
    }

    // Slots hold a bijective mix of the game key (spreads small keys like Kuhn's); 0 marks an empty slot,
    // so the single key mixing to 0 shares a slot with the one mixing to 1 (probability 2^-64).
    static inline std::uint64_t storedKey(std::uint64_t key) noexcept
    {
        const std::uint64_t mixed = omp::splitmix64(key);
        return mixed == empty ? 1 : mixed;
    }

    static inline Values load(const Values &values) noexcept
    {
        Values out{};
        for (std::size_t a = 0; a < G::maxActions; ++a)
        {
            out[a] = std::atomic_ref<float>(const_cast<float &>(values[a])).load(std::memory_order_relaxed);
        }
        return out;
    }

    static inline void add(float &target, double delta, float floor = -std::numeric_limits<float>::infinity()) noexcept
    {
        std::atomic_ref<float> ref(target);
        ref.store(std::max(ref.load(std::memory_order_relaxed) + static_cast<float>(delta), floor), std::memory_order_relaxed);
    }

    static inline double uniform(CfrRng &rng) noexcept { return static_cast<double>(rng() >> 11) * 0x1.0p-53; }

    // Positive part normalized; uniform when nothing is positive. Serves as regret matching and averaging.
    static inline Strategy normalize(const Values &values, std::size_t n) noexcept
    {
        Strategy out{};
        double total = 0.0;
        for (std::size_t a = 0; a < n; ++a)
        {
            total += std::max(values[a], 0.0f);
        }
        for (std::size_t a = 0; a < n; ++a)
        {
            out[a] = total > 0.0 ? std::max(values[a], 0.0f) / total : 1.0 / static_cast<double>(n);
        }
        return out;
    }

    static inline std::size_t sample(const Strategy &sigma, std::size_t n, CfrRng &rng) noexcept
    {
        double r = uniform(rng);
        for (std::size_t a = 0; a + 1 < n; ++a)
        {
            r -= sigma[a];
            if (r < 0.0)
            {
                return a;
            }
        }
        return n - 1;
    }

    // Of a child of the traverser's node: the opponent acts there, so its average is written too.
    inline void prefetch(const typename G::State &s) const noexcept
    {
        if (!G::isTerminal(s) && !G::isChance(s))
        {
            const std::uint64_t key = storedKey(G::infosetKey(s));
            _mm_prefetch(reinterpret_cast<const char *>(&m_regrets.slots[key & m_regrets.mask]), _MM_HINT_T0);
            if (averaged(s))
            {
                _mm_prefetch(reinterpret_cast<const char *>(&m_averages.slots[key & m_averages.mask]), _MM_HINT_T0);
            }
        }
    }

    double traverse(const typename G::State &s, std::size_t traverser, CfrRng &rng, bool prune)
    {
        if (G::isTerminal(s))
        {
            return G::utility(s, traverser);
        }
        if (G::isChance(s))
        {
            return traverse(G::sampleChance(s, rng), traverser, rng, prune);
        }
        const std::size_t n = G::numActions(s);
        const std::uint64_t key = storedKey(G::infosetKey(s));
        Slot &slot = m_regrets.findOrInsert(key);
        const Values regrets = load(slot.values);
        const Strategy sigma = normalize(regrets, n);
        if (G::currentPlayer(s) != traverser)
        {
            if (averaged(s))
            {
                Slot &average = m_averages.findOrInsert(key);
                for (std::size_t a = 0; a < n; ++a)
                {
                    add(average.values[a], sigma[a]);
                }
            }
            return traverse(G::apply(s, sample(sigma, n, rng)), traverser, rng, prune);
        }
        std::array<bool, G::maxActions> explored{};
        bool any = false;
        for (std::size_t a = 0; a < n; ++a)
        {
            explored[a] = !prune || regrets[a] > m_pruneThreshold;
            any = any || explored[a];
        }
        // Build every explored child first and prefetch their slots so the table misses overlap.
        std::array<typename G::State, G::maxActions> children;
        for (std::size_t a = 0; a < n; ++a)
        {
            explored[a] = explored[a] || !any;
            if (explored[a])
            {
                children[a] = G::apply(s, a);
                prefetch(children[a]);
            }
        }
        Strategy values{};
        double nodeValue = 0.0;
        for (std::size_t a = 0; a < n; ++a)
        {
            if (explored[a])
            {
                values[a] = traverse(children[a], traverser, rng, prune);
                nodeValue += sigma[a] * values[a];
            }
        }
        for (std::size_t a = 0; a < n; ++a)
        {
            if (explored[a])
            {
                add(slot.values[a], values[a] - nodeValue, m_regretFloor);
            }
        }
        return nodeValue;
    }
};

// Single-threaded Linear CFR: `blocks` blocks of `perBlock` iterations, discounting after each.
template <CfrGame G>
void trainLinear(Mccfr<G> &solver, std::uint64_t blocks, std::uint64_t perBlock, CfrRng &rng)
{
    for (std::uint64_t t = 1; t <= blocks; ++t)
    {
        solver.train(perBlock, rng);
        solver.discount(static_cast<float>(t) / static_cast<float>(t + 1));
    }
}
#endif // __POKER_CFR_MCCFR_HPP__
