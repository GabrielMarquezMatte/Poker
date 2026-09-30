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

// External-sampling MCCFR over a fixed-capacity open-addressing table of float regrets.
// train() may run concurrently from several threads: keys are claimed with CAS and values are
// updated with relaxed loads/stores, so racing updates can be lost but never tear (as in Pluribus).
// Linear CFR is applied in blocks: call discount(t / (t + 1)) after block t, with no training running.
// Optional regret-based pruning (Pluribus MCCFR-P): see enablePruning.
template <CfrGame G>
class Mccfr
{
public:
    using Strategy = std::array<double, G::maxActions>;

    explicit Mccfr(std::size_t capacity) : m_mask(std::bit_ceil(capacity) - 1), m_slots(std::make_unique<Slot[]>(m_mask + 1)) {}

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
        for (std::size_t i = 0; i <= m_mask; ++i)
        {
            for (std::size_t a = 0; a < G::maxActions; ++a)
            {
                m_slots[i].regret[a] *= factor;
                m_slots[i].strategy[a] *= factor;
            }
        }
    }

    // Average strategy; uniform for infosets never visited.
    Strategy averageStrategy(std::uint64_t key, std::size_t numActions) const noexcept
    {
        const Slot *slot = find(key);
        return normalize(slot == nullptr ? std::array<float, G::maxActions>{} : load(slot->strategy), numActions);
    }

    inline std::size_t numInfosets() const noexcept { return m_used.load(std::memory_order_relaxed); }
    inline std::size_t capacity() const noexcept { return m_mask + 1; }
    inline std::uint64_t iterations() const noexcept { return m_iterations.load(std::memory_order_relaxed); }
    inline std::uint64_t discounts() const noexcept { return m_discounts; }

    // Occupied slots only. Not safe while training.
    bool save(const std::string &path) const
    {
        std::ofstream out(path, std::ios::binary);
        const std::uint64_t header[4] = {G::maxActions, numInfosets(), iterations(), m_discounts};
        out.write(reinterpret_cast<const char *>(header), sizeof(header));
        for (std::size_t i = 0; i <= m_mask; ++i)
        {
            const std::uint64_t key = m_slots[i].key.load(std::memory_order_relaxed);
            if (key != empty)
            {
                out.write(reinterpret_cast<const char *>(&key), sizeof(key));
                out.write(reinterpret_cast<const char *>(m_slots[i].regret.data()), sizeof(m_slots[i].regret));
                out.write(reinterpret_cast<const char *>(m_slots[i].strategy.data()), sizeof(m_slots[i].strategy));
            }
        }
        return static_cast<bool>(out);
    }

    bool load(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        std::uint64_t header[4]{};
        in.read(reinterpret_cast<char *>(header), sizeof(header));
        if (!in || header[0] != G::maxActions)
        {
            return false;
        }
        for (std::uint64_t n = 0; n < header[1]; ++n)
        {
            std::uint64_t key = 0;
            in.read(reinterpret_cast<char *>(&key), sizeof(key));
            Slot &slot = findOrInsert(key);
            in.read(reinterpret_cast<char *>(slot.regret.data()), sizeof(slot.regret));
            in.read(reinterpret_cast<char *>(slot.strategy.data()), sizeof(slot.strategy));
        }
        m_iterations.store(header[2], std::memory_order_relaxed);
        m_discounts = header[3];
        return static_cast<bool>(in);
    }

private:
    static constexpr std::uint64_t empty = 0;
    using Values = std::array<float, G::maxActions>;
    struct Slot
    {
        std::atomic<std::uint64_t> key{empty};
        Values regret{};
        Values strategy{};
    };

    std::size_t m_mask;
    std::unique_ptr<Slot[]> m_slots;
    std::atomic<std::size_t> m_used{0};
    std::atomic<std::uint64_t> m_iterations{0};
    std::uint64_t m_discounts = 0;
    float m_pruneThreshold = 0.0f; // 0 disables pruning
    float m_regretFloor = -std::numeric_limits<float>::infinity();
    double m_pruneProbability = 0.0;

    // Slots hold a bijective mix of the game key (spreads small keys like Kuhn's); 0 marks an empty slot,
    // so the single key mixing to 0 shares a slot with the one mixing to 1 (probability 2^-64).
    static inline std::uint64_t storedKey(std::uint64_t key) noexcept
    {
        const std::uint64_t mixed = omp::splitmix64(key);
        return mixed == empty ? 1 : mixed;
    }

    const Slot *find(std::uint64_t gameKey) const noexcept
    {
        const std::uint64_t key = storedKey(gameKey);
        for (std::size_t i = key & m_mask, probes = 0; probes <= m_mask; i = (i + 1) & m_mask, ++probes)
        {
            const std::uint64_t k = m_slots[i].key.load(std::memory_order_acquire);
            if (k == key)
            {
                return &m_slots[i];
            }
            if (k == empty)
            {
                return nullptr;
            }
        }
        return nullptr;
    }

    // Takes a stored (already mixed) key.
    Slot &findOrInsert(std::uint64_t key)
    {
        for (std::size_t i = key & m_mask, probes = 0; probes <= m_mask; i = (i + 1) & m_mask, ++probes)
        {
            std::uint64_t k = m_slots[i].key.load(std::memory_order_acquire);
            if (k == empty && m_slots[i].key.compare_exchange_strong(k, key, std::memory_order_acq_rel))
            {
                m_used.fetch_add(1, std::memory_order_relaxed);
                return m_slots[i];
            }
            if (k == key)
            {
                return m_slots[i];
            }
        }
        std::fputs("Mccfr: infoset table full, raise capacity\n", stderr);
        std::abort();
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

    inline void prefetch(const typename G::State &s) const noexcept
    {
        if (!G::isTerminal(s) && !G::isChance(s))
        {
            _mm_prefetch(reinterpret_cast<const char *>(&m_slots[storedKey(G::infosetKey(s)) & m_mask]), _MM_HINT_T0);
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
        Slot &slot = findOrInsert(storedKey(G::infosetKey(s)));
        const Values regrets = load(slot.regret);
        const Strategy sigma = normalize(regrets, n);
        if (G::currentPlayer(s) != traverser)
        {
            for (std::size_t a = 0; a < n; ++a)
            {
                add(slot.strategy[a], sigma[a]);
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
                add(slot.regret[a], values[a] - nodeValue, m_regretFloor);
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
