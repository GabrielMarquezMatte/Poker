#ifndef __POKER_CFR_CARD_ABSTRACTION_HPP__
#define __POKER_CFR_CARD_ABSTRACTION_HPP__
#include "hand_indexer.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// Postflop bucket per (hole, board) for boards of 3..5 cards, ordered weakest to strongest.
// Tables are produced by src/generate_abstraction.cpp.
class CardAbstraction
{
public:
    static constexpr std::array<std::uint8_t, 2> flopRounds{2, 3};
    static constexpr std::array<std::uint8_t, 2> turnRounds{2, 4};
    static constexpr std::array<std::uint8_t, 2> riverRounds{2, 5};

    // [0] flop, [1] turn, [2] river; indexed by indexer(street).index({hole, board}).
    std::array<std::vector<std::uint8_t>, 3> buckets;

    CardAbstraction() : m_indexers{HandIndexer(flopRounds), HandIndexer(turnRounds), HandIndexer(riverRounds)} {}

    inline const HandIndexer &indexer(std::size_t street) const noexcept { return m_indexers[street]; }

    inline std::uint8_t bucket(std::uint64_t hole, std::uint64_t board) const noexcept
    {
        const std::size_t street = static_cast<std::size_t>(std::popcount(board)) - 3;
        return buckets[street][m_indexers[street].index({hole, board})];
    }

    bool save(const std::string &path) const
    {
        std::ofstream out(path, std::ios::binary);
        out.write(magic, sizeof(magic));
        for (const auto &table : buckets)
        {
            const std::uint64_t size = table.size();
            out.write(reinterpret_cast<const char *>(&size), sizeof(size));
            out.write(reinterpret_cast<const char *>(table.data()), static_cast<std::streamsize>(size));
        }
        return static_cast<bool>(out);
    }

    bool load(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        char header[sizeof(magic)]{};
        in.read(header, sizeof(header));
        if (!in || !std::equal(header, header + sizeof(header), magic))
        {
            return false;
        }
        for (std::size_t street = 0; street < 3; ++street)
        {
            std::uint64_t size = 0;
            in.read(reinterpret_cast<char *>(&size), sizeof(size));
            if (!in || size != m_indexers[street].size())
            {
                return false;
            }
            buckets[street].resize(size);
            in.read(reinterpret_cast<char *>(buckets[street].data()), static_cast<std::streamsize>(size));
        }
        return static_cast<bool>(in);
    }

private:
    static constexpr char magic[8] = {'P', 'K', 'A', 'B', 'S', 'T', '0', '1'};
    std::array<HandIndexer, 3> m_indexers;
};
#endif // __POKER_CFR_CARD_ABSTRACTION_HPP__
