#ifndef __POKER_CLASSIFICATION_RESULT_HPP__
#define __POKER_CLASSIFICATION_RESULT_HPP__
#include <compare>
#include "card_enums.hpp"
// Layout: [category index:4][primary ranks:13][kicker ranks:13].
// primary = ranks forming the made hand (pair/trips/quads, or the whole mask for
// straight/flush/high card); kickers = tie breakers. Compares as a single integer.
struct ClassificationResult
{
private:
    std::uint32_t m_mask;
public:
    inline constexpr ClassificationResult() noexcept = default;
    inline constexpr ClassificationResult(const Classification classification, const std::uint16_t primary, const std::uint16_t kickers = 0) noexcept
        : m_mask((static_cast<std::uint32_t>(getClassificationIndex(classification)) << 26) | (static_cast<std::uint32_t>(primary) << 13) | kickers) {}
    inline constexpr ClassificationResult(const Classification classification, const Rank primary, const Rank kickers = Rank{}) noexcept
        : ClassificationResult(classification, static_cast<std::uint16_t>(primary), static_cast<std::uint16_t>(kickers)) {}
    inline constexpr Classification getClassification() const noexcept
    {
        return static_cast<Classification>(1u << (m_mask >> 26));
    }
    inline constexpr Rank getPrimary() const noexcept
    {
        return static_cast<Rank>((m_mask >> 13) & 0x1FFF);
    }
    inline constexpr Rank getKickers() const noexcept
    {
        return static_cast<Rank>(m_mask & 0x1FFF);
    }
    inline constexpr auto operator<=>(const ClassificationResult &) const noexcept = default;
};
inline void printRanks(std::ostream &os, Rank ranks)
{
    auto bits = static_cast<std::uint32_t>(ranks);
    while (bits)
    {
        os << static_cast<Rank>(1u << std::countr_zero(bits));
        bits &= bits - 1;
        if (bits)
        {
            os << ' ';
        }
    }
}
inline std::ostream &operator<<(std::ostream &os, const ClassificationResult result)
{
    os << result.getClassification() << ": ";
    printRanks(os, result.getPrimary());
    if (result.getKickers() != Rank{})
    {
        os << " + ";
        printRanks(os, result.getKickers());
    }
    return os;
}
#endif // __POKER_CLASSIFICATION_RESULT_HPP__
