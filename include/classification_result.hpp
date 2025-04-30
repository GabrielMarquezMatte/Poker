#ifndef __POKER_CLASSIFICATION_RESULT_HPP__
#define __POKER_CLASSIFICATION_RESULT_HPP__
#include "card_enums.hpp"
struct ClassificationResult
{
private:
    std::uint32_t m_mask;
public:
    inline constexpr ClassificationResult() noexcept = default;
    inline constexpr ClassificationResult(const Classification classification, const Rank rankFlag) noexcept : m_mask(static_cast<std::uint32_t>(classification) | static_cast<std::uint32_t>(rankFlag << 10)) {}
    inline constexpr Classification getClassification() const noexcept
    {
        return static_cast<Classification>(m_mask & 0x3FF);
    }
    inline constexpr Rank getRankFlag() const noexcept
    {
        return static_cast<Rank>(m_mask >> 10);
    }
    inline constexpr bool operator<(const ClassificationResult &other) const noexcept
    {
        Classification classification = getClassification();
        Classification otherClassification = other.getClassification();
        if (classification != otherClassification)
        {
            return classification < otherClassification;
        }
        return getRankFlag() < other.getRankFlag();
    }
    inline constexpr bool operator==(const ClassificationResult &other) const noexcept
    {
        return getClassification() == other.getClassification() && getRankFlag() == other.getRankFlag();
    }
    inline constexpr bool operator!=(const ClassificationResult &other) const noexcept
    {
        return !(*this == other);
    }
    inline constexpr bool operator>(const ClassificationResult &other) const noexcept
    {
        return other < *this;
    }
    inline constexpr bool operator<=(const ClassificationResult &other) const noexcept
    {
        return !(*this > other);
    }
    inline constexpr bool operator>=(const ClassificationResult &other) const noexcept
    {
        return !(*this < other);
    }
};
inline std::ostream &operator<<(std::ostream &os, const ClassificationResult result) noexcept
{
    os << result.getClassification() << ": ";
    int rankFlag = static_cast<int>(result.getRankFlag());
    while (rankFlag > 0)
    {
        int rank = std::countr_zero(static_cast<uint32_t>(rankFlag));
        rankFlag &= ~(1 << rank);
        os << static_cast<Rank>(1 << rank);
        if (rankFlag > 0)
        {
            os << ' ';
        }
    }
    return os;
}
#endif // __POKER_CLASSIFICATION_RESULT_HPP__