#ifndef __POKER_CLASSIFICATION_RESULT_HPP__
#define __POKER_CLASSIFICATION_RESULT_HPP__
#include "card_enums.hpp"
struct ClassificationResult
{
private:
    std::uint32_t m_mask;
public:
    inline constexpr ClassificationResult() = default;
    inline constexpr ClassificationResult(const Classification classification, const Rank rankFlag) : m_mask(static_cast<std::uint32_t>(classification) | static_cast<std::uint32_t>(rankFlag << 9)) {}
    inline constexpr Classification getClassification() const
    {
        return static_cast<Classification>(m_mask & 0x1FF);
    }
    inline constexpr Rank getRankFlag() const
    {
        return static_cast<Rank>(m_mask >> 9);
    }
    inline constexpr bool operator<(const ClassificationResult &other) const
    {
        Classification classification = getClassification();
        Classification otherClassification = other.getClassification();
        if (classification != otherClassification)
        {
            return static_cast<uint32_t>(classification) < static_cast<uint32_t>(otherClassification);
        }
        Rank rankFlag = getRankFlag();
        Rank otherRankFlag = other.getRankFlag();
        return static_cast<uint32_t>(rankFlag) < static_cast<uint32_t>(otherRankFlag);
    }
    inline constexpr bool operator==(const ClassificationResult &other) const
    {
        Classification classification = getClassification();
        Classification otherClassification = other.getClassification();
        if (classification != otherClassification)
        {
            return false;
        }
        Rank rankFlag = getRankFlag();
        Rank otherRankFlag = other.getRankFlag();
        return rankFlag == otherRankFlag;
    }
    inline constexpr bool operator!=(const ClassificationResult &other) const
    {
        return !(*this == other);
    }
    inline constexpr bool operator>(const ClassificationResult &other) const
    {
        return other < *this;
    }
    inline constexpr bool operator<=(const ClassificationResult &other) const
    {
        return !(*this > other);
    }
    inline constexpr bool operator>=(const ClassificationResult &other) const
    {
        return !(*this < other);
    }
};
inline std::ostream &operator<<(std::ostream &os, const ClassificationResult result)
{
    os << result.getClassification() << ": ";
    int rankFlag = static_cast<int>(result.getRankFlag());
    while (rankFlag > 0)
    {
        int rank = std::countr_zero(static_cast<uint32_t>(rankFlag));
        rankFlag &= ~(1 << rank);
        os << static_cast<Rank>(rank) << " ";
    }
    return os;
}
#endif // __POKER_CLASSIFICATION_RESULT_HPP__