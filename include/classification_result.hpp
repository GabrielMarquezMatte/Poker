#ifndef __POKER_CLASSIFICATION_RESULT_HPP__
#define __POKER_CLASSIFICATION_RESULT_HPP__
#include "card_enums.hpp"
struct ClassificationResult
{
    Classification classification;
    Rank rankFlag;
    inline constexpr ClassificationResult() = default;
    inline constexpr ClassificationResult(Classification classification, Rank rankFlag) : classification(classification), rankFlag(rankFlag) {}
    inline constexpr bool operator<(const ClassificationResult &other) const
    {
        if (classification != other.classification)
        {
            return static_cast<uint32_t>(classification) < static_cast<uint32_t>(other.classification);
        }
        return static_cast<uint32_t>(rankFlag) < static_cast<uint32_t>(other.rankFlag);
    }
    inline constexpr bool operator==(const ClassificationResult &other) const
    {
        return classification == other.classification && rankFlag == rankFlag;
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
inline std::ostream &operator<<(std::ostream &os, const ClassificationResult &result)
{
    static constexpr std::array<std::string_view, 10> classifications = {"High Card", "Pair", "Two Pair", "Three of a Kind",
                                                                         "Straight", "Flush", "Full House", "Four of a Kind",
                                                                         "Straight Flush", "Royal Flush"};
    os << classifications[static_cast<int>(result.classification)] << ": ";
    int rankFlag = static_cast<int>(result.rankFlag);
    while (rankFlag > 0)
    {
        int rank = std::countr_zero(static_cast<uint32_t>(rankFlag));
        rankFlag &= ~(1 << rank);
        os << static_cast<Rank>(rank) << " ";
    }
    return os;
}
#endif // __POKER_CLASSIFICATION_RESULT_HPP__