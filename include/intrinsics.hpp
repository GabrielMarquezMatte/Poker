#ifndef __POKER_INTRINSICS_HPP__
#define __POKER_INTRINSICS_HPP__
#include <cstdint>
#include <bit>
#include <immintrin.h>
template<typename T> requires std::is_unsigned_v<T>
static inline constexpr T pdep_software(T x, T mask) noexcept
{
    T res = 0;
    for (T m = mask; m; m &= m - 1)
    {
        T lowest = m & -static_cast<std::make_signed_t<T>>(m);
        if (x & 1)
        {
            res |= lowest;
        }
        x >>= 1;
    }
    return res;
}
inline constexpr std::uint64_t pdep(std::uint64_t x, std::uint64_t mask) noexcept
{
    if (!std::is_constant_evaluated())
    {
        return _pdep_u64(x, mask);
    }
    return pdep_software(x, mask);
}
inline constexpr std::uint32_t pdep(std::uint32_t x, std::uint32_t mask) noexcept
{
    if (!std::is_constant_evaluated())
    {
        return _pdep_u32(x, mask);
    }
    return pdep_software(x, mask);
}
inline constexpr std::uint16_t keepTopBits(std::uint16_t m, int keep) noexcept
{
    int n = std::popcount(m);
    
    // Fast paths for edge cases
    if (keep >= n) return m;
    if (keep <= 0) return 0;
    
    int excess = n - keep;
    std::uint32_t mask = (1u << keep) - 1u;
    return static_cast<std::uint16_t>(pdep(mask << excess, m));
}
#endif // __POKER_INTRINSICS_HPP__