// Implementation from https://github.com/zekyll/OMPEval
#ifndef OMP_RANDOM_H
#define OMP_RANDOM_H
#include <cstdint>
#include <climits>
#include <bit>
#include <array>
namespace omp
{
    static inline constexpr std::uint64_t splitmix64(std::uint64_t &x)
    {
        std::uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // Fast 64-bit PRNG with a period of 2^128-1.
    class XoroShiro128Plus
    {
    public:
        typedef std::uint64_t result_type;

        constexpr XoroShiro128Plus(std::uint64_t seed)
        {
            std::uint64_t tmp = seed;
            mState[0] = splitmix64(tmp);
            mState[1] = splitmix64(tmp);
        }

        constexpr inline std::uint64_t operator()()
        {
            std::uint64_t s0 = mState[0];
            std::uint64_t s1 = mState[1];
            std::uint64_t result = s0 + s1;
            s1 ^= s0;
            mState[0] = std::rotl(s0, 55) ^ s1 ^ (s1 << 14);
            mState[1] = std::rotl(s1, 36);
            return result;
        }

        inline constexpr void seed(std::uint64_t s)
        {
            mState[0] = splitmix64(s);
            mState[1] = splitmix64(s);
        }

        static constexpr inline std::uint64_t min()
        {
            return 0;
        }

        static constexpr inline std::uint64_t max()
        {
            return ~(uint64_t)0;
        }

    private:
        std::array<std::uint64_t, 2> mState;
    };

    // Simple and fast uniform int distribution for small ranges. Has a bias similar to the classic modulo
    // method, but it's good enough for most poker simulations.
    template <typename T = unsigned, unsigned tBits = 21>
    class FastUniformIntDistribution
    {
        static_assert(tBits > 0 && tBits <= 32, "tBits must be 1..32 for this buffer logic");

    public:
        struct param_type
        {
            T min{}, max{};
            constexpr T diff() const { return max - min + 1; }
        };
        constexpr FastUniformIntDistribution() { init(0, 1); }
        constexpr FastUniformIntDistribution(T min, T max) { init(min, max); }
        constexpr void init(T min, T max)
        {
            mParams = {min, max};
            mBuffer = 0;
            mUsesLeft = 0;
        }
        template <class TRng>
        inline constexpr T operator()(TRng &rng, const param_type &p)
        {
            static_assert(sizeof(typename TRng::result_type) == sizeof(std::uint64_t), "Need 64-bit RNG");
            constexpr std::uint32_t chunks = 64 / tBits;
            if (mUsesLeft == 0)
            {
                mBuffer = rng();
                mUsesLeft = chunks;
            }
            std::uint64_t slice = mBuffer >> (64 - tBits);
            mBuffer <<= tBits;
            --mUsesLeft;
            std::uint64_t r = (slice * std::uint64_t(p.diff())) >> tBits;
            return T(p.min + r);
        }
        template <class TRng>
        inline constexpr T operator()(TRng &rng) { return operator()(rng, mParams); }
    private:
        std::uint64_t mBuffer{};
        std::uint32_t mUsesLeft{};
        param_type mParams{};
    };
}

#endif // OMP_RANDOM_H