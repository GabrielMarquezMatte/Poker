// Implementation from https://github.com/zekyll/OMPEval
#ifndef OMP_RANDOM_H
#define OMP_RANDOM_H
#include <cstdint>
#include <climits>
#include <bit>
namespace omp
{
    // Fast 64-bit PRNG with a period of 2^128-1.
    class XoroShiro128Plus
    {
    public:
        typedef std::uint64_t result_type;

        constexpr XoroShiro128Plus(std::uint64_t seed)
        {
            mState[0] = ~(mState[1] = seed);
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

        static constexpr inline std::uint64_t min()
        {
            return 0;
        }

        static constexpr inline std::uint64_t max()
        {
            return ~(uint64_t)0;
        }

    private:

        std::uint64_t mState[2];
    };

    // Simple and fast uniform int distribution for small ranges. Has a bias similar to the classic modulo
    // method, but it's good enough for most poker simulations.
    template <typename T = unsigned, unsigned tBits = 21> // 64 / 3
    class FastUniformIntDistribution
    {
    public:
        struct param_type
        {
            T min;
            T max;
            constexpr param_type() : min(0), max(0) {}
            constexpr param_type(T min, T max) : min(min), max(max) {}
            inline constexpr T diff() const
            {
                return max - min + 1;
            }
        };
        typedef T TUnsigned;
        constexpr FastUniformIntDistribution()
        {
            init(0, 1);
        }
        constexpr FastUniformIntDistribution(T min, T max)
        {
            init(min, max);
        }
        inline constexpr void init(T min, T max)
        {
            mParams = param_type(min, max);
            mBuffer = 0;
            mBufferUsesLeft = 0;
        }
        template <class TRng>
        inline constexpr T operator()(TRng &rng, const param_type &params)
        {
            static_assert(sizeof(typename TRng::result_type) == sizeof(uint64_t), "64-bit RNG required.");
            constexpr std::uint64_t fullCount = sizeof(mBuffer) * CHAR_BIT / tBits;
            bool doRefill = mBufferUsesLeft == 0;
            mBufferUsesLeft = (doRefill ? fullCount : mBufferUsesLeft) - 1;
            mBuffer = doRefill ? rng() : mBuffer;
            std::uint64_t res = ((std::uint64_t)((std::uint32_t)mBuffer & MASK) * params.diff()) >> tBits;
            mBuffer >>= tBits;
            return static_cast<T>(params.min + res);
        }
        template <class TRng>
        inline constexpr T operator()(TRng &rng)
        {
            return operator()(rng, mParams);
        }

    private:
        static constexpr std::uint32_t MASK = (1u << tBits) - 1;
        std::uint64_t mBuffer;
        std::uint32_t mBufferUsesLeft;
        param_type mParams;
    };
}

#endif // OMP_RANDOM_H