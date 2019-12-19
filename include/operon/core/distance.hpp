#ifndef OPERON_DISTANCE_HPP
#define OPERON_DISTANCE_HPP

#include "types.hpp"

#include <Eigen/Core>

namespace Operon {
namespace Distance {

    using HashVector = std::vector<Operon::Hash, Eigen::aligned_allocator<Operon::Hash>>;

    namespace {
        constexpr int shift_one { _MM_SHUFFLE(0, 3, 2, 1) };
        constexpr int shift_two { _MM_SHUFFLE(1, 0, 3, 2) };
        constexpr int shift_thr { _MM_SHUFFLE(2, 1, 0, 3) };

        static inline bool _mm256_is_zero(__m256i m) noexcept { return _mm256_testz_si256(m, m); }

        static inline bool NullIntersectProbe(Operon::Hash const* lhs, Operon::Hash const* rhs) noexcept
        {
            __m256i a { _mm256_load_si256((__m256i*)lhs) };
            __m256i b { _mm256_load_si256((__m256i*)rhs) };

            __m256i r0 { _mm256_cmpeq_epi64(a, b) };
            if (!_mm256_is_zero(r0))
                return false;

            __m256i r1 { _mm256_cmpeq_epi64(a, _mm256_permute4x64_epi64(b, shift_one)) };
            if (!_mm256_is_zero(r1))
                return false;

            __m256i r2 { _mm256_cmpeq_epi64(a, _mm256_permute4x64_epi64(b, shift_two)) };
            if (!_mm256_is_zero(r2))
                return false;

            __m256i r3 { _mm256_cmpeq_epi64(a, _mm256_permute4x64_epi64(b, shift_thr)) };
            return _mm256_is_zero(r3);
        }

        static inline bool IsSet(HashVector const& vec)
        {
            return std::adjacent_find(vec.begin(), vec.end(), std::equal_to {}) == vec.end();
        }

        static size_t CountIntersectSIMD(HashVector const& lhs, HashVector const& rhs) noexcept
        {
            size_t count = 0;
            size_t i = 0;
            size_t j = 0;
            size_t ls = lhs.size();
            size_t rs = rhs.size();

            Operon::Hash const* p = lhs.data();
            Operon::Hash const* q = rhs.data();

            auto lt = (ls / 4) * 4;
            auto rt = (rs / 4) * 4;

            while (i < lt && j < rt) {
                if (NullIntersectProbe(p + i, q + j)) {
                    auto a = p[i + 3];
                    auto b = q[j + 3];
                    i += (a <= b) * 4;
                    j += (b <= a) * 4;
                } else {
                    break;
                }
            }

            auto lm = lhs.back();
            auto rm = rhs.back();

            while (i < ls && j < rs) {
                auto a = lhs[i];
                auto b = rhs[j];

                count += a == b;
                i += a <= b;
                j += b <= a;

                if (a > rm || b > lm) {
                    break;
                }
            }
            return count;
        }

        static size_t CountIntersect(HashVector const& lhs, HashVector const& rhs) noexcept
        {
            size_t count = 0;
            size_t i = 0;
            size_t j = 0;
            size_t ls = lhs.size();
            size_t rs = rhs.size();

            auto lm = lhs.back();
            auto rm = rhs.back();

            while (i != ls && j != rs) {
                auto a = lhs[i];
                auto b = rhs[j];

                count += a == b;
                i += a <= b;
                j += b <= a;

                if (a > rm || b > lm) {
                    break;
                }
            }
            return count;
        }

        static double Jaccard(HashVector const& lhs, HashVector const& rhs) noexcept
        {
            size_t c = CountIntersectSIMD(lhs, rhs);
            double n = lhs.size() + rhs.size() - c;
            return (n - c) / n;
        }

        static double SorensenDice(HashVector const& lhs, HashVector const& rhs) noexcept
        {
            double n = lhs.size() + rhs.size();
            size_t c = CountIntersectSIMD(lhs, rhs);
            return 1 - 2 * c / n;
        }
    }

} // namespace Distance
} // namespace Operon

#endif
