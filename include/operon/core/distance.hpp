#ifndef OPERON_DISTANCE_HPP
#define OPERON_DISTANCE_HPP

#include "types.hpp"

#include <immintrin.h>

#include <Eigen/Core>

namespace Operon {
namespace Distance {

    using HashVector = Operon::Vector<Operon::Hash>;

    namespace {
#if defined(__AVX2__) && defined(NDEBUG)
        static inline bool _mm256_is_zero(__m256i m) noexcept { return _mm256_testz_si256(m, m); }

        static inline bool NullIntersectProbe(Operon::Hash const* lhs, Operon::Hash const* rhs) noexcept
        {
            __m256i a { _mm256_load_si256((__m256i*)lhs) };
            __m256i b0 { _mm256_set1_epi64x(rhs[0]) };
            __m256i b1 { _mm256_set1_epi64x(rhs[1]) };
            __m256i b2 { _mm256_set1_epi64x(rhs[2]) };
            __m256i b3 { _mm256_set1_epi64x(rhs[3]) };

            __m256i r0 { _mm256_cmpeq_epi64(a, b0) };
            __m256i r1 { _mm256_cmpeq_epi64(a, b1) };
            __m256i r2 { _mm256_cmpeq_epi64(a, b2) };
            __m256i r3 { _mm256_cmpeq_epi64(a, b3) };

            return _mm256_is_zero(_mm256_or_si256(_mm256_or_si256(r0, r1), _mm256_or_si256(r2, r3)));
        }
#endif

        static size_t CountIntersect(HashVector const& lhs, HashVector const& rhs) noexcept
        {
            size_t count = 0;
            size_t i = 0;
            size_t j = 0;
            size_t ls = lhs.size();
            size_t rs = rhs.size();

#if defined(__AVX2__) && defined(NDEBUG)
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
#endif

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

        static inline double Jaccard(HashVector const& lhs, HashVector const& rhs) noexcept
        {
            size_t c = CountIntersect(lhs, rhs);
            size_t n = lhs.size() + rhs.size();
            return static_cast<double>(n - 2 * c) / static_cast<double>(n);
        }

        static inline double SorensenDice(HashVector const& lhs, HashVector const& rhs) noexcept
        {
            size_t n = lhs.size() + rhs.size();
            size_t c = CountIntersect(lhs, rhs);
            return 1.0 - 2.0 * static_cast<double>(c) / static_cast<double>(n);
        }
    }

} // namespace Distance
} // namespace Operon

#endif
