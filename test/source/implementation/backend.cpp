// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string_view>
#include <vector>

#include "operon/core/dispatch.hpp"
#include "operon/interpreter/functions.hpp"

#include <vdt/exp.h>
#include <vdt/log.h>
#include <vdt/sin.h>
#include <vdt/cos.h>
#include <vdt/tanh.h>

namespace Operon::Test {

namespace {
    using T = Operon::Scalar;
    constexpr std::size_t S = Dispatch::DefaultBatchSize<T>;

    struct alignas(32) Buf { std::array<T, S> v; };

    auto UlpDistance(T a, T b) -> uint64_t {
        if (std::isnan(a) && std::isnan(b)) { return 0; }
        if (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)) { return 0; }
        using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
        constexpr U SignBit = U{1} << (sizeof(U) * 8 - 1);
        auto ua = std::bit_cast<U>(a);
        auto ub = std::bit_cast<U>(b);
        if (ua & SignBit) { ua = SignBit - ua; }
        if (ub & SignBit) { ub = SignBit - ub; }
        return ua > ub ? ua - ub : ub - ua;
    }

    struct UlpResult { uint64_t max_ulp; T worst_input; T got; T expected; };

    auto MaxUlpError(auto fn, auto ref, std::vector<T> const& inputs) -> UlpResult {
        auto const nbatch = (inputs.size() + S - 1) / S;
        std::vector<Buf> src(nbatch), dst(nbatch);

        for (auto i = 0UL; i < inputs.size(); ++i) {
            src[i / S].v[i % S] = inputs[i];
        }
        for (auto i = inputs.size(); i < nbatch * S; ++i) {
            src[i / S].v[i % S] = T{0.5};
        }
        for (auto b = 0UL; b < nbatch; ++b) {
            fn(dst[b].v.data(), T{1}, src[b].v.data());
        }

        UlpResult res{0, T{0}, T{0}, T{0}};
        for (auto i = 0UL; i < inputs.size(); ++i) {
            auto got      = dst[i / S].v[i % S];
            auto expected = static_cast<T>(ref(static_cast<double>(inputs[i])));
            if (auto d = UlpDistance(got, expected); d > res.max_ulp) {
                res = {d, inputs[i], got, expected};
            }
        }
        return res;
    }

    // Generate N+1 evenly spaced values in [lo, hi] plus extra edge values.
    auto Linspace(double lo, double hi, int n, std::vector<T> extra = {}) -> std::vector<T> {
        std::vector<T> v;
        v.reserve(static_cast<std::size_t>(n + 1) + extra.size());
        for (int i = 0; i <= n; ++i) {
            v.push_back(static_cast<T>(lo + (hi - lo) * i / n));
        }
        for (auto x : extra) { v.push_back(x); }
        return v;
    }

    constexpr T   Inf = std::numeric_limits<T>::infinity();
    constexpr int N   = 10000;
} // namespace

TEST_CASE("Backend transcendental ULP accuracy", "[backend]")
{
    struct Case {
        std::string_view name;
        void(*fn)(T*, T, T const*);
        std::vector<T> inputs;
        double(*ref)(double);
        uint64_t max_ulp;
    };

    // clang-format off
    std::array cases {
        // Clamped at ±88.376; test only above -86 where outputs are normal floats
        // (below that, eve::ldexp flushes subnormals to 0 -- acceptable for GP use).
        Case{ "Exp",    Backend::Exp<T,S>,
              Linspace(-86, 88, N, {0.f, -0.f, 88.3f, -86.0f}),
              [](double x){ return std::exp(x); },   2 },

        // SQRTHF branch at x≈0.7071; near-1 and small positive values.
        Case{ "Log",    Backend::Log<T,S>,
              Linspace(1e-6, 1e6, N, {1.f, Inf,
                                      0.707106f, 0.707107f,  // near SQRTHF branch
                                      0.999f,    1.001f,     // near log(1)=0
                                      0.f}),                 // → -inf
              [](double x){ return std::log(x); },   2 },

        Case{ "Log1p",  Backend::Log1p<T,S>,
              Linspace(-0.999, 1e6, N, {0.f, Inf}),
              [](double x){ return std::log1p(x); }, 2 },

        Case{ "Logabs", Backend::Logabs<T,S>,
              Linspace(-1e6, 1e6, N, {1.f, -1.f, Inf, -Inf,
                                      0.707106f, -0.707106f}),
              [](double x){ return std::log(std::abs(x)); }, 2 },

        // FMA 3-part range reduction: valid to ~117435; fallback tested beyond.
        Case{ "Sin",    Backend::Sin<T,S>,
              Linspace(-100, 100, N, {0.f, -0.f, Inf, -Inf,
                                      117434.f, -117434.f,   // just inside FMA limit
                                      117436.f, -117436.f}), // fallback region
              [](double x){ return std::sin(x); },   4 },

        // FMA 3-part range reduction: valid to ~71476; fallback tested beyond.
        Case{ "Cos",    Backend::Cos<T,S>,
              Linspace(-100, 100, N, {0.f, Inf, -Inf,
                                      71475.f,  -71475.f,    // just inside FMA limit
                                      71477.f,  -71477.f}),  // fallback region
              [](double x){ return std::cos(x); },   4 },

        Case{ "Tan",    Backend::Tan<T,S>,
              // avoid singularities near ±π/2 + nπ
              Linspace(-1.5, 1.5, N, {0.f}),
              [](double x){ return std::tan(x); },   2 },

        Case{ "Asin",   Backend::Asin<T,S>,
              Linspace(-1, 1, N, {0.f, 1.f, -1.f}),
              [](double x){ return std::asin(x); },  2 },

        Case{ "Acos",   Backend::Acos<T,S>,
              Linspace(-1, 1, N, {0.f, 1.f, -1.f}),
              [](double x){ return std::acos(x); },  2 },

        Case{ "Atan",   Backend::Atan<T,S>,
              Linspace(-100, 100, N, {0.f, Inf, -Inf}),
              [](double x){ return std::atan(x); },  2 },

        Case{ "Sinh",   Backend::Sinh<T,S>,
              Linspace(-10, 10, N, {0.f, -0.f}),
              [](double x){ return std::sinh(x); },  2 },

        Case{ "Cosh",   Backend::Cosh<T,S>,
              Linspace(-10, 10, N, {0.f}),
              [](double x){ return std::cosh(x); },  2 },

        Case{ "Tanh",   Backend::Tanh<T,S>,
              Linspace(-10, 10, N, {0.f, -0.f, 0.0001f, -0.0001f,
                                     7.99f, -7.99f, 8.5f, -8.5f, Inf, -Inf}),
              [](double x){ return std::tanh(x); },  4 },

        Case{ "Sqrt",   Backend::Sqrt<T,S>,
              Linspace(0, 1e6, N, {0.f}),
              [](double x){ return std::sqrt(x); },  2 },

        Case{ "Sqrtabs",Backend::Sqrtabs<T,S>,
              Linspace(-1e6, 1e6, N, {0.f}),
              [](double x){ return std::sqrt(std::abs(x)); }, 2 },

        Case{ "Cbrt",   Backend::Cbrt<T,S>,
              Linspace(-1e6, 1e6, N, {0.f, -0.f, Inf, -Inf}),
              [](double x){ return std::cbrt(x); },  2 },
    };
    // clang-format on

    for (auto& [name, fn, inputs, ref, ulp_limit] : cases) {
        auto [max_ulp, worst, got, expected] = MaxUlpError(fn, ref, inputs);
        INFO(name << ": max ULP = " << max_ulp << " (limit " << ulp_limit
             << ") at x=" << worst << " got=" << got << " expected=" << expected);
        CHECK(max_ulp <= ulp_limit);
    }
}

TEST_CASE("Backend Pow/Powabs ULP accuracy", "[backend]")
{
    auto MaxUlpError2 = [](auto fn, auto ref, std::vector<T> const& xs, std::vector<T> const& ys) {
        auto n = std::min(xs.size(), ys.size());
        auto nbatch = (n + S - 1) / S;
        struct alignas(32) Buf2 { std::array<T, S> v; };
        std::vector<Buf2> sa(nbatch), sb(nbatch), dst(nbatch);
        for (auto i = 0UL; i < n; ++i) {
            sa[i/S].v[i%S] = xs[i]; sb[i/S].v[i%S] = ys[i];
        }
        for (auto i = n; i < nbatch * S; ++i) {
            sa[i/S].v[i%S] = T{1}; sb[i/S].v[i%S] = T{1};
        }
        for (auto b = 0UL; b < nbatch; ++b) {
            fn(dst[b].v.data(), T{1}, sa[b].v.data(), sb[b].v.data());
        }
        UlpResult res{0, T{0}, T{0}, T{0}};
        for (auto i = 0UL; i < n; ++i) {
            auto got      = dst[i/S].v[i%S];
            auto expected = static_cast<T>(ref(static_cast<double>(xs[i]), static_cast<double>(ys[i])));
            if (std::isnan(expected)) { continue; } // skip cases where reference is NaN (e.g. negative base, non-integer exponent)
            if (auto d = UlpDistance(got, expected); d > res.max_ulp) {
                res = {d, xs[i], got, expected};
            }
        }
        return res;
    };

    // Positive x (excluding 0 — FastPow(0,y) is a degenerate path not testing poly accuracy):
    // compare against std::pow for all y.
    auto xs_pos  = Linspace(0.01, 10.0, 200, {1.f, 2.f, 0.5f});
    auto ys_mixed = Linspace(-3, 3, 200, {0.f, 1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 0.5f, -0.5f});

    // Negative x with integer y only: std::pow is well-defined here, sign correction tested.
    std::vector<T> xs_neg_int, ys_neg_int;
    for (auto x : Linspace(-10, -0.01, 50)) {
        for (T y : {-3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f}) {
            xs_neg_int.push_back(x); ys_neg_int.push_back(y);
        }
    }

    // Powabs: any x (abs applied internally), any real y.
    auto xs_abs = Linspace(-10, 10, 200, {0.f, 1.f, -1.f});

    SECTION("Pow positive base") {
        auto ref = [](double x, double y){ return std::pow(x, y); };
        auto [max_ulp, worst, got, expected] = MaxUlpError2(Backend::Pow<T,S>, ref, xs_pos, ys_mixed);
        INFO("Pow(x>0): max ULP = " << max_ulp << " at x=" << worst
             << " got=" << got << " expected=" << expected);
        CHECK(max_ulp <= 8UL);
    }
    SECTION("Pow negative base integer exponent") {
        auto ref = [](double x, double y){ return std::pow(x, y); };
        auto [max_ulp, worst, got, expected] = MaxUlpError2(Backend::Pow<T,S>, ref, xs_neg_int, ys_neg_int);
        INFO("Pow(x<0,y int): max ULP = " << max_ulp << " at x=" << worst
             << " got=" << got << " expected=" << expected);
        CHECK(max_ulp <= 8UL);
    }
    SECTION("Powabs") {
        auto ref = [](double x, double y){ return std::pow(std::abs(x), y); };
        auto [max_ulp, worst, got, expected] = MaxUlpError2(Backend::Powabs<T,S>, ref, xs_abs, ys_mixed);
        INFO("Powabs: max ULP = " << max_ulp << " at x=" << worst
             << " got=" << got << " expected=" << expected);
        CHECK(max_ulp <= 8UL);
    }
}

TEST_CASE("Backend vs vdt ULP accuracy", "[backend]")
{
    // Compare our Eve SIMD implementations against vdt scalar implementations.
    // Both use Cephes-derived polynomials; differences arise from FMA vs non-FMA paths.
    // Expected max ULP is 0-1 for functions sharing identical coefficients.
    struct Case {
        std::string_view name;
        void(*fn)(T*, T, T const*);
        std::vector<T> inputs;
        float(*ref)(float);
        uint64_t max_ulp;
    };

    // clang-format off
    std::array cases {
        Case{ "Exp vs vdt",
              Backend::Exp<T,S>,
              Linspace(-86, 88, N, {0.f, -0.f, 88.3f, -86.f}),
              vdt::fast_expf, 1 },
        Case{ "Log vs vdt",
              Backend::Log<T,S>,
              Linspace(1e-6, 1e6, N, {1.f, 0.707106f, 0.707107f, 0.999f, 1.001f}),
              vdt::fast_logf, 1 },
        Case{ "Sin vs vdt",
              Backend::Sin<T,S>,
              Linspace(-100, 100, N, {0.f, -0.f}),
              vdt::fast_sinf, 1 },
        Case{ "Cos vs vdt",
              Backend::Cos<T,S>,
              Linspace(-100, 100, N, {0.f}),
              vdt::fast_cosf, 1 },
        // vdt fast_tanhf uses a different algorithm (argument-halving + Padé doubling)
        // vs Eigen's 13/6-degree rational polynomial — comparison not meaningful here.
    };
    // clang-format on

    for (auto& [name, fn, inputs, ref, ulp_limit] : cases) {
        auto [max_ulp, worst, got, expected] = MaxUlpError(fn, [&](double x){ return static_cast<double>(ref(static_cast<float>(x))); }, inputs);
        INFO(name << ": max ULP = " << max_ulp << " (limit " << ulp_limit
             << ") at x=" << worst << " got=" << got << " expected=" << expected);
        CHECK(max_ulp <= ulp_limit);
    }
}

} // namespace Operon::Test
