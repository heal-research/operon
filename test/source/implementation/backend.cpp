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
        Case{ "Exp",    Backend::Exp<T,S>,
              Linspace(-87, 88, N, {0.f, -0.f}),
              [](double x){ return std::exp(x); },   2 },

        Case{ "Log",    Backend::Log<T,S>,
              Linspace(1e-6, 1e6, N, {1.f, Inf}),
              [](double x){ return std::log(x); },   2 },

        Case{ "Log1p",  Backend::Log1p<T,S>,
              Linspace(-0.999, 1e6, N, {0.f, Inf}),
              [](double x){ return std::log1p(x); }, 2 },

        Case{ "Logabs", Backend::Logabs<T,S>,
              Linspace(-1e6, 1e6, N, {1.f, -1.f, Inf, -Inf}),
              [](double x){ return std::log(std::abs(x)); }, 2 },

        Case{ "Sin",    Backend::Sin<T,S>,
              Linspace(-10, 10, N, {0.f, -0.f, Inf, -Inf}),
              [](double x){ return std::sin(x); },   2 },

        Case{ "Cos",    Backend::Cos<T,S>,
              Linspace(-10, 10, N, {0.f, Inf, -Inf}),
              [](double x){ return std::cos(x); },   2 },

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

} // namespace Operon::Test
