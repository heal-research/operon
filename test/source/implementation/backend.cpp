// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "operon/core/dispatch.hpp"
#include "operon/interpreter/functions.hpp"

namespace Operon::Test {

namespace {
    using T = Operon::Scalar;
    constexpr std::size_t S = Dispatch::DefaultBatchSize<T>;

    struct alignas(32) Buf { std::array<T, S> v; };

    auto UlpDistance(float a, float b) -> int {
        if (std::isnan(a) && std::isnan(b)) { return 0; }
        if (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)) { return 0; }
        auto ua = std::bit_cast<uint32_t>(a);
        auto ub = std::bit_cast<uint32_t>(b);
        // two's complement trick for signed-magnitude integers
        if (ua >> 31U) { ua = 0x80000000U - ua; }
        if (ub >> 31U) { ub = 0x80000000U - ub; }
        return static_cast<int>(ua > ub ? ua - ub : ub - ua);
    }

    // Run fn on batches of input, compare element-wise against ref, return max ULP error.
    auto MaxUlpError(auto fn, auto ref, std::vector<T> const& inputs) -> int {
        auto nbatch = (inputs.size() + S - 1) / S;
        std::vector<Buf> src(nbatch), dst(nbatch);

        for (auto i = 0UL; i < inputs.size(); ++i) {
            src[i / S].v[i % S] = inputs[i];
        }
        // pad last batch with safe values
        for (auto i = inputs.size(); i < nbatch * S; ++i) {
            src[i / S].v[i % S] = T{0.5};
        }

        for (auto b = 0UL; b < nbatch; ++b) {
            fn(dst[b].v.data(), T{1}, src[b].v.data());
        }

        int max_ulp = 0;
        for (auto i = 0UL; i < inputs.size(); ++i) {
            auto got = dst[i / S].v[i % S];
            auto expected = static_cast<T>(ref(static_cast<double>(inputs[i])));
            max_ulp = std::max(max_ulp, UlpDistance(got, expected));
        }
        return max_ulp;
    }

    auto MakeInputs() -> std::vector<T> {
        std::vector<T> vals;
        // dense grid over [-10, 10]
        constexpr int N = 10000;
        for (int i = 0; i <= N; ++i) {
            vals.push_back(T(-10.0 + 20.0 * i / N));
        }
        // edge cases
        for (auto v : {0.0f, -0.0f, 0.0001f, -0.0001f, 0.0003f, -0.0003f,
                       0.0005f, 7.99f, -7.99f, 8.5f, -8.5f, 100.0f, -100.0f,
                       std::numeric_limits<T>::infinity(),
                       -std::numeric_limits<T>::infinity()}) {
            vals.push_back(v);
        }
        return vals;
    }
} // namespace

TEST_CASE("Backend Tanh ULP accuracy", "[backend][tanh]")
{
    auto inputs = MakeInputs();
    auto fn = Backend::Tanh<T, S>;

    int max_ulp = MaxUlpError(fn, [](double x) { return std::tanh(x); }, inputs);

    // Eigen's generic_fast_tanh_float claims ~2 ULP; allow up to 4 to be safe.
    INFO("Max ULP error vs std::tanh: " << max_ulp);
    CHECK(max_ulp <= 4);
}

} // namespace Operon::Test
