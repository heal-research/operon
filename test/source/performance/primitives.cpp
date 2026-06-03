// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <nanobench.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <new>
#include <numeric>
#include <ranges>

#include "operon/core/dispatch.hpp"
#include "operon/interpreter/functions.hpp"

namespace Operon::Test {

namespace nb = ankerl::nanobench;

namespace {
    using T = Operon::Scalar;
    constexpr std::size_t S = Dispatch::DefaultBatchSize<T>;
    constexpr std::size_t NBATCH = 4096;

    struct alignas(32) Buf { std::array<T, S> v; };

    using UnaryFn = void(*)(T*, T, T const*);

    auto MakeBuffers() {
        auto src = std::vector<Buf>(NBATCH);
        auto dst = std::vector<Buf>(NBATCH);
        for (auto b = 0UL; b < NBATCH; ++b) {
            for (auto i = 0UL; i < S; ++i) {
                // values in (0.1, 1.0] — safe for log/sqrt
                src[b].v[i] = static_cast<T>(0.1 + 0.9 * static_cast<double>(b * S + i) / static_cast<double>(NBATCH * S));
            }
        }
        return std::pair{src, dst};
    }

    void BenchFn(nb::Bench& b, std::string const& name, UnaryFn fn, std::vector<Buf>& src, std::vector<Buf>& dst) {
        b.batch(static_cast<double>(NBATCH * S));
        b.run(name, [&] {
            for (auto i = 0UL; i < NBATCH; ++i) {
                fn(dst[i].v.data(), T{1}, src[i].v.data());
            }
        });
    }
} // namespace

TEST_CASE("Primitive throughput", "[performance][primitives]")
{
    auto [src, dst] = MakeBuffers();

    nb::Bench b;
    b.title("Primitive throughput").relative(true).performanceCounters(true).minEpochIterations(100);

    struct Case { const char* name; UnaryFn fn; };
    std::array cases{
        Case{ "Exp",   Backend::Exp<T, S>  },
        Case{ "Log",   Backend::Log<T, S>  },
        Case{ "Sin",   Backend::Sin<T, S>  },
        Case{ "Cos",   Backend::Cos<T, S>  },
        Case{ "Sqrt",  Backend::Sqrt<T, S> },
        Case{ "Tanh",  Backend::Tanh<T, S> },
        Case{ "Abs",   Backend::Abs<T, S>  },
        Case{ "Square",Backend::Square<T,S>},
    };

    for (auto& [name, fn] : cases) {
        BenchFn(b, name, fn, src, dst);
    }
}

} // namespace Operon::Test
