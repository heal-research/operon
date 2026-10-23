// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <fmt/base.h>
#include <fmt/format.h>
#include <iterator>
#include <string>
#include <string_view>

#include "buildinfo.hpp"
#include "buildinfo_flags.hpp"
#include "operon/core/version.hpp"

namespace Operon {

namespace {
    // The generated flag text is assembled from several parts and can contain runs of blanks.
    auto CompilerFlags() -> std::string
    {
        std::string flags;
        std::string_view const raw { OPERON_COMPILER_FLAGS };
        for (auto const c : raw) {
            if (c == ' ' || c == '\t' || c == '\n') {
                if (!flags.empty() && flags.back() != ' ') {
                    flags.push_back(' ');
                }
            } else {
                flags.push_back(c);
            }
        }
        if (!flags.empty() && flags.back() == ' ') {
            flags.pop_back();
        }
        return flags;
    }
} // namespace

auto Version() -> std::string
{
#ifdef USE_SINGLE_PRECISION
    constexpr auto precision = "single";
#else
    constexpr auto precision = "double";
#endif
    fmt::memory_buffer buf; // NOLINT
    fmt::format_to(std::back_inserter(buf), "operon rev. {} {} {} {}, timestamp {}\n", OPERON_REVISION, OPERON_BUILD,
        OPERON_PLATFORM, OPERON_ARCH, OPERON_BUILD_TIMESTAMP); // NOLINT
    fmt::format_to(
        std::back_inserter(buf), "{}-precision build using eigen {}", precision, OPERON_EIGEN_VERSION); // NOLINT
    if (OPERON_CERES_VERSION) { // NOLINT
        fmt::format_to(std::back_inserter(buf), ", ceres {}", OPERON_CERES_VERSION); // NOLINT
    } // NOLINT
    fmt::format_to(std::back_inserter(buf), ", taskflow {}", OPERON_TF_VERSION); // NOLINT
    fmt::format_to(std::back_inserter(buf), "\n");
    fmt::format_to(std::back_inserter(buf), "compiler: {} {}, flags: {}\n", OPERON_COMPILER_ID, OPERON_COMPILER_VERSION,
        CompilerFlags());
    return { buf.begin(), buf.end() };
}

auto GetBuildInfo() -> BuildInfo
{
    BuildInfo info;
    info.Revision = std::string_view { OPERON_REVISION }.empty() ? "unknown" : OPERON_REVISION;
    info.BuildType = OPERON_BUILD;
    info.Platform = OPERON_PLATFORM;
    info.Arch = OPERON_ARCH;
    info.CompilerId = OPERON_COMPILER_ID;
    info.CompilerVersion = OPERON_COMPILER_VERSION;
    info.CompilerFlags = CompilerFlags();
#ifdef USE_SINGLE_PRECISION
    info.Precision = "single";
#else
    info.Precision = "double";
#endif
    info.EigenVersion = OPERON_EIGEN_VERSION;
    info.TaskflowVersion = OPERON_TF_VERSION;
#ifdef HAVE_ASMJIT
    info.Jit = true;
#endif

    // Compile-time ISA of this translation unit; the library is built with one -march for all of its sources.
#if defined(__x86_64__) || defined(_M_X64)
    // clang-format off
#if defined(__SSE4_2__)
    info.IsaFeatures.emplace_back("sse4.2");
#endif
#if defined(__AVX__)
    info.IsaFeatures.emplace_back("avx");
#endif
#if defined(__AVX2__)
    info.IsaFeatures.emplace_back("avx2");
#endif
#if defined(__FMA__)
    info.IsaFeatures.emplace_back("fma");
#endif
#if defined(__BMI2__)
    info.IsaFeatures.emplace_back("bmi2");
#endif
#if defined(__AVX512F__)
    info.IsaFeatures.emplace_back("avx512f");
#endif
    // clang-format on
    info.IsaLevel = "x86-64-v1";
#if defined(__SSE4_2__) && defined(__POPCNT__) && defined(__SSSE3__)
    info.IsaLevel = "x86-64-v2";
#endif
#if defined(__AVX2__) && defined(__FMA__) && defined(__BMI2__) && defined(__BMI__) && defined(__F16C__)                \
    && defined(__LZCNT__) && defined(__MOVBE__) && defined(__SSE4_2__)
    info.IsaLevel = "x86-64-v3";
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512CD__) && defined(__AVX512DQ__)                    \
    && defined(__AVX512VL__) && defined(__AVX2__) && defined(__FMA__)
    info.IsaLevel = "x86-64-v4";
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    info.IsaFeatures.emplace_back("neon");
#endif
    return info;
}

} // namespace Operon
// NOLINTEND
