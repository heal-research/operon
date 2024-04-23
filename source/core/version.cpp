// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <fmt/core.h>
#include <fmt/format.h>
#include <iterator>
#include <string>

#include "buildinfo.hpp"
#include "operon/core/version.hpp"

namespace Operon {

auto Version() -> std::string {
#if defined(USE_SINGLE_PRECISION)
        constexpr auto precision = "single";
#else
        constexpr auto precision = "double";
#endif
        fmt::memory_buffer buf;// NOLINT
        fmt::format_to(std::back_inserter(buf), "operon rev. {} {} {} {}, timestamp {}\n", OPERON_REVISION, OPERON_BUILD, OPERON_PLATFORM, OPERON_ARCH, OPERON_BUILD_TIMESTAMP);// NOLINT
        fmt::format_to(std::back_inserter(buf), "{}-precision build using eigen {}" , precision, OPERON_EIGEN_VERSION);// NOLINT
        if (OPERON_CERES_VERSION) {// NOLINT
            fmt::format_to(std::back_inserter(buf), ", ceres {}", OPERON_CERES_VERSION);// NOLINT
        }// NOLINT
        fmt::format_to(std::back_inserter(buf), ", taskflow {}", OPERON_TF_VERSION);// NOLINT
        fmt::format_to(std::back_inserter(buf), "\n");
        fmt::format_to(std::back_inserter(buf), "compiler: {} {}, flags: {}\n", OPERON_COMPILER_ID, OPERON_COMPILER_VERSION, OPERON_COMPILER_FLAGS);
        return { buf.begin(), buf.end() };
}

} // namespace Operon
// NOLINTEND
