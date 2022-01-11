// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "buildinfo.hpp"
#include "operon/core/version.hpp"

namespace Operon {

auto Version() -> std::string {
#if defined(CERES_TINY_SOLVER)
        constexpr auto solver_name = "tiny";
#else
        constexpr auto solver_name = "ceres";
#endif

#if defined(USE_SINGLE_PRECISION)
        constexpr auto precision = "single";
#else
        constexpr auto precision = "double";
#endif
        fmt::memory_buffer buf;// NOLINT
        fmt::format_to(buf, "operon rev. {} {} {} {}, timestamp {}\n", OPERON_REVISION, OPERON_BUILD, OPERON_PLATFORM, OPERON_ARCH, OPERON_BUILD_TIMESTAMP);// NOLINT
        fmt::format_to(buf, "{}-precision build using eigen {}" , precision, OPERON_EIGEN_VERSION);// NOLINT
        if (OPERON_CERES_VERSION) {// NOLINT
            fmt::format_to(buf, ", ceres {}", OPERON_CERES_VERSION);// NOLINT
        }// NOLINT
        fmt::format_to(buf, ", taskflow {}", OPERON_TF_VERSION);// NOLINT
        fmt::format_to(buf, "\n");
        return std::string(buf.begin(), buf.end());
}

} // namespace Operon
// NOLINTEND
