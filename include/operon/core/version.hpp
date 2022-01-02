// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_VERSION_HPP
#define OPERON_VERSION_HPP

#include "buildinfo.hpp"

#include <chrono>
#include <string>
#include <fmt/format.h>

namespace Operon {
// NOLINTBEGIN
    static auto Version() -> std::string {

#if defined(CERES_TINY_SOLVER)
        //constexpr char solver_name[] = "tiny";
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
        if (OPERON_PYBIND11_VERSION) {// NOLINT
            fmt::format_to(buf, ", pybind11 {}", OPERON_PYBIND11_VERSION);// NOLINT
        }// NOLINT
        fmt::format_to(buf, "\n");
        return std::string(buf.begin(), buf.end());
    };
// NOLINTEND
} // namespace Operon

#endif
