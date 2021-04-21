// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_VERSION_HPP
#define OPERON_VERSION_HPP

#include "buildinfo.hpp"

#include <chrono>
#include <string>
#include <fmt/format.h>

namespace Operon {
    static std::string Version() {

#if defined(CERES_TINY_SOLVER)
        constexpr char solver_name[] = "tiny";
#else
        constexpr char solver_name[] = "ceres";
#endif

#if defined(USE_SINGLE_PRECISION)
        constexpr char precision[] = "single";
#else
        constexpr char precision[] = "double";
#endif
        fmt::memory_buffer buf;
        fmt::format_to(buf, "operon rev. {} {} {} {}, timestamp {}\n", OPERON_REVISION, OPERON_BUILD, OPERON_PLATFORM, OPERON_ARCH, OPERON_BUILD_TIMESTAMP);
        fmt::format_to(buf, "{}-precision build using eigen {}" , precision, OPERON_EIGEN_VERSION);
        if (OPERON_CERES_VERSION) {
            fmt::format_to(buf, ", ceres {}", OPERON_CERES_VERSION);
        }
        fmt::format_to(buf, ", tbb {}" , OPERON_TBB_VERSION);
        if (OPERON_PYBIND11_VERSION) {
            fmt::format_to(buf, ", pybind11 {}", OPERON_PYBIND11_VERSION);
        }
        fmt::format_to(buf, "\n");
        return std::string(buf.begin(), buf.end());
    };
}

#endif
