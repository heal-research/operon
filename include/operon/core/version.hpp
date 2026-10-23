// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_VERSION_HPP
#define OPERON_VERSION_HPP

#include "operon/operon_export.hpp"
#include <chrono>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace Operon {
// Build-time facts about the library, for machine-readable reporting; Version() is the free-text equivalent.
struct BuildInfo {
    std::string Revision; // git revision, "unknown" when it was not available at build time
    std::string BuildType;
    std::string Platform;
    std::string Arch;
    std::string CompilerId;
    std::string CompilerVersion;
    // Effective C++ flags of the built configuration: CMAKE_CXX_FLAGS, CMAKE_CXX_FLAGS_<CONFIG> of that configuration
    // only, and the compile options of the operon_operon target (e.g. -march). Flags that consumers add when compiling
    // against the library are not included.
    std::string CompilerFlags;
    std::string Precision; // "single" or "double"
    std::string EigenVersion;
    std::string TaskflowVersion;
    std::string IsaLevel; // x86-64 microarchitecture level ("x86-64-v1" .. "x86-64-v4"); empty on other architectures
    std::vector<std::string> IsaFeatures; // ISA extensions the library was compiled for
    bool Jit { false }; // asmjit-based JIT backend available
};

// NOLINTBEGIN
auto OPERON_EXPORT Version() -> std::string;
auto OPERON_EXPORT GetBuildInfo() -> BuildInfo;
} // namespace Operon

#endif
