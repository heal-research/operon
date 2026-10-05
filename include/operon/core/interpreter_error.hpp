// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERPRETER_ERROR_HPP
#define OPERON_INTERPRETER_ERROR_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "operon/core/types.hpp"

namespace Operon {

struct InterpreterError {
    enum class Code : std::uint8_t {
        MissingVariable,
        MissingPrimitive,
        MissingDerivative,
        InvalidOutputSize,
        InvalidCoefficientSize,
        InvalidRootIndex,
    };

    Code Kind { Code::MissingVariable };
    Operon::Hash Hash {};
    std::size_t ExpectedSize {};
    std::size_t ActualSize {};
};

[[nodiscard]] inline auto FormatInterpreterError(InterpreterError const& error) -> std::string
{
    switch (error.Kind) {
    case InterpreterError::Code::MissingVariable:
        return fmt::format("missing dataset variable with hash {}", error.Hash);
    case InterpreterError::Code::MissingPrimitive:
        return fmt::format("missing primitive with hash {}", error.Hash);
    case InterpreterError::Code::MissingDerivative:
        return fmt::format("missing derivative for primitive with hash {}", error.Hash);
    case InterpreterError::Code::InvalidOutputSize:
        return fmt::format("invalid output size: expected {}, got {}", error.ExpectedSize, error.ActualSize);
    case InterpreterError::Code::InvalidCoefficientSize:
        return fmt::format("invalid coefficient size: expected {}, got {}", error.ExpectedSize, error.ActualSize);
    case InterpreterError::Code::InvalidRootIndex:
        return fmt::format("invalid root index: expected less than {}, got {}", error.ExpectedSize, error.ActualSize);
    }
    std::unreachable();
}

} // namespace Operon

#endif
