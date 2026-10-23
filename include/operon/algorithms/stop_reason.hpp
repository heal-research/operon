// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_STOP_REASON_HPP
#define OPERON_ALGORITHMS_STOP_REASON_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace Operon {

// Why a search driver stopped (or was asked to). Several can hold at the same time; a run records the set of all
// of them as a StopMask, and PrimaryStopReason() picks the one to report first.
enum class StopReason : std::uint8_t {
    GenerationLimit, // the configured number of generations was reached
    EvaluationLimit, // the evaluation budget (residual + Jacobian evaluations) is used up
    PressureLimit, // offspring selection exceeded its maximum selection pressure
    TimeLimit, // the time limit elapsed
    Callback, // the report callback asked to stop (or RequestStop() was called without another source)
    Signal, // an operating-system signal handler asked to stop
    EngineError, // the run is being abandoned because of an error
};

// One bit per StopReason (bit index = enumerator value).
using StopMask = std::uint32_t;

[[nodiscard]] constexpr auto MaskOf(StopReason reason) -> StopMask
{
    return StopMask { 1 } << static_cast<unsigned>(reason);
}

[[nodiscard]] constexpr auto Contains(StopMask mask, StopReason reason) -> bool { return (mask & MaskOf(reason)) != 0; }

// The reasons that are requests from outside the search loop (as opposed to limits the loop checks itself).
// StoppableAlgorithm::StopRequested() is true exactly when one of these is recorded.
inline constexpr StopMask RequestedStopMask
    = MaskOf(StopReason::Callback) | MaskOf(StopReason::Signal) | MaskOf(StopReason::EngineError);

// Reporting precedence (precedence_version 1). Earlier entries win when several reasons hold at once.
inline constexpr std::array<StopReason, 7> StopPrecedence {
    StopReason::EngineError,
    StopReason::Signal,
    StopReason::Callback,
    StopReason::EvaluationLimit,
    StopReason::PressureLimit,
    StopReason::GenerationLimit,
    StopReason::TimeLimit,
};
inline constexpr int StopPrecedenceVersion = 1;

// The highest-precedence reason in `mask`, or nullopt if the mask is empty.
[[nodiscard]] constexpr auto PrimaryStopReason(StopMask mask) -> std::optional<StopReason>
{
    for (auto const reason : StopPrecedence) {
        if (Contains(mask, reason)) {
            return reason;
        }
    }
    return std::nullopt;
}

// The snake_case names used in the machine-readable records (e.g. "evaluation_limit").
[[nodiscard]] constexpr auto ToString(StopReason reason) -> std::string_view
{
    switch (reason) {
    case StopReason::GenerationLimit:
        return "generation_limit";
    case StopReason::EvaluationLimit:
        return "evaluation_limit";
    case StopReason::PressureLimit:
        return "pressure_limit";
    case StopReason::TimeLimit:
        return "time_limit";
    case StopReason::Callback:
        return "callback";
    case StopReason::Signal:
        return "signal";
    case StopReason::EngineError:
        return "engine_error";
    }
    return "unknown";
}

} // namespace Operon

#endif
