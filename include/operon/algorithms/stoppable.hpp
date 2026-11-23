// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_STOPPABLE_HPP
#define OPERON_ALGORITHMS_STOPPABLE_HPP

#include <atomic>
#include <functional>
#include <version>

#include "stop_reason.hpp"

namespace Operon {

// std::move_only_function (P0288R9) is a very recent libc++ addition; some
// shipped toolchains (e.g. the libc++ bundled in current Apple SDKs) don't
// implement it yet even under -std=c++23, so fall back to std::function
// there - losing move-only-capture support, but keeping the build green.
// Prefer MoveOnlyFunction<Sig> over referencing either type directly.
#if defined(__cpp_lib_move_only_function)
template <typename Sig> using MoveOnlyFunction = std::move_only_function<Sig>;
#else
template <typename Sig> using MoveOnlyFunction = std::function<Sig>;
#endif

// Invoked once per generation (GeneticAlgorithmBase-derived algorithms) or
// once per budget level (GrammarEnumerationAlgorithm) by an algorithm's
// Run() to report progress. Returning true requests early termination
// (recorded as StopReason::Callback); each algorithm's own stop condition
// ORs StopRequested() in alongside its own generation-count and time-limit
// checks. (The evaluation budget of the GA drivers is not a polled stop
// condition: they apply it when they decide whether a generation commits,
// see GeneticAlgorithmConfig::Evaluations.)
// MoveOnlyFunction, not std::function directly: this is only ever called and
// moved (passed by value into Run(), never copied), so it permits move-only
// captures (e.g. a captured unique_ptr progress sink) that std::function
// couldn't hold, on toolchains where std::move_only_function is available.
// pyoperon binds this via an explicit std::function-taking lambda shim
// rather than nb::overload_cast directly, since nanobind has no built-in
// caster for move_only_function - see pyoperon's source/algorithm.cpp.
using ReportCallback = MoveOnlyFunction<bool()>;

// Shared stop mask + StopRequested()/RequestStop() idiom for search
// drivers, whether population-based (GeneticAlgorithmBase) or not
// (GrammarEnumerationAlgorithm). Factored out because both independently
// need it and neither can inherit the other (one has a population/
// generation model, the other a bottom-up DP construction with none).
//
// The mask holds every StopReason recorded so far: external requests
// (callback, signal, engine error) set through RequestStop(), and, for
// algorithms that have them, the limit conditions (generation, evaluation,
// pressure, time) they observed while running. All updates are a lock-free
// atomic fetch_or, so they may be issued concurrently by worker threads and
// from a signal handler. StopRequested() is only about the external
// requests: a run that merely hit its generation limit still reports false.
//
// std::atomic isn't copyable, so copy/assign are hand-written here
// once instead of being duplicated (and risking drift) in every derived
// class. Move is deleted for the same reason GeneticAlgorithmBase deletes
// it: there's no use case for moving a live search driver, and skipping it
// avoids having to define a moved-from state for the atomic.
class StoppableAlgorithm {
public:
    StoppableAlgorithm() = default;
    virtual ~StoppableAlgorithm() = default;

    StoppableAlgorithm(StoppableAlgorithm const& other) : stopMask_(other.stopMask_.load(std::memory_order_acquire)) {}
    StoppableAlgorithm(StoppableAlgorithm&&) = delete;

    auto operator=(StoppableAlgorithm const& other) -> StoppableAlgorithm&
    {
        if (this == &other) {
            return *this;
        }
        stopMask_.store(other.stopMask_.load(std::memory_order_acquire), std::memory_order_release);
        return *this;
    }
    auto operator=(StoppableAlgorithm&&) -> StoppableAlgorithm& = delete;

    // True once an external stop was requested (a report callback returned true, or RequestStop() was called
    // from another thread or a signal handler while Run() is in progress); each algorithm's own stop condition
    // ORs this in. Limit conditions recorded by the algorithm itself do not count.
    [[nodiscard]] auto StopRequested() const -> bool
    {
        return (stopMask_.load(std::memory_order_acquire) & RequestedStopMask) != 0;
    }

    // Asks the algorithm to stop and records why. `source` is one of Callback (the default, also what a report
    // callback returning true records), Signal or EngineError; any other value is recorded as Callback. Safe to
    // call from any thread and from a signal handler.
    auto RequestStop(StopReason source = StopReason::Callback) noexcept -> void
    {
        if ((MaskOf(source) & RequestedStopMask) == 0) {
            source = StopReason::Callback;
        }
        RecordStop(MaskOf(source));
    }

    // Every stop reason recorded since the last Reset(), as a mask (see stop_reason.hpp). After Run() returns
    // this is the set of conditions that were observed true while it ran.
    [[nodiscard]] auto StopReasons() const -> StopMask { return stopMask_.load(std::memory_order_acquire); }

protected:
    // Not public: only a derived class's own Reset() (if it has one) should
    // decide when clearing this is safe - e.g. GeneticAlgorithmBase::Reset()
    // documents that it's only valid to call between runs.
    auto ClearStopRequested() -> void { stopMask_.store(0, std::memory_order_release); }

    // ORs `mask` into the recorded stop reasons (atomic fetch_or). Used by RequestStop() and by algorithms that
    // record the limit conditions they observe.
    auto RecordStop(StopMask mask) noexcept -> void { stopMask_.fetch_or(mask, std::memory_order_acq_rel); }

private:
    std::atomic<StopMask> stopMask_ { 0 };
};

} // namespace Operon

#endif
