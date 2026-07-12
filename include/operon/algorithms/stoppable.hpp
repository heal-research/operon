// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_STOPPABLE_HPP
#define OPERON_ALGORITHMS_STOPPABLE_HPP

#include <atomic>
#include <functional>
#include <version>

namespace Operon {

// std::move_only_function (P0288R9) is a very recent libc++ addition; some
// shipped toolchains (e.g. the libc++ bundled in current Apple SDKs) don't
// implement it yet even under -std=c++23, so fall back to std::function
// there - losing move-only-capture support, but keeping the build green.
// Prefer MoveOnlyFunction<Sig> over referencing either type directly.
#if defined(__cpp_lib_move_only_function)
template<typename Sig>
using MoveOnlyFunction = std::move_only_function<Sig>;
#else
template<typename Sig>
using MoveOnlyFunction = std::function<Sig>;
#endif

// Invoked once per generation (GeneticAlgorithmBase-derived algorithms) or
// once per budget level (GrammarEnumerationAlgorithm) by an algorithm's
// Run() to report progress. Returning true requests early termination; each
// algorithm's own stop condition ORs StopRequested() in alongside its own
// generation-count/budget, evaluator-budget, and time-limit checks.
// MoveOnlyFunction, not std::function directly: this is only ever called and
// moved (passed by value into Run(), never copied), so it permits move-only
// captures (e.g. a captured unique_ptr progress sink) that std::function
// couldn't hold, on toolchains where std::move_only_function is available.
// pyoperon binds this via an explicit std::function-taking lambda shim
// rather than nb::overload_cast directly, since nanobind has no built-in
// caster for move_only_function - see pyoperon's source/algorithm.cpp.
using ReportCallback = MoveOnlyFunction<bool()>;

// Shared stopRequested_ + StopRequested()/RequestStop() idiom for search
// drivers, whether population-based (GeneticAlgorithmBase) or not
// (GrammarEnumerationAlgorithm). Factored out because both independently
// need it and neither can inherit the other (one has a population/
// generation model, the other a bottom-up DP construction with none).
//
// std::atomic<bool> isn't copyable, so copy/assign are hand-written here
// once instead of being duplicated (and risking drift) in every derived
// class. Move is deleted for the same reason GeneticAlgorithmBase deletes
// it: there's no use case for moving a live search driver, and skipping it
// avoids having to define a moved-from state for the atomic.
class StoppableAlgorithm {
public:
    StoppableAlgorithm() = default;
    virtual ~StoppableAlgorithm() = default;

    StoppableAlgorithm(StoppableAlgorithm const& other)
        : stopRequested_(other.stopRequested_.load(std::memory_order_acquire))
    {
    }
    StoppableAlgorithm(StoppableAlgorithm&&) = delete;

    auto operator=(StoppableAlgorithm const& other) -> StoppableAlgorithm&
    {
        if (this == &other) { return *this; }
        stopRequested_.store(other.stopRequested_.load(std::memory_order_acquire), std::memory_order_release);
        return *this;
    }
    auto operator=(StoppableAlgorithm&&) -> StoppableAlgorithm& = delete;

    // Set by an algorithm's Run() when its ReportCallback returns true; each
    // algorithm's own stop condition ORs this in. Atomic so it's also safe
    // to call RequestStop() from outside the callback (e.g. another thread,
    // a signal handler) while Run() is in progress.
    [[nodiscard]] auto StopRequested() const -> bool { return stopRequested_.load(std::memory_order_acquire); }
    auto RequestStop() -> void { stopRequested_.store(true, std::memory_order_release); }

protected:
    // Not public: only a derived class's own Reset() (if it has one) should
    // decide when clearing this is safe - e.g. GeneticAlgorithmBase::Reset()
    // documents that it's only valid to call between runs.
    auto ClearStopRequested() -> void { stopRequested_.store(false, std::memory_order_release); }

private:
    std::atomic<bool> stopRequested_{false};
};

} // namespace Operon

#endif
