// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_STOPPABLE_HPP
#define OPERON_ALGORITHMS_STOPPABLE_HPP

#include <atomic>
#include <functional>

namespace Operon {

// Invoked once per generation (GeneticAlgorithmBase-derived algorithms) or
// once per budget level (GrammarEnumerationAlgorithm) by an algorithm's
// Run() to report progress. Returning true requests early termination; each
// algorithm's own stop condition ORs StopRequested() in alongside its own
// generation-count/budget, evaluator-budget, and time-limit checks.
using ReportCallback = std::function<bool()>;

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
