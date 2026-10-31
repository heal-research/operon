// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_BUDGET_LEDGER_HPP
#define OPERON_ALGORITHMS_BUDGET_LEDGER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/ga_base.hpp"
#include "operon/operators/evaluator.hpp"

namespace Operon::detail {

// All ledger arithmetic is on 64 bit integers and saturates (detail::SaturatingAdd and friends, evaluator.hpp): a
// saturated evaluator total or attempt sum is treated as exceeding every budget and cap (it is irreversible within
// the generation, like any other exceeded condition).

// The evaluation budget limits *committed* work. Initialization always commits; afterwards a generation commits iff
// its offspring pool is full, everything it spent still fits in what the budget has left and, for a generator with an
// attempt cap, it stayed within the cap. Anything else is discarded and ends the run with the last committed state.
//
// The commit decision is independent of the thread count and of the order in which workers run:
//  - what one pool slot spends (evaluations, attempts) is a function of its RNG stream and of the committed state
//    (parents, committed cache) alone, and a sum of integers does not depend on the order of the summands;
//  - the counters only grow while workers run, so "spent > remaining" or "attempts > cap" observed at any time during
//    the generation implies the same at its end. Workers use that to stop starting attempts early (Exceeded()); the
//    early-out can fire only for a generation that could not have committed anyway.
// The guarantee needs every counted cost to be a function of the slot alone, which a transposition cache with
// CacheVisibility::Immediate (hits depend on arrival order) does not provide.
//
// Attempt cap (offspring-selection pressure). One attempt is one invocation of the generator for a pool slot, whether
// it evaluates the offspring or finds it in the transposition cache, and whether it is accepted or rejected: the
// selection pressure of offspring selection is the number of candidates generated per parent. Because every attempt
// counts, the cap always advances, also when every candidate is a cache hit. The rule is the sequential one: slots
// are filled in index order by one worker, and an attempt may start only while the attempts made so far in this
// generation do not exceed the cap. The generation therefore commits iff its pool is full and every attempt started
// within the cap. All attempts but the very last one (the accepted attempt of the last pool slot) start before it, and
// the attempts so far grow monotonically, so the condition reduces to: the attempts of every slot except the last,
// plus the rejected attempts of the last slot, are at most the cap. Each term is a function of its slot alone, so the
// reduced sum is order independent and equals the sequential rule at any thread count; at one thread without a
// transposition cache (attempts = evaluator calls) it is exactly the earlier behavior (`calls / parents >
// MaxSelectionPressure` stops the generation). The final accepted attempt is not part of the sum: it started within
// the cap, and may end above it.
//
// The earlier stop was also checked between generations, with the count of the generation that had just ended: a
// committed generation whose total attempts (final attempt included) exceed the cap therefore ended the run. That is
// kept (AttemptCapReached(): the order independent total of all slots against the cap), so that the whole run, not
// only each generation, equals the earlier single-threaded behavior.
class BudgetLedger {
public:
    // The budget is consumed by two sources with distinct owners: the scoring work of the evaluator tree
    // (EvaluatorBase::Stats(), composite children and shape-structural work included) and the coefficient-optimizer
    // work the run recorded in its OptimizerAccounting. Both are borrowed and must outlive the ledger.
    BudgetLedger(EvaluatorBase const* evaluator, OptimizerAccounting const* optimizer, std::size_t budget)
        : evaluator_(evaluator)
        , optimizer_(optimizer)
        , budget_(budget)
    {
    }

    // Units of the evaluation budget consumed so far: scoring residual + Jacobian evaluations plus optimizer
    // residual + Jacobian evaluations. Saturating.
    [[nodiscard]] auto Consumed() const -> std::uint64_t
    {
        auto const [residual, jacobian, calls] = evaluator_->Stats();
        auto const scoring = SaturatingAdd(static_cast<std::uint64_t>(residual), static_cast<std::uint64_t>(jacobian));
        return SaturatingAdd(scoring, optimizer_->Evaluations());
    }

    // Start of Run(): everything counted from here on belongs to this run.
    auto Begin() -> void
    {
        baseline_ = Consumed();
        committed_ = 0;
        discarded_ = 0;
        generations_ = 0;
        end_ = RunEnd::None;
    }

    // Initialization (or the warm-start re-evaluation) always commits, even beyond the budget.
    auto CommitInitialization() -> void { committed_ = SaturatingSub(Consumed(), baseline_); }

    // A zero-cost evaluator may keep generating under a zero budget. Any positive budget that has been reached,
    // and every overshoot, stops new generations.
    [[nodiscard]] auto Exhausted() const -> bool { return committed_ > budget_ || (committed_ == budget_ && budget_ != 0); }

    [[nodiscard]] auto Remaining() const -> std::uint64_t { return committed_ < budget_ ? budget_ - committed_ : 0; }

    // Start of a generation attempt. `maxAttempts` is the generator's attempt cap, if any.
    auto BeginGeneration(std::optional<std::size_t> maxAttempts) -> void
    {
        generationStart_ = Consumed();
        maxAttempts_ = maxAttempts;
        slotAttempts_.store(0, std::memory_order_relaxed);
        finalAttempts_.store(0, std::memory_order_relaxed);
    }

    // Reports one finished attempt of a pool slot. The attempts the commit rule counts are added to the running sum:
    // every attempt of a slot but the last, and the rejected attempts of the last slot. The accepted attempt of the
    // last slot is kept apart (it only enters AttemptCapReached()).
    auto AddAttempt(bool lastSlot, bool accepted) -> void
    {
        if (!maxAttempts_) {
            return;
        }
        if (lastSlot && accepted) {
            finalAttempts_.store(1, std::memory_order_relaxed);
        } else {
            SaturatingFetchAdd(slotAttempts_, 1);
        }
    }

    // The current generation attempt spent more than the budget has left (or the evaluator total saturated).
    [[nodiscard]] auto BudgetExceeded() const -> bool
    {
        auto const consumed = Consumed();
        return consumed == kSaturated || SaturatingSub(consumed, generationStart_) > Remaining();
    }

    // The current generation attempt made more attempts than the cap allows.
    [[nodiscard]] auto AttemptCapExceeded() const -> bool
    {
        return maxAttempts_ && ExceedsCap(slotAttempts_.load(std::memory_order_relaxed));
    }

    // After a committed generation: its total attempts, the last accepted attempt included, exceed the cap. The run
    // ends (without a discard), as the single-threaded selection-pressure stop always did.
    [[nodiscard]] auto AttemptCapReached() const -> bool
    {
        return maxAttempts_
            && ExceedsCap(SaturatingAdd(
                slotAttempts_.load(std::memory_order_relaxed), finalAttempts_.load(std::memory_order_relaxed)));
    }

    // True when the current generation attempt can no longer be committed because of the budget or the attempt cap.
    // Safe to call concurrently with the workers; monotone while they run.
    [[nodiscard]] auto Exceeded() const -> bool { return BudgetExceeded() || AttemptCapExceeded(); }

    // Why the attempt cannot be committed (see DiscardReason): a resource limit if either limit was observed to be
    // exceeded, else the asynchronous cause. Meaningful only for a generation that is not committed.
    [[nodiscard]] auto Reason(bool stopRequested) const -> DiscardReason
    {
        if (Exceeded()) {
            return DiscardReason::ResourceLimit;
        }
        return stopRequested ? DiscardReason::StopRequested : DiscardReason::TimeLimit;
    }

    // The progress record of the discarded attempt (the work counts come from the caller).
    [[nodiscard]] auto Progress(std::size_t attempted, std::size_t completed, std::size_t poolSize,
        bool stopRequested) const -> OffspringProgress
    {
        return OffspringProgress { attempted, completed, poolSize, Reason(stopRequested), BudgetExceeded(),
            AttemptCapExceeded() };
    }

    // Why the run ended (BudgetReport().End); set once, by the loop condition that ends it.
    auto SetEnd(RunEnd end) -> void { end_ = end; }

    // The attempt is committed: its spending becomes committed work.
    auto CommitGeneration() -> void
    {
        committed_ = SaturatingAdd(committed_, SaturatingSub(Consumed(), generationStart_));
        ++generations_;
    }

    // The attempt is discarded: its spending is recorded as discarded work.
    auto DiscardGeneration() -> void
    {
        discarded_ = SaturatingAdd(discarded_, SaturatingSub(Consumed(), generationStart_));
    }

    [[nodiscard]] auto Report() const -> BudgetAccounting
    {
        return BudgetAccounting {
            .Budget = budget_,
            .GenerationsCommitted = generations_,
            .EvaluationsCommitted = committed_,
            .EvaluationsExecuted = SaturatingSub(Consumed(), baseline_),
            .EvaluationsDiscarded = discarded_,
            .BudgetOvershoot = committed_ > budget_ ? committed_ - budget_ : 0,
            .End = end_,
        };
    }

private:
    // A saturated sum counts as exceeding, also for a cap of the maximum value.
    [[nodiscard]] auto ExceedsCap(std::uint64_t attempts) const -> bool
    {
        return attempts == kSaturated || attempts > *maxAttempts_;
    }

    EvaluatorBase const* evaluator_;
    OptimizerAccounting const* optimizer_;
    std::size_t budget_;
    std::uint64_t baseline_ { 0 };
    std::uint64_t committed_ { 0 };
    std::uint64_t discarded_ { 0 };
    RunEnd end_ { RunEnd::None };
    std::size_t generations_ { 0 };
    std::uint64_t generationStart_ { 0 };
    std::optional<std::size_t> maxAttempts_;
    std::atomic<std::uint64_t> slotAttempts_ { 0 };
    std::atomic<std::uint64_t> finalAttempts_ { 0 };
};

// A population algorithm needs something to evolve: both sizes must be positive. Evaluation cost is defined by the
// evaluator, so initialization may consume zero work (for example, a structural evaluator) or more than the budget.
// The ledger records either case after initialization commits.
inline auto RequireViableConfiguration(GeneticAlgorithmConfig const& config) -> void
{
    if (config.PopulationSize == 0) {
        throw std::invalid_argument("population size must be at least 1");
    }
    if (config.PoolSize == 0) {
        throw std::invalid_argument("pool size must be at least 1");
    }
}

} // namespace Operon::detail

#endif
