// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LOCAL_SEARCH_HPP
#define OPERON_LOCAL_SEARCH_HPP

#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/core/operator.hpp"
#include "operon/operon_export.hpp"
#include "operon/optimizer/optimizer.hpp"
#include <gsl/pointers>

namespace Operon {

class OPERON_EXPORT CoefficientOptimizer : public OperatorBase<std::tuple<Operon::Tree, FitOutcome>, Operon::Tree> {
public:
    explicit CoefficientOptimizer(gsl::not_null<OptimizerBase const*> optimizer) : optimizer_(optimizer) {}

    auto operator()(Operon::RandomGenerator& rng, Operon::Tree tree) const
        -> std::tuple<Operon::Tree, FitOutcome> override;

private:
    gsl::not_null<Operon::OptimizerBase const*> optimizer_;
};

// What one LocalSearch call did. `Outcome` is the complete optimizer result (diagnostics included) of an invoked
// optimizer, whatever its outcome (improvement, no improvement, typed failure); it is empty iff no optimizer was
// invoked (no optimizer, `pLocal` not positive, or the Bernoulli trial skipped the search), in which case no work was
// done and `OptimizerTimeUs` is zero. `OriginalCoefficients` is set iff a non-Lamarckian update must be undone
// by the caller after scoring the optimized genotype. `OptimizerTimeUs` is the wall time of the optimizer call.
struct LocalSearchResult {
    std::optional<std::vector<Operon::Scalar>> OriginalCoefficients;
    std::optional<FitOutcome> Outcome;
    std::uint64_t OptimizerTimeUs {};
};

// A point-in-time reading of an OptimizerAccounting. Counts are saturating.
struct OptimizerWork {
    std::uint64_t Residual {}; // optimizer residual (function) evaluations
    std::uint64_t Jacobian {}; // optimizer Jacobian evaluations
    std::uint64_t TimeUs {}; // wall time spent in the optimizer, summed over threads

    // The part of the evaluation budget spent by coefficient optimization (saturating).
    [[nodiscard]] constexpr auto Evaluations() const -> std::uint64_t
    {
        return detail::SaturatingAdd(Residual, Jacobian);
    }
};

// Run-level reduction of per-fit optimizer diagnostics (FitOutcome stays the authoritative per-fit record). It is
// owned by whoever orchestrates a run (GeneticAlgorithmBase), never by an evaluator, an optimizer or a generator:
// those are borrowed and may be shared between runs. It is passed explicitly to every path that can invoke local
// search (ScoreIndividual, offspring generators, cold initialization), so an invoked optimizer cannot silently drop
// its work. Recording is lock-free and may be called from any number of worker threads; counts saturate instead of
// wrapping. A reading taken while workers record is a non-atomic cross-counter snapshot; one taken after the workers
// finished is exact.
class OPERON_EXPORT OptimizerAccounting {
public:
    OptimizerAccounting() = default;
    // Copies a quiescent accounting (no concurrent Record/Reset), so that an algorithm that owns one stays copyable.
    OptimizerAccounting(OptimizerAccounting const& other) { *this = other; }
    auto operator=(OptimizerAccounting const& other) -> OptimizerAccounting&;
    ~OptimizerAccounting() = default;
    OptimizerAccounting(OptimizerAccounting&&) = delete;
    auto operator=(OptimizerAccounting&&) -> OptimizerAccounting& = delete;

    // Publishes one optimizer invocation. Negative counters of a diagnostics record (the signed fit fields) count as
    // zero. An invocation that performed no work contributes zero.
    auto Record(FitDiagnostics const& diagnostics, std::uint64_t elapsedUs) -> void;

    // Publishes the optimizer invocation of a LocalSearch call, if any: a skipped search contributes nothing.
    auto Record(LocalSearchResult const& search) -> void;

    [[nodiscard]] auto Snapshot() const -> OptimizerWork;

    // Optimizer residual + Jacobian evaluations (saturating): the optimizer's share of the evaluation budget.
    [[nodiscard]] auto Evaluations() const -> std::uint64_t { return Snapshot().Evaluations(); }

    auto Reset() -> void;

private:
    std::atomic<std::uint64_t> residual_ { 0 };
    std::atomic<std::uint64_t> jacobian_ { 0 };
    std::atomic<std::uint64_t> timeUs_ { 0 };
};

// Optionally runs local search (coefficient optimization) on `ind`'s genotype with probability `pLocal`; does not
// score it and does not touch any counter: the caller publishes the returned optimizer work to the accounting of
// its run (OptimizerAccounting::Record) before anything that can fail or discard the candidate. Non-Lamarckian
// updates return the original coefficients so the caller can evaluate the optimized genotype, then restore them.
// Consumes random numbers in the order: local-search trial, optimizer, Lamarckian trial.
OPERON_EXPORT auto LocalSearch(Operon::RandomGenerator& random, Operon::Individual& ind,
    Operon::CoefficientOptimizer const* coeffOptimizer, double pLocal, double pLamarck) -> LocalSearchResult;

} // namespace Operon

#endif
