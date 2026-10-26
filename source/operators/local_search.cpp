// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators//local_search.hpp"

#include <algorithm>
#include <chrono>
#include <random>

#include "operon/core/tree.hpp"
#include "operon/optimizer/optimizer.hpp"

namespace Operon {

namespace {
    // Adds `value` to `counter`, saturating at the maximum instead of wrapping.
    auto SaturatingFetchAdd(std::atomic<std::uint64_t>& counter, std::uint64_t value) -> void
    {
        if (value == 0U) {
            return;
        }
        auto current = counter.load(std::memory_order_relaxed);
        while (!counter.compare_exchange_weak(
            current, detail::SaturatingAdd(current, value), std::memory_order_relaxed)) { }
    }

    auto NonNegative(int value) -> std::uint64_t { return static_cast<std::uint64_t>(std::max(value, 0)); }
} // namespace

auto OptimizerAccounting::operator=(OptimizerAccounting const& other) -> OptimizerAccounting&
{
    if (this != &other) {
        residual_.store(other.residual_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        jacobian_.store(other.jacobian_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        timeUs_.store(other.timeUs_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

auto OptimizerAccounting::Record(FitDiagnostics const& diagnostics, std::uint64_t elapsedUs) -> void
{
    SaturatingFetchAdd(residual_, NonNegative(diagnostics.FunctionEvaluations));
    SaturatingFetchAdd(jacobian_, NonNegative(diagnostics.JacobianEvaluations));
    SaturatingFetchAdd(timeUs_, elapsedUs);
}

auto OptimizerAccounting::Record(LocalSearchResult const& search) -> void
{
    if (search.Outcome) {
        Record(Diagnostics(*search.Outcome), search.OptimizerTimeUs);
    }
}

auto OptimizerAccounting::Snapshot() const -> OptimizerWork
{
    return OptimizerWork { .Residual = residual_.load(std::memory_order_relaxed),
        .Jacobian = jacobian_.load(std::memory_order_relaxed),
        .TimeUs = timeUs_.load(std::memory_order_relaxed) };
}

auto OptimizerAccounting::Reset() -> void
{
    residual_.store(0, std::memory_order_relaxed);
    jacobian_.store(0, std::memory_order_relaxed);
    timeUs_.store(0, std::memory_order_relaxed);
}

auto LocalSearch(Operon::RandomGenerator& random, Operon::Individual& ind,
    Operon::CoefficientOptimizer const* coeffOptimizer, double pLocal, double pLamarck) -> LocalSearchResult
{
    using BernoulliTrial = std::bernoulli_distribution;

    if (coeffOptimizer == nullptr || pLocal <= 0 || !BernoulliTrial { pLocal }(random)) {
        return {};
    }

    auto c = ind.Genotype.GetCoefficients(); // save original coefficients
    auto t0 = std::chrono::steady_clock::now();
    auto [optimizedTree, outcome] = (*coeffOptimizer)(random, std::move(ind.Genotype));
    auto t1 = std::chrono::steady_clock::now();
    ind.Genotype = std::move(optimizedTree);

    LocalSearchResult result;
    result.OptimizerTimeUs
        = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    result.Outcome = std::move(outcome);
    if (!BernoulliTrial { pLamarck }(random)) {
        result.OriginalCoefficients = std::move(c);
    }
    return result;
}

auto CoefficientOptimizer::operator()(Operon::RandomGenerator& rng, Operon::Tree tree) const
    -> std::tuple<Operon::Tree, FitOutcome>
{
    auto const* optimizer = optimizer_.get();

    if (optimizer->Iterations() > 0) {
        auto outcome = optimizer->Optimize(rng, tree);
        if (outcome) {
            tree.SetCoefficients(outcome->FinalParameters);
        }
        return { tree, outcome };
    }
    // Iterations() == 0: matches the previous default-constructed
    // OptimizerSummary (FinalCost == 0.0, Success == false) contract -
    // GrammarEnumerationAlgorithm::Run's EXPECT(Iterations() > 0) depends on
    // this remaining "unsuccessful", not on any specific cost value.
    return { tree, tl::unexpected(FitFailure {}) };
}
} // namespace Operon
