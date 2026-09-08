// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LOCAL_SEARCH_HPP
#define OPERON_LOCAL_SEARCH_HPP

#include <gsl/pointers>
#include <tl/expected.hpp>
#include "operon/core/operator.hpp"
#include "operon/operon_export.hpp"
#include "operon/optimizer/population_local_search.hpp"


namespace Operon {

// forward declarations
class Tree;
class OptimizerBase;
struct FitResult;
struct EvaluatorBase;
struct FitFailure;

class OPERON_EXPORT CoefficientOptimizer : public OperatorBase<std::tuple<Operon::Tree, tl::expected<FitResult, FitFailure>>, Operon::Tree> {
public:
    explicit CoefficientOptimizer(gsl::not_null<OptimizerBase const*> optimizer)
        : optimizer_(optimizer)
    { }

    auto operator()(Operon::RandomGenerator& rng, Operon::Tree tree) const -> std::tuple<Operon::Tree, tl::expected<FitResult, FitFailure>> override;
    [[nodiscard]] auto Iterations() const -> std::size_t;

private:
    gsl::not_null<Operon::OptimizerBase const*> optimizer_;
};

// Applies an optional accelerator backend to the individuals selected by the
// same Bernoulli local-search policy as LocalSearch. Unsupported trees are an
// input error: a requested backend must never silently switch them to CPU.
// Non-Lamarckian improvements are recorded in originalCoefficients so callers
// can score the optimized tree, then restore inherited coefficients.
OPERON_EXPORT auto LocalSearchPopulation(
    Operon::Span<Operon::Individual> population,
    Operon::Span<Operon::RandomGenerator> random,
    Operon::EvaluatorBase const& evaluator,
    Operon::CoefficientOptimizer const* coefficientOptimizer,
    double pLocal,
    double pLamarck,
    Operon::PopulationLocalSearchBackend& backend,
    uint32_t maxIterations,
    Operon::Span<std::optional<std::vector<Operon::Scalar>>> originalCoefficients) -> void;

} // namespace Operon

#endif
