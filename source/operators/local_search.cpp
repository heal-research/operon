// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/local_search.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

#include "operon/core/problem.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/population_encoding.hpp"
#include "operon/optimizer/optimizer.hpp"

namespace Operon {

auto CoefficientOptimizer::operator()(Operon::RandomGenerator& rng, Operon::Tree tree) const -> std::tuple<Operon::Tree, tl::expected<FitResult, FitFailure>> {
    auto const* optimizer = optimizer_.get();

    if (optimizer->Iterations() > 0) {
        auto outcome = optimizer->Optimize(rng, tree);
        if (outcome) {
            tree.SetCoefficients(outcome->FinalParameters);
        }
        return {tree, outcome};
    }
    // Iterations() == 0: matches the previous default-constructed
    // OptimizerSummary (FinalCost == 0.0, Success == false) contract -
    // GrammarEnumerationAlgorithm::Run's EXPECT(Iterations() > 0) depends on
    // this remaining "unsuccessful", not on any specific cost value.
    return {tree, tl::unexpected(FitFailure{})};
}

auto CoefficientOptimizer::Iterations() const -> std::size_t
{
    return optimizer_->Iterations();
}

auto LocalSearchPopulation(
    Operon::Span<Operon::Individual> population,
    Operon::Span<Operon::RandomGenerator> random,
    Operon::EvaluatorBase const& evaluator,
    Operon::CoefficientOptimizer const* coefficientOptimizer,
    double pLocal,
    double pLamarck,
    Operon::PopulationLocalSearchBackend& backend,
    uint32_t maxIterations,
    Operon::Span<std::optional<std::vector<Operon::Scalar>>> originalCoefficients) -> void
{
    if (population.size() != random.size() || population.size() != originalCoefficients.size()) {
        throw std::invalid_argument("population local search requires matching population, RNG, and result spans");
    }
    if (coefficientOptimizer == nullptr || pLocal <= 0 || maxIterations == 0) { return; }

    auto const* problem = evaluator.GetProblem();
    auto const range = problem->TrainingRange();
    auto const& variables = problem->GetInputs();
    if (variables.empty()) { throw std::invalid_argument("population local search requires at least one input variable"); }

    std::vector<std::size_t> selected;
    selected.reserve(population.size());
    for (std::size_t i = 0; i < population.size(); ++i) {
        if (!std::bernoulli_distribution{pLocal}(random[i]) || population[i].Genotype.CoefficientsCount() == 0) { continue; }
        if (!backend.Supports(population[i].Genotype, variables)) {
            throw std::invalid_argument("requested population local-search backend does not support a selected tree");
        }
        selected.push_back(i);
    }
    if (selected.empty()) { return; }

    auto encoded = PopulationOptimization::EncodePopulation(population, selected);
    if (!encoded) { throw std::invalid_argument("unable to encode population for local search"); }

    std::vector<Operon::Scalar> columns;
    columns.reserve(variables.size() * range.Size());
    for (auto const hash : variables) {
        auto const values = problem->GetDataset()->GetValues(hash).subspan(range.Start(), range.Size());
        columns.insert(columns.end(), values.begin(), values.end());
    }
    auto const target = problem->TargetValues(range);
    auto const weights = problem->Weights(range).value_or(Operon::Span<Operon::Scalar const>{});
    auto result = backend.Optimize(*encoded, variables, columns, variables.size(), range.Size(), target, weights, maxIterations);

    if (result.Status.size() != selected.size() || result.InitialCosts.size() != selected.size()
        || result.FinalCosts.size() != selected.size() || result.Iterations.size() != selected.size()
        || result.AcceptedSteps.size() != selected.size() || result.Coefficients.size() != encoded->Coefficients.size()) {
        throw std::runtime_error("population local-search backend returned an invalid result shape");
    }
    for (std::size_t i = 0; i < encoded->Trees.size(); ++i) {
        auto const& tree = encoded->Trees[i];
        auto const originalIndex = static_cast<std::size_t>(tree.OriginalIndex);
        if (result.Status[i] != PopulationLocalSearchStatus::Improved) {
            if (result.Status[i] != PopulationLocalSearchStatus::Retained) {
                throw std::runtime_error("population local-search backend rejected a selected tree");
            }
            continue;
        }
        if (!(std::isfinite(result.InitialCosts[i]) && std::isfinite(result.FinalCosts[i])
              && result.FinalCosts[i] < result.InitialCosts[i])) {
            throw std::runtime_error("population local-search backend violated the delivery contract");
        }
        auto coefficients = std::span{result.Coefficients}.subspan(tree.CoefficientOffset, tree.CoefficientCount);
        if (!std::bernoulli_distribution{pLamarck}(random[originalIndex])) {
            originalCoefficients[originalIndex] = population[originalIndex].Genotype.GetCoefficients();
        }
        population[originalIndex].Genotype.SetCoefficients(coefficients);
    }
}
} // namespace Operon
