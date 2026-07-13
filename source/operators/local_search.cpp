// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators//local_search.hpp"

#include "operon/core/tree.hpp"
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
} // namespace Operon
