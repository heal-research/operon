// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#include "operon/operators//local_search.hpp"

#include "operon/core/tree.hpp"
#include "operon/optimizer/optimizer.hpp"
#include <random>

namespace Operon {

auto CoefficientOptimizer::operator()(Operon::RandomGenerator& rng, Operon::Tree tree) const -> std::tuple<Operon::Tree, OptimizerSummary> {
    OptimizerSummary summary;
    auto const* optimizer = optimizer_.get();

    if (optimizer->Iterations() > 0) {
        summary = optimizer->Optimize(rng, tree);

        if (summary.Success) {
            tree.SetCoefficients(summary.FinalParameters);
        }
    }
    return {tree, summary};
}
} // namespace Operon
