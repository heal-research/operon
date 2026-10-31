// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_PARETO_FRONT_HPP
#define OPERON_CLI_PARETO_FRONT_HPP

#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include <string>
#include <vector>

namespace Operon {

// Compute the front from population fitness, ignoring stored Rank. Epsilon-equivalent individuals are grouped
// before ranking; whether epsilon affects dominance depends on the sorter. The CLI's rs and ms rank exactly.
// This is not NSGA2::Best(), which after a completed generation refers to parents plus offspring.
// Result is ordered by the first objective.
auto SelectParetoFront(Operon::Span<Individual const> population, NondominatedSorterBase const& sorter,
    Operon::Scalar eps) -> std::vector<Individual const*>;

// Write the non-dominated individuals (SelectParetoFront under `sorter` and `eps`) from `population` to a JSON file at
// `path`. Each entry contains the infix expression, tree length, weighted complexity (k),
// raw objective values, R²/MSE/NMSE/MAE on train+test, MDL, and FBF.
// If problem.LinearScalingEnabled() is true, a least-squares (a, b) fit is
// applied to each individual's train predictions before computing all metrics.
auto WriteParetoFront(std::string const& path, Operon::Span<Individual const> population, ScalarDispatch const& dtable,
    Problem const& problem, NondominatedSorterBase const& sorter, Operon::Scalar eps) -> void;

} // namespace Operon
#endif
