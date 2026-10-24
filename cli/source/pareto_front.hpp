// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_PARETO_FRONT_HPP
#define OPERON_CLI_PARETO_FRONT_HPP

#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/types.hpp"
#include <string>
#include <vector>

namespace Operon {

// Non-dominated members of `population`, computed from the fitness vectors (stored Rank is not consulted).
// Individuals with identical fitness are reported once (the first). Result is ordered by the first objective.
auto SelectParetoFront(Operon::Span<Individual const> population) -> std::vector<Individual const*>;

// Write the non-dominated individuals (SelectParetoFront) from `population` to a JSON file at `path`.
// Each entry contains the infix expression, tree length, weighted complexity (k),
// raw objective values, R²/MSE/NMSE/MAE on train+test, MDL, and FBF.
// If problem.LinearScalingEnabled() is true, a least-squares (a, b) fit is
// applied to each individual's train predictions before computing all metrics.
auto WriteParetoFront(std::string const& path, Operon::Span<Individual const> population, ScalarDispatch const& dtable,
    Problem const& problem) -> void;

} // namespace Operon
#endif
