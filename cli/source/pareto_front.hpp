// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CLI_PARETO_FRONT_HPP
#define OPERON_CLI_PARETO_FRONT_HPP

#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/types.hpp"
#include <string>

namespace Operon {

// Write the rank-0 individuals from `population` to a JSON file at `path`.
// Each entry contains the infix expression, tree length, weighted complexity (k),
// raw objective values, R²/MSE/NMSE/MAE on train+test, MDL, and FBF.
// If `linearScaling` is true, a least-squares (a, b) fit is applied to each
// individual's train predictions before computing all metrics.
auto WriteParetoFront(std::string const& path,
                      Operon::Span<Individual const> population,
                      ScalarDispatch const& dtable,
                      Problem const& problem,
                      bool linearScaling) -> void;

} // namespace Operon
#endif
