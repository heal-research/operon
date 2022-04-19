// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_CLI_OPERATOR_FACTORY_HPP
#define OPERON_CLI_OPERATOR_FACTORY_HPP

#include <stddef.h>                            // for size_t
#include <memory>                              // for unique_ptr, make_unique
#include <string>                              // for operator==, string
#include <utility>                             // for addressof
#include <vector>                              // for vector
#include "operon/core/types.hpp"               // for Span
#include "operon/core/individual.hpp"          // for Comparison
#include "operon/interpreter/interpreter.hpp"  // for Interpreter
#include "util.hpp"                            // for Split
namespace Operon { class EvaluatorBase; }
namespace Operon { class KeepBestReinserter; }
namespace Operon { class OffspringGeneratorBase; }
namespace Operon { class PrimitiveSet; }
namespace Operon { class Problem; }
namespace Operon { class ReinserterBase; }
namespace Operon { class ReplaceWorstReinserter; }
namespace Operon { class SelectorBase; }
namespace Operon { struct CreatorBase; }
namespace Operon { struct CrossoverBase; }
namespace Operon { struct ErrorMetric; }
namespace Operon { struct MutatorBase; }
namespace Operon { struct Variable; }

namespace Operon {

auto ParseReinserter(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<ReinserterBase>;

auto ParseSelector(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<SelectorBase>;

auto ParseCreator(std::string const& str, PrimitiveSet const& pset, Operon::Span<Variable const> inputs) -> std::unique_ptr<CreatorBase>;

auto ParseEvaluator(std::string const& str, Problem& problem, Interpreter& interpreter) -> std::unique_ptr<EvaluatorBase>;

auto ParseErrorMetric(std::string const& str) -> std::tuple<std::unique_ptr<Operon::ErrorMetric>, bool>;

auto ParseGenerator(std::string const& str, EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel) -> std::unique_ptr<OffspringGeneratorBase>;

} // namespace Operon

#endif
