// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CLI_OPERATOR_FACTORY_HPP
#define OPERON_CLI_OPERATOR_FACTORY_HPP

#include <cstddef>                            // for size_t
#include <memory>                              // for unique_ptr, make_unique
#include <string>                              // for operator==, string
#include <utility>                             // for addressof
#include <vector>                              // for vector
#include "operon/core/types.hpp"               // for Span
#include "operon/core/individual.hpp"          // for Comparison
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/interpreter/interpreter.hpp"  // for Interpreter
#include "operon/optimizer/optimizer.hpp"
#include "util.hpp"                            // for Split
namespace Operon { struct EvaluatorBase; }
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
namespace Operon { class CoefficientOptimizer; }
namespace Operon { struct MutatorBase; }
namespace Operon { struct Variable; }

namespace Operon {

auto ParseReinserter(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<ReinserterBase>;

auto ParseSelector(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<SelectorBase>;

auto ParseCreator(std::string const& str, PrimitiveSet const& pset, std::vector<Operon::Hash> const& inputs) -> std::unique_ptr<CreatorBase>;

auto ParseEvaluator(std::string const& str, Problem& problem, DefaultDispatch& dtable, bool scale = true) -> std::unique_ptr<EvaluatorBase>;

auto ParseErrorMetric(std::string const& str) -> std::tuple<std::unique_ptr<Operon::ErrorMetric>, bool>;

auto ParseGenerator(std::string const& str, EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel, CoefficientOptimizer const* cOpt) -> std::unique_ptr<OffspringGeneratorBase>;

auto ParseOptimizer(std::string const& str, Problem const& problem, DefaultDispatch const& dtable) -> std::unique_ptr<OptimizerBase>;

} // namespace Operon

#endif
