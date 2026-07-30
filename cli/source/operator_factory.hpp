// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_OPERATOR_FACTORY_HPP
#define OPERON_CLI_OPERATOR_FACTORY_HPP

#include <cstddef>                            // for size_t
#include <memory>                              // for unique_ptr, make_unique
#include <string>                              // for operator==, string
#include <unordered_map>                       // for unordered_map
#include <utility>                             // for addressof
#include <vector>                              // for vector
#include "operon/core/dispatch.hpp"            // for DispatchTable
#include "operon/core/types.hpp"               // for Span
#include "operon/core/individual.hpp"          // for Comparison
#include "operon/interpreter/interpreter.hpp"  // for Interpreter
#include "operon/optimizer/optimizer.hpp"
#include "util.hpp"                            // for Split
namespace Operon { struct EvaluatorBase; }
namespace Operon { class KeepBestReinserter; }
namespace Operon { struct MultiMutation; }
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

auto ParseReinserter(std::string const& str, ComparisonCallback&& comp, size_t eliteCount = 0) -> std::unique_ptr<ReinserterBase>;

auto ParseSelector(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<SelectorBase>;

// Result of parsing a --creator spec ("name:bias:mindepth:maxdepth:maxlength",
// all but name optional). mindepth/maxdepth/maxlength are Config::UniformTree
// Initializer settings, not CreatorBase constructor args (only bias and
// maxlength are) - bundled here since the CLI's --creator string is the one
// place a user names all of them together; the caller applies minDepth/
// maxDepth/maxLength to its own treeInitializer and (for maxLength) to the
// creator's own construction.
struct CreatorConfig {
    std::unique_ptr<CreatorBase> creator;
    size_t maxLength;
    size_t minDepth;
    size_t maxDepth;
};

// defaultMaxLength/defaultMinDepth/defaultMaxDepth are used for any of
// bias/mindepth/maxdepth/maxlength the spec string leaves unspecified.
auto ParseCreator(std::string const& str, PrimitiveSet const& pset, std::vector<Operon::Hash> const& inputs,
    size_t defaultMaxLength, size_t defaultMinDepth, size_t defaultMaxDepth) -> CreatorConfig;

// Parses a comma-separated "name:weight" list (weight optional, default 1.0)
// and adds each named operator to `mutator` with its weight. `available` maps
// mutator names to the already-constructed operator instances a caller has
// on hand (construction is context-specific - some operators need the
// creator, coefficient initializer, or primitive set - so this only handles
// the shared "which of my already-built operators, at what weight" parsing).
auto ParseMutators(std::string const& str, std::unordered_map<std::string, MutatorBase*> const& available, MultiMutation& mutator) -> void;

auto ParseEvaluator(std::string const& str, Problem& problem, ScalarDispatch& dtable, bool scale = true, bool skipNonFinite = false, double nonFinitePenaltyWeight = 1.0) -> std::unique_ptr<EvaluatorBase>;

auto ParseErrorMetric(std::string const& str) -> std::tuple<std::unique_ptr<Operon::ErrorMetric>, bool>;

auto ParseGenerator(std::string const& str, EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel, CoefficientOptimizer const* coeffOptimizer) -> std::unique_ptr<OffspringGeneratorBase>;

auto ParseOptimizer(std::string const& str, Problem const& problem, ScalarDispatch const& dtable) -> std::unique_ptr<OptimizerBase>;

} // namespace Operon

#endif
