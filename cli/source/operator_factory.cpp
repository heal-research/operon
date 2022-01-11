// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operator_factory.hpp"
#include <stdexcept>                       // for runtime_error
#include <fmt/format.h>                        // for format
#include "operon/operators/creator.hpp"    // for CreatorBase, BalancedTreeC...
#include "operon/operators/evaluator.hpp"  // for Evaluator, EvaluatorBase
#include "operon/operators/generator.hpp"  // for OffspringGeneratorBase
namespace Operon { class PrimitiveSet; }
namespace Operon { class Problem; }
namespace Operon { struct CrossoverBase; }
namespace Operon { struct MutatorBase; }
namespace Operon { struct Variable; }

namespace Operon {
auto ParseCreator(std::string const& str, PrimitiveSet const& pset, Operon::Span<Variable const> inputs) -> std::unique_ptr<CreatorBase>
{
    std::unique_ptr<CreatorBase> creator;

    auto tok = Split(str, ':');
    auto name = tok[0];

    double bias{0}; // irregularity bias (used by btc and ptc20)
    if(tok.size() > 1) { scn::scan(tok[1], "{}", bias); }

    if (str == "btc") {
        creator = std::make_unique<BalancedTreeCreator>(pset, inputs, bias);
    } else if (str == "ptc2") {
        creator = std::make_unique<ProbabilisticTreeCreator>(pset, inputs, bias);
    } else if (str == "grow") {
        creator = std::make_unique<GrowTreeCreator>(pset, inputs);
    }
    return creator;
}

auto ParseEvaluator(std::string const& str, Problem& problem, Interpreter& interpreter) -> std::unique_ptr<EvaluatorBase>
{
    std::unique_ptr<EvaluatorBase> evaluator;
    if (str == "r2") {
        evaluator = std::make_unique<Operon::R2Evaluator>(problem, interpreter);
    } else if (str == "c2") {
        evaluator = std::make_unique<Operon::SquaredCorrelationEvaluator>(problem, interpreter);
    } else if (str == "nmse") {
        evaluator = std::make_unique<Operon::NormalizedMeanSquaredErrorEvaluator>(problem, interpreter);
    } else if (str == "mse") {
        evaluator = std::make_unique<Operon::MeanSquaredErrorEvaluator>(problem, interpreter);
    } else if (str == "rmse") {
        evaluator = std::make_unique<Operon::RootMeanSquaredErrorEvaluator>(problem, interpreter);
    } else if (str == "mae") {
        evaluator = std::make_unique<Operon::MeanAbsoluteErrorEvaluator>(problem, interpreter);
    } else if (str == "l2") {
        evaluator = std::make_unique<Operon::L2NormEvaluator>(problem, interpreter);
    } else {
        throw std::runtime_error(fmt::format("Unknown metric {}\n", str));
    }
    return evaluator;
}

auto ParseGenerator(std::string const& str, EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel) -> std::unique_ptr<OffspringGeneratorBase>
{
    std::unique_ptr<OffspringGeneratorBase> generator;
    auto tok = Split(str, ':');
    auto name = tok[0];
    if (name == "basic") {
        generator = std::make_unique<BasicOffspringGenerator>(eval, cx, mut, femSel, maleSel);
    } else if (name == "os") {
        size_t maxSelectionPressure{100};
        double comparisonFactor{0};
        if (tok.size() > 1) { scn::scan(tok[1], "{}", maxSelectionPressure); }
        if (tok.size() > 2) { scn::scan(tok[2], "{}", comparisonFactor); }
        generator = std::make_unique<OffspringSelectionGenerator>(eval, cx, mut, femSel, maleSel);
        dynamic_cast<OffspringSelectionGenerator*>(generator.get())->MaxSelectionPressure(maxSelectionPressure);
        dynamic_cast<OffspringSelectionGenerator*>(generator.get())->ComparisonFactor(comparisonFactor);
    } else if (name == "brood") {
        generator = std::make_unique<BroodOffspringGenerator>(eval, cx, mut, femSel, maleSel);
        size_t broodSize{5};
        if (tok.size() > 1) { scn::scan(tok[1], "{}", broodSize); }
        dynamic_cast<BroodOffspringGenerator*>(generator.get())->BroodSize(broodSize);
    } else if (name == "poly") {
        generator = std::make_unique<BroodOffspringGenerator>(eval, cx, mut, femSel, maleSel);
        size_t polygenicSize{5};
        if (tok.size() > 1) { scn::scan(tok[1], "{}", polygenicSize); }
        dynamic_cast<PolygenicOffspringGenerator*>(generator.get())->PolygenicSize(polygenicSize);
    }
    return generator;
}

} // namespace Operon
