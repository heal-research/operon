// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operator_factory.hpp"
#include <stdexcept>                       // for runtime_error
#include <fmt/format.h>                        // for format
#include <tuple>
#include <scn/scan/scan.h>
#include "operon/operators/creator.hpp"    // for CreatorBase, BalancedTreeC...
#include "operon/operators/evaluator.hpp"  // for Evaluator, EvaluatorBase
#include "operon/operators/generator.hpp"  // for OffspringGeneratorBase
#include "operon/operators/reinserter.hpp"  // for OffspringGeneratorBase
#include "operon/operators/selector.hpp"

namespace Operon { class PrimitiveSet; }
namespace Operon { class Problem; }
namespace Operon { struct CrossoverBase; }
namespace Operon { struct MutatorBase; }
namespace Operon { struct Variable; }

namespace Operon {
auto ParseReinserter(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<ReinserterBase>
{
    std::unique_ptr<ReinserterBase> reinserter;
    if (str == "keep-best") {
        reinserter = std::make_unique<KeepBestReinserter>(std::move(comp));
    } else if (str == "replace-worst") {
        reinserter = std::make_unique<ReplaceWorstReinserter>(std::move(comp));
    }
    return reinserter;
}

auto ParseSelector(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<Operon::SelectorBase>
{
    auto tok = Split(str, ':');
    auto name = tok[0];
    std::unique_ptr<Operon::SelectorBase> selector;
    constexpr size_t defaultTournamentSize{5};
    if (name == "tournament") {
        selector = std::make_unique<Operon::TournamentSelector>(std::move(comp));
        size_t tournamentSize{defaultTournamentSize};
        if (tok.size() > 1) { (void) scn::scan(tok[1], "{}", tournamentSize); }
        dynamic_cast<Operon::TournamentSelector*>(selector.get())->SetTournamentSize(tournamentSize);
    } else if (name == "proportional") {
        selector = std::make_unique<Operon::ProportionalSelector>(std::move(comp));
        dynamic_cast<Operon::ProportionalSelector*>(selector.get())->SetObjIndex(0);
    } else if (name == "rank") {
        selector = std::make_unique<Operon::RankTournamentSelector>(std::move(comp));
        size_t tournamentSize{defaultTournamentSize};
        if (tok.size() > 1) { (void) scn::scan(tok[1], "{}", tournamentSize); }
        dynamic_cast<Operon::RankTournamentSelector*>(selector.get())->SetTournamentSize(tournamentSize);
    } else if (name == "random") {
        selector = std::make_unique<Operon::RandomSelector>();
    }
        
    return selector;
}

auto ParseCreator(std::string const& str, PrimitiveSet const& pset, Operon::Span<Variable const> inputs) -> std::unique_ptr<CreatorBase>
{
    std::unique_ptr<CreatorBase> creator;

    auto tok = Split(str, ':');
    auto name = tok[0];

    double bias{0}; // irregularity bias (used by btc and ptc20)
    if(tok.size() > 1) { (void) scn::scan(tok[1], "{}", bias); }

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
        evaluator = std::make_unique<Operon::Evaluator>(problem, interpreter, Operon::R2{}, true);
    } else if (str == "c2") {
        evaluator = std::make_unique<Operon::Evaluator>(problem, interpreter, Operon::C2{}, false);
    } else if (str == "nmse") {
        evaluator = std::make_unique<Operon::Evaluator>(problem, interpreter, Operon::NMSE{}, true);
    } else if (str == "mse") {
        evaluator = std::make_unique<Operon::Evaluator>(problem, interpreter, Operon::MSE{}, true);
    } else if (str == "rmse") {
        evaluator = std::make_unique<Operon::Evaluator>(problem, interpreter, Operon::RMSE{}, true);
    } else if (str == "mae") {
        evaluator = std::make_unique<Operon::Evaluator>(problem, interpreter, Operon::MAE{}, true);
    } else {
        throw std::runtime_error(fmt::format("Unknown metric {}\n", str));
    }
    return evaluator;
}

auto ParseErrorMetric(std::string const& str) -> std::tuple<std::unique_ptr<ErrorMetric>, bool>
{
    std::unique_ptr<ErrorMetric> error;
    bool scale{true};
    if (str == "r2") {
        error = std::make_unique<Operon::R2>();
    } else if (str == "c2") {
        scale = false;
        error = std::make_unique<Operon::C2>();
    } else if (str == "nmse") {
        error = std::make_unique<Operon::NMSE>();
    } else if (str == "mse") {
        error = std::make_unique<Operon::MSE>();
    } else if (str == "rmse") {
        error = std::make_unique<Operon::RMSE>();
    } else if (str == "mae") {
        error = std::make_unique<Operon::MAE>();
    } else {
        throw std::runtime_error(fmt::format("Unknown metric {}\n", str));
    }
    return std::make_tuple(std::move(error), scale);
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
        if (tok.size() > 1) { (void) scn::scan(tok[1], "{}", maxSelectionPressure); }
        if (tok.size() > 2) { (void) scn::scan(tok[2], "{}", comparisonFactor); }
        generator = std::make_unique<OffspringSelectionGenerator>(eval, cx, mut, femSel, maleSel);
        dynamic_cast<OffspringSelectionGenerator*>(generator.get())->MaxSelectionPressure(maxSelectionPressure);
        dynamic_cast<OffspringSelectionGenerator*>(generator.get())->ComparisonFactor(comparisonFactor);
    } else if (name == "brood") {
        generator = std::make_unique<BroodOffspringGenerator>(eval, cx, mut, femSel, maleSel);
        size_t broodSize{5};
        if (tok.size() > 1) { (void) scn::scan(tok[1], "{}", broodSize); }
        dynamic_cast<BroodOffspringGenerator*>(generator.get())->BroodSize(broodSize);
    } else if (name == "poly") {
        generator = std::make_unique<BroodOffspringGenerator>(eval, cx, mut, femSel, maleSel);
        size_t polygenicSize{5};
        if (tok.size() > 1) { (void) scn::scan(tok[1], "{}", polygenicSize); }
        dynamic_cast<PolygenicOffspringGenerator*>(generator.get())->PolygenicSize(polygenicSize);
    }
    return generator;
}

} // namespace Operon
