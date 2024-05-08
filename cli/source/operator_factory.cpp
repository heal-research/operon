// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operator_factory.hpp"
#include <stdexcept>                       // for runtime_error
#include <fmt/format.h>                        // for format
#include <scn/scan.h>
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/operators/creator.hpp"    // for CreatorBase, BalancedTreeC...
#include "operon/operators/evaluator.hpp"  // for Evaluator, EvaluatorBase
#include "operon/operators/generator.hpp"  // for OffspringGeneratorBase
#include "operon/operators/reinserter.hpp"  // for OffspringGeneratorBase
#include "operon/operators/selector.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/optimizer/optimizer.hpp"

#include <cxxopts.hpp>


namespace Operon { class PrimitiveSet; }
namespace Operon { class Problem; }
namespace Operon { struct CrossoverBase; }
namespace Operon { struct MutatorBase; }
namespace Operon { struct Variable; }

namespace Operon {

namespace detail {
    auto GetErrorString(std::string const& name, std::string const& arg) {
        return fmt::format("unable to parse {} argument '{}'", name, arg);
    }
} // namespace detail

auto ParseReinserter(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<ReinserterBase>
{
    std::unique_ptr<ReinserterBase> reinserter;
    if (str == "keep-best") {
        reinserter = std::make_unique<KeepBestReinserter>(std::move(comp));
    } else if (str == "replace-worst") {
        reinserter = std::make_unique<ReplaceWorstReinserter>(std::move(comp));
    } else {
        throw std::invalid_argument(detail::GetErrorString("reinserter", str));
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
        if (tok.size() > 1) {
            auto result = scn::scan<std::size_t>(tok[1], "{}");
            ENSURE(result);
            tournamentSize = result->value();
        }
        dynamic_cast<Operon::TournamentSelector*>(selector.get())->SetTournamentSize(tournamentSize);
    } else if (name == "proportional") {
        selector = std::make_unique<Operon::ProportionalSelector>(std::move(comp));
        dynamic_cast<Operon::ProportionalSelector*>(selector.get())->SetObjIndex(0);
    } else if (name == "rank") {
        selector = std::make_unique<Operon::RankTournamentSelector>(std::move(comp));
        size_t tournamentSize{defaultTournamentSize};
        if (tok.size() > 1) {
            auto result = scn::scan<std::size_t>(tok[1], "{}");
            ENSURE(result);
            tournamentSize = result->value();
        }
        dynamic_cast<Operon::RankTournamentSelector*>(selector.get())->SetTournamentSize(tournamentSize);
    } else if (name == "random") {
        selector = std::make_unique<Operon::RandomSelector>();
    } else {
        throw std::invalid_argument(detail::GetErrorString("selector", str));
    }

    return selector;
}

auto ParseCreator(std::string const& str, PrimitiveSet const& pset, std::vector<Operon::Hash> const& inputs) -> std::unique_ptr<CreatorBase>
{
    std::unique_ptr<CreatorBase> creator;

    auto tok = Split(str, ':');
    auto name = tok[0];

    double bias{0}; // irregularity bias (used by btc and ptc20)
    if(tok.size() > 1) {
        auto res = scn::scan<double>(tok[1], "{}");
        ENSURE(res);
        bias = res->value();
    }

    if (str == "btc") {
        creator = std::make_unique<BalancedTreeCreator>(pset, inputs, bias);
    } else if (str == "ptc2") {
        creator = std::make_unique<ProbabilisticTreeCreator>(pset, inputs, bias);
    } else if (str == "grow") {
        creator = std::make_unique<GrowTreeCreator>(pset, inputs);
    } else {
        throw std::invalid_argument(detail::GetErrorString("creator", str));
    }
    return creator;
}

auto ParseEvaluator(std::string const& str, Problem& problem, DefaultDispatch& dtable, bool scale) -> std::unique_ptr<EvaluatorBase>
{
    using T = DefaultDispatch;

    std::unique_ptr<EvaluatorBase> evaluator;
    if (str == "r2") {
        evaluator = std::make_unique<Operon::Evaluator<T>>(problem, dtable, Operon::R2{}, scale);
    } else if (str == "c2") {
        evaluator = std::make_unique<Operon::Evaluator<T>>(problem, dtable, Operon::C2{}, scale);
    } else if (str == "nmse") {
        evaluator = std::make_unique<Operon::Evaluator<T>>(problem, dtable, Operon::NMSE{}, scale);
    } else if (str == "mse") {
        evaluator = std::make_unique<Operon::Evaluator<T>>(problem, dtable, Operon::MSE{}, scale);
    } else if (str == "rmse") {
        evaluator = std::make_unique<Operon::Evaluator<T>>(problem, dtable, Operon::RMSE{}, scale);
    } else if (str == "mae") {
        evaluator = std::make_unique<Operon::Evaluator<T>>(problem, dtable, Operon::MAE{}, scale);
    } else if (str == "mdl_gauss") {
        evaluator = std::make_unique<Operon::MinimumDescriptionLengthEvaluator<T, GaussianLikelihood<Operon::Scalar>>>(problem, dtable);
    } else if (str == "mdl_poisson") {
        evaluator = std::make_unique<Operon::MinimumDescriptionLengthEvaluator<T, PoissonLikelihood<Operon::Scalar>>>(problem, dtable);
    } else if (str == "gauss") {
        evaluator = std::make_unique<Operon::GaussianLikelihoodEvaluator<T>>(problem, dtable);
    } else {
        throw std::runtime_error(fmt::format("unable to parse evaluator metric '{}'\n", str));
    }
    return evaluator;
}

auto ParseGenerator(std::string const& str, EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr) -> std::unique_ptr<OffspringGeneratorBase>
{
    std::unique_ptr<OffspringGeneratorBase> generator;
    auto tok = Split(str, ':');
    auto name = tok[0];
    if (name == "basic") {
        generator = std::make_unique<BasicOffspringGenerator>(eval, cx, mut, femSel, maleSel, coeffOptimizer);
    } else if (name == "os") {
        size_t maxSelectionPressure{100};
        double comparisonFactor{0};
        if (tok.size() > 1) {
            maxSelectionPressure = scn::scan<size_t>(tok[1], "{}")->value();
        }
        if (tok.size() > 2) {
            comparisonFactor = scn::scan<double>(tok[2], "{}")->value();
        }
        generator = std::make_unique<OffspringSelectionGenerator>(eval, cx, mut, femSel, maleSel, coeffOptimizer);
        dynamic_cast<OffspringSelectionGenerator*>(generator.get())->MaxSelectionPressure(maxSelectionPressure);
        dynamic_cast<OffspringSelectionGenerator*>(generator.get())->ComparisonFactor(comparisonFactor);
    } else if (name == "brood") {
        generator = std::make_unique<BroodOffspringGenerator>(eval, cx, mut, femSel, maleSel, coeffOptimizer);
        size_t broodSize{BroodOffspringGenerator::DefaultBroodSize};
        if (tok.size() > 1) { broodSize = scn::scan<size_t>(tok[1], "{}")->value(); }
        dynamic_cast<BroodOffspringGenerator*>(generator.get())->BroodSize(broodSize);
    } else if (name == "poly") {
        generator = std::make_unique<PolygenicOffspringGenerator>(eval, cx, mut, femSel, maleSel, coeffOptimizer);
        size_t polygenicSize{PolygenicOffspringGenerator::DefaultBroodSize};
        if (tok.size() > 1) { polygenicSize = scn::scan<size_t>(tok[1], "{}")->value(); }
        dynamic_cast<PolygenicOffspringGenerator*>(generator.get())->PolygenicSize(polygenicSize);
    } else {
        throw std::invalid_argument(detail::GetErrorString("generator", str));
    }
    return generator;
}

auto ParseOptimizer(std::string const& /*str*/, Problem const& /*problem*/, DefaultDispatch const& /*dtable*/) -> std::unique_ptr<OptimizerBase> {
    throw std::runtime_error("not implemented");
}

} // namespace Operon
