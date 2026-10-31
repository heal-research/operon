// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operator_factory.hpp"
#include "cli_args.hpp"
#include "operon/operators/creator.hpp" // for CreatorBase, BalancedTreeC...
#include "operon/operators/evaluator.hpp" // for Evaluator, EvaluatorBase
#include "operon/operators/generator.hpp" // for OffspringGeneratorBase
#include "operon/operators/local_search.hpp"
#include "operon/operators/mutation.hpp" // for MultiMutation, MutatorBase
#include "operon/operators/reinserter.hpp" // for OffspringGeneratorBase
#include "operon/operators/selector.hpp"
#include "operon/optimizer/optimizer.hpp"
#include <cmath> // for isfinite
#include <fmt/format.h> // for format
#include <scn/scan.h>
#include <stdexcept> // for runtime_error

namespace Operon {
class PrimitiveSet;
}
namespace Operon {
class Problem;
}
namespace Operon {
struct CrossoverBase;
}
namespace Operon {
struct MutatorBase;
}
namespace Operon {
struct Variable;
}

namespace Operon {

namespace {
    auto GetErrorString(std::string const& name, std::string const& arg)
    {
        return fmt::format("unable to parse {} argument '{}'", name, arg);
    }

    // Scans one numeric field of a `name:field:...` spec; the error names the option, the whole spec and the field.
    template <typename T>
    auto ScanField(std::string const& option, std::string const& spec, std::string const& token, std::string_view field)
        -> T
    {
        auto const value = Cli::ScanExact<T>(token);
        if (!value) {
            throw std::invalid_argument(
                fmt::format("unable to parse {} argument '{}': '{}' is not a valid {}", option, spec, token, field));
        }
        return *value;
    }

    // Splits on `delimiter` keeping empty fields (also a trailing one), unlike Split(): "os:" has two fields.
    auto SplitFields(std::string const& s, char delimiter) -> std::vector<std::string>
    {
        std::vector<std::string> fields { std::string {} };
        for (char const c : s) {
            if (c == delimiter) {
                fields.emplace_back();
            } else {
                fields.back().push_back(c);
            }
        }
        return fields;
    }

    // A size field that must be at least one: zero would build an empty brood and dereference its end iterator.
    auto ScanPositiveSize(
        std::string const& option, std::string const& spec, std::string const& token, std::string_view field) -> size_t
    {
        auto const value = ScanField<size_t>(option, spec, token, field);
        if (value == 0) {
            throw std::invalid_argument(
                fmt::format("unable to parse {} argument '{}': '{}' is not a valid {} (must be at least 1)", option,
                    spec, token, field));
        }
        return value;
    }
} // namespace

auto ParseReinserter(std::string const& str, ComparisonCallback&& comp, size_t eliteCount)
    -> std::unique_ptr<ReinserterBase>
{
    std::unique_ptr<ReinserterBase> reinserter;
    if (str == "keep-best") {
        reinserter = std::make_unique<KeepBestReinserter>(std::move(comp), eliteCount);
    } else if (str == "replace-worst") {
        reinserter = std::make_unique<ReplaceWorstReinserter>(std::move(comp), eliteCount);
    } else {
        throw std::invalid_argument(GetErrorString("reinserter", str));
    }
    return reinserter;
}

auto ParseCacheVisibility(std::string const& str) -> CacheVisibility
{
    if (str == "immediate") {
        return CacheVisibility::Immediate;
    }
    if (str == "committed") {
        return CacheVisibility::Committed;
    }
    throw std::invalid_argument(fmt::format(
        "{}; --cache-visibility must be one of: immediate, committed", GetErrorString("cache-visibility", str)));
}

auto ParseSelector(std::string const& str, ComparisonCallback&& comp) -> std::unique_ptr<Operon::SelectorBase>
{
    auto tok = Split(str, ':');
    auto name = tok[0];
    std::unique_ptr<Operon::SelectorBase> selector;
    constexpr size_t defaultTournamentSize { 5 };
    if (name == "tournament") {
        selector = std::make_unique<Operon::TournamentSelector>(std::move(comp));
        size_t tournamentSize { defaultTournamentSize };
        if (tok.size() > 1) {
            auto result = scn::scan<std::size_t>(tok[1], "{}");
            ENSURE(result);
            tournamentSize = result->value();
        }
        dynamic_cast<Operon::TournamentSelector*>(selector.get())->SetTournamentSize(tournamentSize);
    } else if (name == "proportional") {
        selector = std::make_unique<Operon::ProportionalSelector>(std::move(comp));
        dynamic_cast<Operon::ProportionalSelector*>(selector.get())->SetObjIndex(0);
    } else if (name == "random") {
        selector = std::make_unique<Operon::RandomSelector>();
    } else {
        throw std::invalid_argument(GetErrorString("selector", str));
    }

    return selector;
}

auto ParseCreator(std::string const& str, PrimitiveSet const& pset, std::vector<Operon::Hash> const& inputs,
    size_t defaultMaxLength, size_t defaultMinDepth, size_t defaultMaxDepth) -> CreatorConfig
{
    auto tok = Split(str, ':');
    if (tok.empty()) {
        throw std::invalid_argument(GetErrorString("creator", str));
    }
    auto const& name = tok[0]; // NOLINT(readability-identifier-length) - previously compared `str` itself here, which
                               // could never match once any param token was appended

    double bias { 0 }; // irregularity bias (used by btc and ptc2)
    size_t maxLength = defaultMaxLength;
    size_t minDepth = defaultMinDepth;
    size_t maxDepth = defaultMaxDepth;

    if (tok.size() > 1 && !tok[1].empty()) {
        auto res = scn::scan<double>(tok[1], "{}");
        ENSURE(res);
        bias = res->value();
    }
    if (tok.size() > 2 && !tok[2].empty()) {
        auto res = scn::scan<size_t>(tok[2], "{}");
        ENSURE(res);
        minDepth = res->value();
    }
    if (tok.size() > 3 && !tok[3].empty()) {
        auto res = scn::scan<size_t>(tok[3], "{}");
        ENSURE(res);
        maxDepth = res->value();
    }
    if (tok.size() > 4 && !tok[4].empty()) {
        auto res = scn::scan<size_t>(tok[4], "{}");
        ENSURE(res);
        maxLength = res->value();
    }

    std::unique_ptr<CreatorBase> creator;
    if (name == "btc") {
        creator = std::make_unique<BalancedTreeCreator>(&pset, inputs, bias, maxLength);
    } else if (name == "ptc2") {
        creator = std::make_unique<ProbabilisticTreeCreator>(&pset, inputs, bias, maxLength);
    } else if (name == "grow") {
        creator = std::make_unique<GrowTreeCreator>(&pset, inputs, maxLength);
    } else {
        throw std::invalid_argument(GetErrorString("creator", str));
    }
    return CreatorConfig { std::move(creator), maxLength, minDepth, maxDepth };
}

auto ParseMutators(std::string const& str, std::unordered_map<std::string, MutatorBase*> const& available,
    MultiMutation& mutator) -> void
{
    for (auto const& spec : Split(str, ',')) {
        auto tok = Split(spec, ':');
        if (tok.empty() || tok[0].empty()) {
            throw std::invalid_argument(GetErrorString("mutators", str));
        }
        auto const& name = tok[0];

        double weight { 1.0 };
        if (tok.size() > 1 && !tok[1].empty()) {
            auto res = scn::scan<double>(tok[1], "{}");
            ENSURE(res);
            weight = res->value();
        }
        if (!(weight > 0)) {
            throw std::invalid_argument(GetErrorString("mutators", str));
        }

        auto it = available.find(name);
        if (it == available.end()) {
            throw std::invalid_argument(GetErrorString("mutators", name));
        }
        mutator.Add(it->second, weight);
    }
    if (mutator.Count() == 0) {
        throw std::invalid_argument(GetErrorString("mutators", str));
    }
}

auto ParseEvaluator(std::string const& str, Problem& problem, ScalarDispatch& dtable, bool skipNonFinite,
    double nonFinitePenaltyWeight, std::optional<ShapeConstraintSpec> const& shape) -> std::unique_ptr<EvaluatorBase>
{
    using T = ScalarDispatch;

    if (skipNonFinite && (str == "r2" || str == "c2")) {
        throw std::runtime_error(fmt::format(
            "--skip-nonfinite is not supported with objective '{}': R2/C2 have no finite-subset semantics\n", str));
    }

    if (skipNonFinite && !(std::isfinite(nonFinitePenaltyWeight) && nonFinitePenaltyWeight >= 0.0)) {
        throw std::runtime_error(fmt::format(
            "--nonfinite-penalty-weight must be a finite, non-negative value (got {})\n", nonFinitePenaltyWeight));
    }

    // Each branch is a concrete value evaluator; MakeValueEvaluator constructs it, or its shape-constrained adapter,
    // before the result is erased to EvaluatorBase.
    using Gauss = GaussianLikelihood<Operon::Scalar>;
    using Poisson = PoissonLikelihood<Operon::Scalar>;
    if (str == "r2") {
        return MakeValueEvaluator<Operon::Evaluator<T>>(shape, &dtable, &problem, &dtable, Operon::R2 {});
    }
    if (str == "c2") {
        return MakeValueEvaluator<Operon::Evaluator<T>>(shape, &dtable, &problem, &dtable, Operon::C2 {});
    }
    if (str == "nmse") {
        return MakeValueEvaluator<Operon::Evaluator<T>>(
            shape, &dtable, &problem, &dtable, Operon::NMSE {}, skipNonFinite, nonFinitePenaltyWeight);
    }
    if (str == "mse") {
        return MakeValueEvaluator<Operon::Evaluator<T>>(
            shape, &dtable, &problem, &dtable, Operon::MSE {}, skipNonFinite, nonFinitePenaltyWeight);
    }
    if (str == "rmse") {
        return MakeValueEvaluator<Operon::Evaluator<T>>(
            shape, &dtable, &problem, &dtable, Operon::RMSE {}, skipNonFinite, nonFinitePenaltyWeight);
    }
    if (str == "mae") {
        return MakeValueEvaluator<Operon::Evaluator<T>>(
            shape, &dtable, &problem, &dtable, Operon::MAE {}, skipNonFinite, nonFinitePenaltyWeight);
    }
    if (str == "mdl_gauss") {
        return MakeValueEvaluator<Operon::MinimumDescriptionLengthEvaluator<T, Gauss>>(
            shape, &dtable, &problem, &dtable);
    }
    if (str == "mdl_poisson") {
        return MakeValueEvaluator<Operon::MinimumDescriptionLengthEvaluator<T, Poisson>>(
            shape, &dtable, &problem, &dtable);
    }
    if (str == "fbf_gauss") {
        return MakeValueEvaluator<Operon::FractionalBayesFactorEvaluator<T, Gauss>>(shape, &dtable, &problem, &dtable);
    }
    if (str == "fbf_poisson") {
        throw std::runtime_error(
            "fbf_poisson is not supported: the fractional Bayes factor is derived under Gaussian assumptions");
    }
    if (str == "gauss") {
        return MakeValueEvaluator<Operon::GaussianLikelihoodEvaluator<T>>(shape, &dtable, &problem, &dtable);
    }
    throw std::runtime_error(fmt::format("unable to parse evaluator metric '{}'\n", str));
}

auto ParseErrorMetric(std::string const& str) -> std::tuple<std::unique_ptr<Operon::ErrorMetric>, bool>
{
    std::unique_ptr<Operon::ErrorMetric> metric;
    bool scale { true };
    if (str == "r2") {
        metric = std::make_unique<Operon::R2>();
    } else if (str == "c2") {
        scale = false;
        metric = std::make_unique<Operon::C2>();
    } else if (str == "nmse") {
        metric = std::make_unique<Operon::NMSE>();
    } else if (str == "mse") {
        metric = std::make_unique<Operon::MSE>();
    } else if (str == "rmse") {
        metric = std::make_unique<Operon::RMSE>();
    } else if (str == "mae") {
        metric = std::make_unique<Operon::MAE>();
    } else {
        throw std::runtime_error(fmt::format("unsupported error metric '{}'\n", str));
    }
    return { std::move(metric), scale };
}

auto ParseGenerator(std::string const& str, EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut,
    SelectorBase& femSel, SelectorBase& maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
    -> std::unique_ptr<OffspringGeneratorBase>
{
    std::unique_ptr<OffspringGeneratorBase> generator;
    auto tok = SplitFields(str, ':');
    auto const& name = tok[0];
    auto const maxFields = name == "basic" ? 1UL : (name == "os" ? 3UL : 2UL);
    if ((name == "basic" || name == "os" || name == "brood" || name == "poly") && tok.size() > maxFields) {
        throw std::invalid_argument(
            fmt::format("unable to parse generator argument '{}': '{}' takes at most {} field(s) after the name", str,
                name, maxFields - 1));
    }
    if (name == "basic") {
        generator = std::make_unique<BasicOffspringGenerator>(&eval, &cx, &mut, &femSel, &maleSel, coeffOptimizer);
    } else if (name == "os") {
        size_t maxSelectionPressure { 100 };
        double comparisonFactor { 0 };
        if (tok.size() > 1) {
            maxSelectionPressure = ScanField<size_t>("generator", str, tok[1], "maximum selection pressure");
        }
        if (tok.size() > 2) {
            comparisonFactor = ScanField<double>("generator", str, tok[2], "comparison factor");
        }
        generator = std::make_unique<OffspringSelectionGenerator>(&eval, &cx, &mut, &femSel, &maleSel, coeffOptimizer);
        dynamic_cast<OffspringSelectionGenerator*>(generator.get())->MaxSelectionPressure(maxSelectionPressure);
        dynamic_cast<OffspringSelectionGenerator*>(generator.get())->ComparisonFactor(comparisonFactor);
    } else if (name == "brood") {
        generator = std::make_unique<BroodOffspringGenerator>(&eval, &cx, &mut, &femSel, &maleSel, coeffOptimizer);
        size_t broodSize { BroodOffspringGenerator::DefaultBroodSize };
        if (tok.size() > 1) {
            broodSize = ScanPositiveSize("generator", str, tok[1], "brood size");
        }
        dynamic_cast<BroodOffspringGenerator*>(generator.get())->BroodSize(broodSize);
    } else if (name == "poly") {
        generator = std::make_unique<PolygenicOffspringGenerator>(&eval, &cx, &mut, &femSel, &maleSel, coeffOptimizer);
        size_t polygenicSize { PolygenicOffspringGenerator::DefaultBroodSize };
        if (tok.size() > 1) {
            polygenicSize = ScanPositiveSize("generator", str, tok[1], "polygenic size");
        }
        dynamic_cast<PolygenicOffspringGenerator*>(generator.get())->PolygenicSize(polygenicSize);
    } else {
        throw std::invalid_argument(GetErrorString("generator", str));
    }
    return generator;
}

} // namespace Operon
