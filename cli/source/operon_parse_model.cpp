// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <fstream>
#include <optional>
#include <string>

#include "operon/core/serialization.hpp"
#include "operon/random/random.hpp"
#include "operon/core/types.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"
#include "reporter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/range_tightening.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/linear_scaling.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "shape_constraints_config.hpp"

#include <cxxopts.hpp>
#include <scn/scan.h>
#include <tl/expected.hpp>

namespace {
    enum class ParseError : std::uint8_t  {
        Success        = 0,
        MissingDataset = 1,
        MissingInfix   = 2,
        NoOptions      = 3,
        UnknownError   = 4
    };

    auto ParseOptions(int argc, char** argv) noexcept -> tl::expected<cxxopts::ParseResult, ParseError> { // NOLINT(bugprone-exception-escape)
        cxxopts::Options opts("operon_parse_model", "Parse and evaluate a model in infix form");

        opts.add_options()
            ("dataset", "Dataset file name (csv) (required)", cxxopts::value<std::string>())
            ("target", "Name of the target variable (if none provided, model output will be printed)", cxxopts::value<std::string>())
            ("range", "Data range [A:B)", cxxopts::value<std::string>())
            ("scale", "Linear scaling slope:intercept", cxxopts::value<std::string>())
            ("optimizer", "Optimizer for model coefficients (lm, lbfgs, sgd)", cxxopts::value<std::string>()->default_value("lm"))
            ("likelihood", "Optimizer loss function (gaussian, poisson)", cxxopts::value<std::string>()->default_value("gaussian"))
            ("iterations", "Optimizer iterations (0 disables refitting; reported stats are the model's own coefficients as given)", cxxopts::value<int>()->default_value("0"))
            ("shape-constraints-config", "Path to a JSON shape-constraints config; when set with --target, also prints affine-certified feasibility for the parsed model", cxxopts::value<std::string>())
            ("shape-bound-mode", "Arithmetic backend for the printed naive bound: combined (default), interval-only, affine-only", cxxopts::value<std::string>()->default_value("combined"))
            ("tighten-range", "With --shape-constraints-config, also print TightenRange's mean-value-form bound alongside the naive one, per constraint", cxxopts::value<bool>()->default_value("false"))
            ("sample-check", "With --shape-constraints-config, also Monte-Carlo sample N points from the domain box per constraint and print the observed [min:max], as an independent soundness cross-check on the printed bound", cxxopts::value<std::size_t>())
            ("dump-tree-json", "Write the parsed model tree (exact structure, via Operon::Serialization::ToJson) to this path before any other processing", cxxopts::value<std::string>())
            ("debug", "Show some debugging information", cxxopts::value<bool>()->default_value("false"))
            ("format", "Format string (see https://fmt.dev/latest/syntax.html)", cxxopts::value<std::string>()->default_value(":>#8.4g"))
            ("help", "Print help");

        opts.allow_unrecognised_options();

        cxxopts::ParseResult result;
        try {
            result = opts.parse(argc, argv);
        } catch (cxxopts::exceptions::parsing const& ex) {
            fmt::print(stderr, "error: {}. rerun with --help to see available options.\n", ex.what());
            return tl::make_unexpected(ParseError::UnknownError);
        };

        if (result.arguments().empty() || result.contains("help")) {
            fmt::print("{}\n", opts.help());
            return tl::make_unexpected(ParseError::NoOptions);
        }

        if (!result.contains("dataset")) {
            fmt::print(stderr, "error: no dataset was specified.\n");
            return tl::make_unexpected(ParseError::MissingDataset);
        }

        if (result.unmatched().empty()) {
            fmt::print(stderr, "error: no infix string was provided.\n");
            return tl::make_unexpected(ParseError::MissingInfix);
        }
        return result;
    }

    auto ParseOptimizer(Operon::ScalarDispatch const* dtable, Operon::Problem const* problem, std::string const& optimizer, std::string const& likelihood) {
        std::unique_ptr<Operon::OptimizerBase> opt;

        if (optimizer == "lm") {
            // Eigen backend, matching operon_gp/operon_nsgp/operon_enum
            // (all hardcode OptimizerType::Eigen) - not the class template's
            // own default (Tiny), so "lm" means the same thing everywhere.
            opt = std::make_unique<Operon::LevenbergMarquardtOptimizer<Operon::ScalarDispatch, Operon::OptimizerType::Eigen>>(dtable, problem);
        } else if (optimizer == "lbfgs") {
            if (likelihood == "gaussian") {
                opt = std::make_unique<Operon::LBFGSOptimizer<Operon::ScalarDispatch, Operon::GaussianLoss<Operon::Scalar>>>(dtable, problem);
            } else if (likelihood == "poisson") {
                opt = std::make_unique<Operon::LBFGSOptimizer<Operon::ScalarDispatch, Operon::PoissonLoss<Operon::Scalar>>>(dtable, problem);
            }
        } else if (optimizer == "sgd") {
            if (likelihood == "gaussian") {
                opt = std::make_unique<Operon::SGDOptimizer<Operon::ScalarDispatch, Operon::GaussianLoss<Operon::Scalar>>>(dtable, problem);
            } else if (likelihood == "poisson") {
                opt = std::make_unique<Operon::SGDOptimizer<Operon::ScalarDispatch, Operon::PoissonLoss<Operon::Scalar>>>(dtable, problem);
            }
        }
        return opt;
    }

    auto FitScale(cxxopts::ParseResult const& result,
                  Operon::Tree const& model,
                  Operon::Problem const& problem,
                  Operon::ScalarDispatch const& dtable,
                  Operon::Range range) -> std::pair<Operon::Scalar, Operon::Scalar>
    {
        if (result["scale"].count() > 0) {
            auto res = scn::scan<Operon::Scalar, Operon::Scalar>(result["scale"].as<std::string>(), "{}:{}");
            ENSURE(res);
            auto [a, b] = res->values();
            return { static_cast<Operon::Scalar>(a), static_cast<Operon::Scalar>(b) };
        }
        auto const scaling = Operon::FitLinearScaling(model, problem, dtable, range);
        return scaling ? std::pair{static_cast<Operon::Scalar>(scaling->Scale), static_cast<Operon::Scalar>(scaling->Offset)}
                       : std::pair{Operon::Scalar{1}, Operon::Scalar{0}};
    }

    // Duplicated from shape_constrained_evaluator.cpp's anonymous-namespace
    // helper of the same name (same Ref-node-DAG-slicing idiom as the
    // tree_diff tests) -- not exported from that translation unit, and this
    // is a ~6-line utility, so a local copy is cheaper than exporting a
    // private implementation detail across a module boundary for one caller.
    constexpr std::size_t kNoGrad = std::numeric_limits<std::size_t>::max();

    auto SliceToTree(Operon::VariableGradientDag const& dag, std::size_t root) -> std::optional<Operon::Tree>
    {
        if (root == kNoGrad) { return std::nullopt; }
        Operon::Vector<Operon::Node> sliced(dag.Nodes.begin(), dag.Nodes.begin() + static_cast<std::ptrdiff_t>(root) + 1);
        Operon::Tree t(std::move(sliced));
        t.UpdateNodes();
        return t;
    }

    auto VariableIndex(Operon::VariableGradientDag const& dag, Operon::Hash variable) -> std::optional<std::size_t>
    {
        auto it = std::ranges::find(dag.Variables, variable);
        if (it == dag.Variables.end()) { return std::nullopt; }
        return static_cast<std::size_t>(std::distance(dag.Variables.begin(), it));
    }

    // The same per-constraint tree ShapeConstrainedEvaluator::BoundFor (in
    // shape_constrained_evaluator.cpp, also private) would bound -- Identity
    // is the model itself, First-/SecondDerivative slice a gradient dag built
    // via BuildVariableGradientDag. Returns nullopt when the derivative is
    // identically zero or uncertifiable, mirroring that function's [0,0] /
    // error handling, since this caller only wants a bound to print, not a
    // certification decision.
    auto ConstraintTreeFor(Operon::ShapeConstraint const& c, Operon::Tree const& model, Operon::Hash variable) -> std::optional<Operon::Tree>
    {
        if (c.Op == Operon::ShapeConstraintOp::Identity) { return model; }
        auto dag1 = Operon::BuildVariableGradientDag(model, model.GetCoefficients());
        auto const i1 = VariableIndex(dag1, variable);
        if (!i1 || !dag1.Certain[*i1]) { return std::nullopt; }
        auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
        if (c.Op == Operon::ShapeConstraintOp::FirstDerivative || !d1) { return d1; }
        auto dag2 = Operon::BuildVariableGradientDag(*d1, d1->GetCoefficients());
        auto const i2 = VariableIndex(dag2, variable);
        if (!i2 || !dag2.Certain[*i2]) { return std::nullopt; }
        return SliceToTree(dag2, dag2.Roots[*i2]);
    }

    auto BuildDomainMap(Operon::ShapeConstraintSet const& constraints, Operon::Dataset const& ds)
        -> Operon::IntervalEvaluator::DomainMap
    {
        Operon::IntervalEvaluator::DomainMap domains;
        for (auto const& [name, bound] : constraints.Domains) {
            auto v = ds.GetVariable(name);
            if (!v) { throw std::invalid_argument(fmt::format("domain references unknown variable '{}'", name)); }
            domains.insert_or_assign(v->Hash, bound);
        }
        return domains;
    }

    auto PrintTargetAnalysis(
        cxxopts::ParseResult const& result,
        Operon::Dataset& ds,
        Operon::Range range,
        Operon::ScalarDispatch const& dtable,
        std::string const& format,
        Operon::Tree& model
    ) -> void
    {
        auto tgt = ds.GetValues(result["target"].as<std::string>()).subspan(range.Start(), range.Size());

        Operon::Problem problem{&ds};
        problem.SetTrainingRange(range);
        problem.SetTestRange(range);
        problem.SetTarget(result["target"].as<std::string>());
        problem.SetDefaultInputs();
        Operon::RandomGenerator rng{0};

        // Optionally refit model's coefficients (--iterations > 0) before
        // evaluating it - the caller decides whether "parse and evaluate"
        // means "as literally given" (default, --iterations 0) or "best fit
        // achievable from these starting coefficients" (--iterations N).
        // Optimize() itself doesn't mutate model (it takes Tree const&); the
        // optimized coefficients only take effect once applied back via
        // SetCoefficients, which is what makes this refit actually visible in
        // the stats below, unlike before.
        auto opt = ParseOptimizer(&dtable, &problem, result["optimizer"].as<std::string>(), result["likelihood"].as<std::string>());
        opt->SetIterations(result["iterations"].as<int>());
        auto summary = Operon::FitOutcome{tl::unexpected(Operon::FitFailure{})};
        if (opt->Iterations() > 0) {
            summary = opt->Optimize(rng, model);
            if (summary.has_value()) { model.SetCoefficients(summary->FinalParameters); }
        }

        using Interpreter = Operon::Interpreter<Operon::Scalar, Operon::ScalarDispatch>;
        auto est = Interpreter::Evaluate(model, ds, range);

        auto [a, b] = FitScale(result, model, problem, dtable, range);
        std::ranges::transform(est, est.begin(), [&](auto v) -> auto { return (v * a) + b; });
        auto r2   = -Operon::R2{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto rs   = -Operon::C2{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto mae  =  Operon::MAE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto mse  =  Operon::MSE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto rmse =  Operon::RMSE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto nmse =  Operon::NMSE{}(Operon::Span<Operon::Scalar>{est}, tgt);

        Operon::Individual ind;
        ind.Genotype = model;
        Operon::MinimumDescriptionLengthEvaluator<Operon::ScalarDispatch, Operon::GaussianLikelihood<Operon::Scalar>> const mdlEval{&problem, &dtable};
        auto mdl = mdlEval(rng, ind).front();

        std::vector<std::tuple<std::string, double, std::string>> const stats{
            {"slope", a, format},
            {"intercept", b, format},
            {"r2", r2, format},
            {"rs", rs, format},
            {"mae", mae, format},
            {"mse", mse, format},
            {"rmse", rmse, format},
            {"nmse", nmse, format},
            {"mdl", mdl, format}
        };
        Operon::Reporter<void>::PrintStats(stats, /*printHeader=*/true);

        if (result.contains("shape-constraints-config")) {
            auto constraints = Operon::LoadShapeConstraints(result["shape-constraints-config"].as<std::string>());
            if (!constraints) { throw std::runtime_error("empty shape-constraints config path"); }
            Operon::Evaluator<Operon::ScalarDispatch> eval{&problem, &dtable, Operon::NMSE{}};
            Operon::ShapeConstrainedEvaluator shapeEval{&eval, &dtable, *constraints};
            shapeEval.SetBoundMode(Operon::ParseShapeBoundMode(result["shape-bound-mode"].as<std::string>()));
            auto const summary = shapeEval.Measure(model);
            fmt::print("shape_feasible {} shape_violation {}\n", summary.Feasible, summary.Violation);

            bool const tighten = result["tighten-range"].as<bool>();
            bool const sampleCheck = result.contains("sample-check");
            Operon::IntervalEvaluator::DomainMap const domains = (tighten || sampleCheck) ? BuildDomainMap(*constraints, ds) : Operon::IntervalEvaluator::DomainMap{};
            // Shared across both tighten and sample-check: m.Bound is the
            // scaled bound (TransformBound applies the same fitted linear
            // scaling used to check feasibility), so any raw-tree quantity
            // compared against it must go through the identical transform
            // or the two columns compare different quantities.
            std::optional<Operon::LinearScaling> const scaling = (tighten || sampleCheck) ? Operon::FitLinearScaling(model, problem, dtable, range) : std::nullopt;
            std::size_t const nSamples = sampleCheck ? result["sample-check"].as<std::size_t>() : 0;
            Operon::RandomGenerator sampleRng{0};
            // Fixed sample columns built once from the constraints' declared
            // domain box (not per-constraint) -- every constraint tree only
            // ever references a subset of these variables, so one shared
            // sample matrix is reused across constraints and evaluated
            // against whichever sliced tree ConstraintTreeFor returns.
            std::vector<std::string> sampleNames;
            std::vector<std::vector<Operon::Scalar>> sampleCols;
            if (sampleCheck) {
                sampleNames.reserve(constraints->Domains.size());
                sampleCols.reserve(constraints->Domains.size());
                for (auto const& [name, bound] : constraints->Domains) {
                    sampleNames.push_back(name);
                    std::vector<Operon::Scalar> col(nSamples);
                    for (auto& v : col) { v = Operon::Random::Uniform(sampleRng, bound.first, bound.second); }
                    sampleCols.push_back(std::move(col));
                }
            }
            std::optional<Operon::Dataset> sampleDs;
            if (sampleCheck) { sampleDs.emplace(sampleNames, sampleCols); }
            Operon::Range const sampleRange{0, nSamples};
            for (std::size_t i = 0; i < summary.Measurements.size(); ++i) {
                auto const& m = summary.Measurements[i];
                fmt::print("shape_measurement {} certified {} violation {}", i, m.Certified, m.Violation);
                if (m.Bound) { fmt::print(" bound [{}:{}]", m.Bound->first, m.Bound->second); }
                std::optional<Operon::Tree> ctreeStorage;
                if (tighten || sampleCheck) {
                    auto const& c = constraints->Constraints[i];
                    auto const variable = c.Op == Operon::ShapeConstraintOp::Identity
                        ? Operon::Hash{} : ds.GetVariable(c.Variable)->Hash;
                    ctreeStorage = ConstraintTreeFor(c, model, variable);
                }
                if (tighten) {
                    if (auto const& ctree = ctreeStorage) {
                        auto tr = Operon::TightenRange(*ctree, domains, ctree->GetCoefficients());
                        if (scaling) {
                            auto const& c = constraints->Constraints[i];
                            auto const [lo, hi] = c.Op == Operon::ShapeConstraintOp::Identity
                                ? scaling->ApplyToValueInterval(tr.inf(), tr.sup())
                                : scaling->ApplyToDerivativeInterval(tr.inf(), tr.sup());
                            tr = Operon::IntervalEvaluator::Interval(lo, hi);
                        }
                        if (std::isfinite(tr.inf()) && std::isfinite(tr.sup())) {
                            fmt::print(" tightened [{}:{}]", tr.inf(), tr.sup());
                        } else {
                            fmt::print(" tightened non-finite");
                        }
                    } else {
                        fmt::print(" tightened n/a (identically-zero or uncertifiable derivative)");
                    }
                }
                if (sampleCheck) {
                    if (auto const& ctree = ctreeStorage) {
                        using Interpreter = Operon::Interpreter<Operon::Scalar, Operon::ScalarDispatch>;
                        auto vals = Interpreter::Evaluate(*ctree, *sampleDs, sampleRange);
                        auto const nNonFinite = std::ranges::count_if(vals, [](auto v) { return !std::isfinite(v); });
                        if (nNonFinite == static_cast<std::ptrdiff_t>(vals.size())) {
                            fmt::print(" sampled all-non-finite (n={})", nSamples);
                        } else {
                            auto const [mn, mx] = std::ranges::minmax(
                                vals | std::views::filter([](auto v) { return std::isfinite(v); }));
                            // Sampled values come from the raw (unscaled)
                            // constraint tree, same as TightenRange's `tr`
                            // above -- apply the identical linear-scaling
                            // transform so this compares in the same units
                            // as `bound` (m.Bound, always scaled). Without
                            // this the reported range looks unsound against
                            // a scaled model even though nothing is wrong.
                            double slo = mn;
                            double shi = mx;
                            if (scaling) {
                                auto const& c = constraints->Constraints[i];
                                auto const [lo, hi] = c.Op == Operon::ShapeConstraintOp::Identity
                                    ? scaling->ApplyToValueInterval(slo, shi)
                                    : scaling->ApplyToDerivativeInterval(slo, shi);
                                slo = lo; shi = hi;
                            }
                            if (nNonFinite > 0) {
                                fmt::print(" sampled [{}:{}] (n={}, {} non-finite excluded)", slo, shi, nSamples, nNonFinite);
                            } else {
                                fmt::print(" sampled [{}:{}] (n={})", slo, shi, nSamples);
                            }
                        }
                    } else {
                        fmt::print(" sampled n/a (identically-zero or uncertifiable derivative)");
                    }
                }
                fmt::print("\n");
            }
        }

        if (opt->Iterations() > 0) {
            auto const& diag = Operon::Diagnostics(summary);
            if (summary.has_value()) {
                fmt::print("optimized_model {:infix:roundtrip}\n", Operon::Fmt::WithNames{model, ds});
            }
            fmt::print("optimization summary:\n");
            fmt::print("status: {}\n", summary.has_value());
            fmt::print("initial cost: {}\n", diag.InitialCost);
            fmt::print("final cost: {}\n", diag.FinalCost);
        }
    }
} // namespace

auto main(int argc, char** argv) -> int // NOLINT(bugprone-exception-escape)
{
    auto out = ParseOptions(argc, argv);
    if (!out.has_value()) { return EXIT_FAILURE; }
    auto const& result = out.value();

    Operon::Dataset ds(result["dataset"].as<std::string>(), /*hasHeader=*/true);
    auto infix = result.unmatched().front();
    auto model = Operon::InfixParser::Parse(infix, ds);

    if (result.contains("dump-tree-json")) {
        auto const path = result["dump-tree-json"].as<std::string>();
        std::ofstream out(path);
        out << Operon::Serialization::ToJson(model);
        if (!out) {
            fmt::print(stderr, "error: failed to write tree JSON to '{}'\n", path);
            return EXIT_FAILURE;
        }
    }

    Operon::ScalarDispatch const dtable;
    Operon::Range range{0, ds.Rows<std::size_t>()};
    if (result["range"].count() > 0) {
        auto res = scn::scan<std::size_t, std::size_t>(result["range"].as<std::string>(), "{}:{}");
        ENSURE(res);
        auto [a, b] = res->values();
        range = Operon::Range{a, b};
    }

    if (result["debug"].as<bool>()) {
        fmt::print("\nInput string:\n{}\n", infix);
        fmt::print("Parsed tree:\n{:infix:roundtrip}\n", Operon::Fmt::WithNames{model, ds});
        fmt::print("Data range: {}:{}\n", range.Start(), range.End());
        fmt::print("Scale: {}\n", result["scale"].count() > 0 ? result["scale"].as<std::string>() : std::string("auto"));
    }
    std::string const format = result["format"].as<std::string>();
    if (result["target"].count() > 0) {
        PrintTargetAnalysis(result, ds, range, dtable, format, model);
    } else {
        using Interpreter = Operon::Interpreter<Operon::Scalar, Operon::ScalarDispatch>;
        auto est = Interpreter::Evaluate(model, ds, range);
        std::string out{};
        for (auto v : est) {
            fmt::format_to(std::back_inserter(out), fmt::runtime(fmt::format("{{{}}}\n", format)), v);
        }
        fmt::print("{}", out);
    }

    return EXIT_SUCCESS;
}
