// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <memory>
#include <string>


#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/evaluator.hpp"
#include "util.hpp"
#include "reporter.hpp"

#include <cxxopts.hpp>
#include <fmt/core.h>
#include <scn/scan.h>
#include <outcome.hpp>

namespace outcome = OUTCOME_V2_NAMESPACE;

namespace {
    enum class ParseError : std::uint8_t  {
        Success        = 0,
        MissingDataset = 1,
        MissingInfix   = 2,
        NoOptions      = 3,
        UnknownError   = 4
    };

    auto ParseOptions(int argc, char** argv) noexcept -> outcome::unchecked<cxxopts::ParseResult, ParseError> {
        cxxopts::Options opts("operon_parse_model", "Parse and evaluate a model in infix form");

        opts.add_options()
            ("dataset", "Dataset file name (csv) (required)", cxxopts::value<std::string>())
            ("target", "Name of the target variable (if none provided, model output will be printed)", cxxopts::value<std::string>())
            ("range", "Data range [A:B)", cxxopts::value<std::string>())
            ("scale", "Linear scaling slope:intercept", cxxopts::value<std::string>())
            ("optimizer", "Optimizer for model coefficients (lm, lbfgs, sgd)", cxxopts::value<std::string>()->default_value("lm"))
            ("likelihood", "Optimizer loss function (gaussian, poisson)", cxxopts::value<std::string>()->default_value("gaussian"))
            ("iterations", "Optimizer iterations", cxxopts::value<int>()->default_value("50"))
            ("debug", "Show some debugging information", cxxopts::value<bool>()->default_value("false"))
            ("format", "Format string (see https://fmt.dev/latest/syntax.html)", cxxopts::value<std::string>()->default_value(":>#8.4g"))
            ("help", "Print help");

        opts.allow_unrecognised_options();

        cxxopts::ParseResult result;
        try {
            result = opts.parse(argc, argv);
        } catch (cxxopts::exceptions::parsing const& ex) {
            fmt::print(stderr, "error: {}. rerun with --help to see available options.\n", ex.what());
            return ParseError::UnknownError;
        };

        if (result.arguments().empty() || result.count("help") > 0) {
            fmt::print("{}\n", opts.help());
            return ParseError::NoOptions;
        }

        if (result.count("dataset") == 0) {
            fmt::print(stderr, "error: no dataset was specified.\n");
            return ParseError::MissingDataset;
        }

        if (result.unmatched().empty()) {
            fmt::print(stderr, "error: no infix string was provided.\n");
            return ParseError::MissingInfix;
        }
        return result;
    }

    auto ParseOptimizer(Operon::DefaultDispatch const* dtable, Operon::Problem const* problem, std::string const& optimizer, std::string const& likelihood) {
        std::unique_ptr<Operon::OptimizerBase> opt;

        if (optimizer == "lm") {
            opt = std::make_unique<Operon::LevenbergMarquardtOptimizer<Operon::DefaultDispatch>>(dtable, problem);
        } else if (optimizer == "lbfgs") {
            if (likelihood == "gaussian") {
                opt = std::make_unique<Operon::LBFGSOptimizer<Operon::DefaultDispatch, Operon::GaussianLikelihood<Operon::Scalar>>>(dtable, problem);
            } else if (likelihood == "poisson") {
                opt = std::make_unique<Operon::LBFGSOptimizer<Operon::DefaultDispatch, Operon::PoissonLikelihood<Operon::Scalar>>>(dtable, problem);
            }
        } else if (optimizer == "sgd") {
            if (likelihood == "gaussian") {
                opt = std::make_unique<Operon::SGDOptimizer<Operon::DefaultDispatch, Operon::GaussianLikelihood<Operon::Scalar>>>(dtable, problem);
            } else if (likelihood == "poisson") {
                opt = std::make_unique<Operon::SGDOptimizer<Operon::DefaultDispatch, Operon::PoissonLikelihood<Operon::Scalar>>>(dtable, problem);
            }
        }
        return opt;
    }
} // namespace

auto main(int argc, char** argv) -> int
{
    auto out = ParseOptions(argc, argv);
    if (!out.has_value()) { return EXIT_FAILURE; }
    auto const& result = out.value();

    Operon::Dataset ds(result["dataset"].as<std::string>(), /*hasHeader=*/true);
    auto infix = result.unmatched().front();
    Operon::Map<std::string, Operon::Hash> vars;
    for (auto const& v : ds.GetVariables()) {
        vars.insert({ v.Name, v.Hash });
    }
    auto model = Operon::InfixParser::Parse(infix, vars);

    Operon::DefaultDispatch dtable;
    Operon::Range range{0, ds.Rows<std::size_t>()};
    if (result["range"].count() > 0) {
        auto res = scn::scan<std::size_t, std::size_t>(result["range"].as<std::string>(), "{}:{}");
        ENSURE(res);
        auto [a, b] = res->values();
        range = Operon::Range{a, b};
    }

    int constexpr defaultPrecision{6};
    if (result["debug"].as<bool>()) {
        fmt::print("\nInput string:\n{}\n", infix);
        fmt::print("Parsed tree:\n{}\n", Operon::InfixFormatter::Format(model, ds, defaultPrecision));
        fmt::print("Data range: {}:{}\n", range.Start(), range.End());
        fmt::print("Scale: {}\n", result["scale"].count() > 0 ? result["scale"].as<std::string>() : std::string("auto"));
    }
    using Interpreter = Operon::Interpreter<Operon::Scalar, Operon::DefaultDispatch>;
    auto est = Interpreter::Evaluate(model, ds, range);

    std::string format = result["format"].as<std::string>();
    if (result["target"].count() > 0) {
        auto tgt = ds.GetValues(result["target"].as<std::string>()).subspan(range.Start(), range.Size());

        Operon::Scalar a{0};
        Operon::Scalar b{0};
        if (result["scale"].count() > 0) {
            auto res = scn::scan<Operon::Scalar, Operon::Scalar>(result["scale"].as<std::string>(), "{}:{}");
            ENSURE(res);
            a = std::get<0>(res->values());
            b = std::get<1>(res->values());
        } else {
            auto [a_, b_] = Operon::FitLeastSquares(est, tgt);
            a = static_cast<Operon::Scalar>(a_);
            b = static_cast<Operon::Scalar>(b_);
        }

        std::ranges::transform(est, est.begin(), [&](auto v) { return (v * a) + b; });
        auto r2 = -Operon::R2{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto rs = -Operon::C2{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto mae = Operon::MAE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto mse = Operon::MSE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto rmse = Operon::RMSE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto nmse = Operon::NMSE{}(Operon::Span<Operon::Scalar>{est}, tgt);

        Operon::Problem problem{&ds};
        problem.SetTrainingRange(range);
        problem.SetTestRange(range);
        Operon::RandomGenerator rng{0};
        Operon::Individual ind;
        ind.Genotype = model;

        Operon::Interpreter<Operon::Scalar, Operon::DefaultDispatch> interpreter{&dtable, &ds, &ind.Genotype};
        Operon::MinimumDescriptionLengthEvaluator<Operon::DefaultDispatch, Operon::GaussianLikelihood<Operon::Scalar>> mdlEval{&problem, &dtable};
        auto mdl = mdlEval(rng, ind).front();
        auto opt = ParseOptimizer(&dtable, &problem, result["optimizer"].as<std::string>(), result["likelihood"].as<std::string>());
        opt->SetIterations(result["iterations"].as<int>());
        auto summary = opt->Optimize(rng, model);

        std::vector<std::tuple<std::string, double, std::string>> stats{
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

        if (opt->Iterations() > 0) {
            fmt::print("optimization summary:\n");
            fmt::print("status: {}\n", summary.Success);
            fmt::print("initial cost: {}\n", summary.InitialCost);
            fmt::print("final cost: {}\n", summary.FinalCost);
        }
    } else {
        std::string out{};
        for (auto v : est) {
            fmt::format_to(std::back_inserter(out), fmt::runtime(fmt::format("{{{}}}\n", format)), v);
        }
        fmt::print("{}", out);
    }

    return EXIT_SUCCESS;
}
