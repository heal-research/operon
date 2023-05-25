// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <string>

#include <cxxopts.hpp>
#include <fmt/core.h>
#include <scn/scn.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/parser/infix.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/evaluator.hpp"
#include "util.hpp"

auto main(int argc, char** argv) -> int
{
    cxxopts::Options opts("operon_parse_model", "Parse and evaluate a model in infix form");

    opts.add_options()
        ("dataset", "Dataset file name (csv) (required)", cxxopts::value<std::string>())
        ("target", "Name of the target variable (if none provided, model output will be printed)", cxxopts::value<std::string>())
        ("range", "Data range [A:B)", cxxopts::value<std::string>())
        ("scale", "Linear scaling slope:intercept", cxxopts::value<std::string>())
        ("debug", "Show some debugging information", cxxopts::value<bool>()->default_value("false"))
        ("format", "Format string (see https://fmt.dev/latest/syntax.html)", cxxopts::value<std::string>()->default_value(":>#8.4g"))
        ("help", "Print help");

    opts.allow_unrecognised_options();

    cxxopts::ParseResult result;
    try {
        result = opts.parse(argc, argv);
    } catch (cxxopts::exceptions::parsing const& ex) {
        fmt::print(stderr, "error: {}. rerun with --help to see available options.\n", ex.what());
        return EXIT_FAILURE;
    };

    if (result.arguments().empty() || result.count("help") > 0) {
        fmt::print("{}\n", opts.help());
        return EXIT_SUCCESS;
    }

    if (result.count("dataset") == 0) {
        fmt::print(stderr, "error: no dataset was specified.\n");
        return EXIT_FAILURE;
    }

    if (result.unmatched().empty()) {
        fmt::print(stderr, "error: no infix string was provided.\n");
        return EXIT_FAILURE;
    }

    Operon::Dataset ds(result["dataset"].as<std::string>(), /*hasHeader=*/true);
    auto infix = result.unmatched().front();
    Operon::Map<std::string, Operon::Hash> vars;
    for (auto const& v : ds.GetVariables()) {
        vars.insert({ v.Name, v.Hash });
    }
    auto model = Operon::InfixParser::Parse(infix, vars);

    Operon::Interpreter interpreter;
    Operon::Range range{0, ds.Rows<std::size_t>()};
    if (result["range"].count() > 0) {
        size_t a{0};
        size_t b{0};
        [[maybe_unused]] auto s = scn::scan(result["range"].as<std::string>(), "{}:{}", a, b);
        range = Operon::Range{a, b};
    }

    int constexpr defaultPrecision{6};
    if (result["debug"].as<bool>()) {
        fmt::print("\nInput string:\n{}\n", infix);
        fmt::print("Parsed tree:\n{}\n", Operon::InfixFormatter::Format(model, ds, defaultPrecision));
        fmt::print("Data range: {}:{}\n", range.Start(), range.End());
        fmt::print("Scale: {}\n", result["scale"].count() > 0 ? result["scale"].as<std::string>() : std::string("auto"));
    }

    auto est = interpreter(model, ds, range);

    std::string format = result["format"].as<std::string>();
    if (result["target"].count() > 0) {
        auto tgt = ds.GetValues(result["target"].as<std::string>()).subspan(range.Start(), range.Size());

        Operon::Scalar a{0};
        Operon::Scalar b{0};
        if (result["scale"].count() > 0) {
            [[maybe_unused]] auto s = scn::scan(result["scale"].as<std::string>(), "{}:{}", a, b);
        } else {
            auto [a_, b_] = Operon::FitLeastSquares(est, tgt);
            a = static_cast<Operon::Scalar>(a_);
            b = static_cast<Operon::Scalar>(b_);
        }

        std::transform(est.begin(), est.end(), est.begin(), [&](auto v) { return v * a + b; });
        auto r2 = -Operon::R2{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto rs = -Operon::C2{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto mae = Operon::MAE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto mse = Operon::MSE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto rmse = Operon::RMSE{}(Operon::Span<Operon::Scalar>{est}, tgt);
        auto nmse = Operon::NMSE{}(Operon::Span<Operon::Scalar>{est}, tgt);

        std::vector<std::tuple<std::string, double, std::string>> stats{
            {"slope", a, format},
            {"intercept", b, format},
            {"r2", r2, format},
            {"rs", rs, format},
            {"mae", mae, format},
            {"mse", mse, format},
            {"rmse", rmse, format},
            {"nmse", nmse, format},
        };
        Operon::PrintStats(stats);
    } else {
        auto out = fmt::memory_buffer();
        for (auto v : est) {
            fmt::format_to(std::back_inserter(out), fmt::runtime(fmt::format("{{{}}}\n", format)), v);
        }
        fmt::print("{}", fmt::to_string(out));
    }

    return EXIT_SUCCESS;
}
