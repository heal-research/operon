// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <string>

#include <cxxopts.hpp>
#include <fmt/core.h>
#include <scn/scn.h>

#include "operon/core/dataset.hpp"
#include "operon/core/format.hpp"
#include "operon/parser/infix.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/evaluator.hpp"

auto main(int argc, char** argv) -> int
{
    cxxopts::Options opts("operon_parse_model", "Parse and evaluate a model in infix form");

    opts.add_options()
        ("dataset", "Dataset file name (csv) (required)", cxxopts::value<std::string>())
        ("target", "Name of the target variable (if none provided, model output will be printed)", cxxopts::value<std::string>())
        ("range", "Data range [A:B)", cxxopts::value<std::string>())
        ("scale", "Linear scaling slope:intercept", cxxopts::value<std::string>())
        ("help", "Print help");

    opts.allow_unrecognised_options();

    cxxopts::ParseResult result;
    try {
        result = opts.parse(argc, argv);
    }  catch (cxxopts::OptionParseException const& ex) {
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
    auto tmap = Operon::InfixParser::DefaultTokens();
    robin_hood::unordered_flat_map<std::string, Operon::Hash> vmap;
    for (auto const& v : ds.Variables()) {
        vmap.insert({ v.Name, v.Hash });
    }
    auto model = Operon::InfixParser::Parse(infix, tmap, vmap);
    Operon::Interpreter interpreter;
    Operon::Range range{0, ds.Rows()};
    if (result["range"].count() > 0) {
        size_t a{0};
        size_t b{0};
        scn::scan(result["range"].as<std::string>(), "{}:{}", a, b);
        range = Operon::Range{a, b};
        fmt::print("range = {}:{}\n", a, b);
    }

    auto est = interpreter.Evaluate<Operon::Scalar>(model, ds, range);

    if (result["target"].count() > 0) {
        auto tgt = ds.GetValues(result["target"].as<std::string>()).subspan(range.Start(), range.Size());

        Operon::Scalar a{0};
        Operon::Scalar b{0};
        if (result["scale"].count() > 0) {
            scn::scan(result["scale"].as<std::string>(), "{}:{}", a, b);
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
        fmt::print("   slope intercept       r2       rs      mae      mse     rmse     nmse\n");
        fmt::print("{:>8.4g} {:>8.4g} {:>8.4g} {:>8.4g} {:>8.4g} {:>8.4g} {:>8.4g} {:>8.4g}\n", a, b, r2, rs, mae, mse, rmse, nmse);
    } else {
        for (auto v : est) {
            fmt::print("{:.6f}\n", v);
        }
    }

    return EXIT_SUCCESS;
}
