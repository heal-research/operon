// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include "../operon_test.hpp"

#include "nanobench.h"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"
#include "operon/autodiff/autodiff.hpp"

#include <iomanip>

namespace dt = doctest;

namespace Operon::Test {

TEST_CASE("reverse mode" * dt::test_suite("[autodiff]")) {
    constexpr auto nrow{10};
    constexpr auto ncol{3};;

    Operon::Dataset::Matrix values(1, 2);
    values << 2, 5; // NOLINT

    Operon::RandomGenerator rng(0);
    Operon::Dataset ds(values);
    ds.SetVariableNames({"x", "y"});
    Operon::Map<std::string, Operon::Hash> variables;
    for (auto&& v : ds.Variables()) {
        variables.insert({v.Name, v.Hash});
    }

    Operon::GenericInterpreter<Operon::Scalar> interpreter;
    Operon::GenericInterpreter<Operon::Scalar, Operon::Dual> interpreterDual;

    Operon::Autodiff::Reverse::DerivativeCalculator rcalc{ interpreter };
    Operon::Autodiff::Forward::DerivativeCalculator fcalc{ interpreterDual };

    Operon::Range range{0, ds.Rows()};

    Operon::DispatchTable<Operon::Scalar> dtable;

    auto derive = [&](std::string const& expr) {
        fmt::print(fmt::emphasis::bold, "f(x, y) = {}\n", expr);
        auto tree = Operon::InfixParser::Parse(expr, variables);
        fmt::print(fmt::fg(fmt::color::orange), "infix: {}\n", Operon::InfixFormatter::Format(tree, ds));

        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};
        auto parameters = tree.GetCoefficients();
        auto jfwd = fcalc(tree, ds, { parameters.data(), parameters.size() }, range);
        auto jrev = rcalc(tree, ds, { parameters.data(), parameters.size() }, range);

        std::cout << "J_forward: " << jfwd << "\n";
        std::cout << "J_reverse: " << jrev << "\n\n";
    };

    SUBCASE("0.51 * x") { derive("0.51 * x"); }

    SUBCASE("cos(sin(3))") { derive("cos(sin(3))"); }

    SUBCASE("exp(sin(2))") { derive("exp(sin(2))"); }

    SUBCASE("exp(x) + exp(y)") { derive("exp(x) + exp(y)"); }

    SUBCASE("log(x) + xy - sin(y)") { derive("log(x) + x * y - sin(y)"); }

    SUBCASE("1 / x") { derive("1 / x"); }

    SUBCASE("1 / x * y") { derive("1 / x * y"); }

    SUBCASE("sqrt(x) + sqrt(y)") { derive("sqrt(x) + sqrt(y)"); }

    SUBCASE("x + y + x * y + sin(x) + sin(y) + cos(x) + cos(y) + sin(x * y) + cos(x * y)") {
        derive("x + y + x * y + sin(x) + sin(y) + cos(x) + cos(y) + sin(x * y) + cos(x * y)");
    };

    SUBCASE("ln(x) + xy - sin(y)") {
        std::string expr{"sin(y) + cos(x) + sin(2) * sin(3) - sin(4) + cos(5) + 1 + 2 + 3 - cos(6) * cos(7) * sin(8)"};
    }

    SUBCASE("exp(x)") { derive("exp(x)"); }

    SUBCASE("sin(exp(x))") { derive("sin(exp(x))"); }

    SUBCASE("tan(x)") { derive("tan(x)"); }

    SUBCASE("tan(x-2)") { derive("tan(x - 2)"); }

    SUBCASE("tan(x+y)") { derive("tan(x + y)"); }

    SUBCASE("sin(exp((0.0798202157 / 0.0111869667)))") { derive("sin(exp((0.0798202157 / 0.0111869667)))"); }

    SUBCASE("pow(x, 2)") { derive("x ^ 2"); }

    SUBCASE("pow(x, 3)") { derive("x ^ 3"); }

    SUBCASE("asin(x)") { derive("asin(x)"); }

    SUBCASE("acos(x)") { derive("acos(x)"); }

    SUBCASE("atan(x)") { derive("atan(x)"); }

    SUBCASE("random trees") {
        using Operon::NodeType;
        Operon::PrimitiveSet pset(Operon::PrimitiveSet::Arithmetic |
                Operon::NodeType::Pow | Operon::NodeType::Aq |
                Operon::NodeType::Exp | Operon::NodeType::Log |
                Operon::NodeType::Logabs | Operon::NodeType::Log1p |
                Operon::NodeType::Sin | Operon::NodeType::Asin |
                Operon::NodeType::Cos | Operon::NodeType::Acos |
                Operon::NodeType::Tan | Operon::NodeType::Atan |
                Operon::NodeType::Tanh | Operon::NodeType::Cbrt |
                Operon::NodeType::Sqrt | Operon::NodeType::Sqrtabs);
        //Operon::PrimitiveSet pset(Operon::PrimitiveSet::Arithmetic);
        Operon::BalancedTreeCreator btc(pset, ds.Variables());
        Operon::UniformCoefficientInitializer initializer;

        constexpr auto n{1'000'000};
        constexpr auto maxsize{5};
        constexpr auto mindepth{1};
        constexpr auto maxdepth{1000};

        std::uniform_int_distribution<size_t> dist(1, maxsize);

        // comparison precision
        constexpr auto epsilon{1e-4};

        for (auto i = 0; i < n; ++i) {
            auto tree = btc(rng, dist(rng), mindepth, maxdepth);
            initializer(rng, tree);

            auto parameters = tree.GetCoefficients();

            // forward mode
            auto jfwd = fcalc(tree, ds, { parameters.data(), parameters.size() }, range);

            // reverse mode
            auto jrev = rcalc(tree, ds, { parameters.data(), parameters.size() }, range);

            constexpr auto precision{20};
            auto finite{ std::isfinite(jfwd.sum()) };
            auto areEqual{ jfwd.isApprox(jrev, epsilon) };
            auto ok = !finite || areEqual;

            if(!ok) {
                fmt::print(fmt::fg(fmt::color::orange), "infix: {}\n", Operon::InfixFormatter::Format(tree, ds, precision));
                std::cout << std::setprecision(precision) << "J_forward: " << jfwd << "\n";
                std::cout << std::setprecision(precision) << "J_reverse: " << jrev << "\n\n";
            }
            CHECK(ok);
        }
    }
}

} // namespace Operon::Test
