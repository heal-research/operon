// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include "../operon_test.hpp"

#include "nanobench.h"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/derivative_calculator.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"

#include <iomanip>

namespace dt = doctest;

namespace Operon::Test {

namespace detail {
    using T = Operon::detail::Array<Operon::Scalar>;

} // namespace detail

TEST_CASE("reverse mode" * dt::test_suite("[autodiff]")) {
    constexpr auto nrow{10};
    constexpr auto ncol{3};;

    Operon::Dataset::Matrix values(1, 2);
    values << 2, 5; // NOLINT

    Operon::RandomGenerator rng(0);
    Operon::Dataset ds(values);
    Operon::Map<std::string, Operon::Hash> variables;
    for (auto&& v : ds.Variables()) {
        variables.insert({v.Name, v.Hash});
    }

    Operon::Interpreter interpreter;
    Operon::Range range{0, ds.Rows()};

    Operon::DispatchTable<Operon::Scalar> dtable;

    auto derive = [&](std::string const& expr) {
        fmt::print(fmt::emphasis::bold, "f(x1, x2) = {}\n", expr);
        auto tree = Operon::InfixParser::Parse(expr, variables);
        fmt::print(fmt::fg(fmt::color::orange), "infix: {}\n", Operon::InfixFormatter::Format(tree, ds));

        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};
        Operon::ResidualEvaluator re(interpreter, tree, ds, target, range);

        Operon::Interpreteur interpreteur{ tree, ds, range, dtable};
        auto parameters = tree.GetCoefficients();

        Eigen::Matrix<Operon::Scalar, -1, -1> jacobian(range.Size(), parameters.size());
        auto autodiff = Operon::detail::Autodiff<decltype(re), Operon::Dual, Operon::Scalar, Eigen::ColMajor>;
        autodiff(re, parameters.data(), nullptr, jacobian.data());

        Operon::DerivativeCalculator dt(interpreteur);
        dt(parameters);

        std::cout << "J_forward: " << jacobian << "\n";
        std::cout << "J_reverse: " << dt.Jacobian() << "\n\n";
    };

    SUBCASE("0.51 * X1") { derive("0.51 * X1"); }

    SUBCASE("cos(sin(3))") { derive("cos(sin(3))"); }

    SUBCASE("exp(sin(2))") { derive("exp(sin(2))"); }

    SUBCASE("exp(x1) + exp(x2)") { derive("exp(X1) + exp(X2)"); }

    SUBCASE("log(x1) + x1x2 - sin(x2)") { derive("log(X1) + X1 * X2 - sin(X2)"); }

    SUBCASE("1 / x1") { derive("1 / X1"); }

    SUBCASE("1 / x1 * X2") { derive("1 / X1 * X2"); }

    SUBCASE("x1 + x2 + x1 * x2 + sin(x1) + sin(x2) + cos(x1) + cos(x2) + sin(x1 * x2) + cos(x1 * x2)") {
        derive("X1 + X2 + X1 * X2 + sin(X1) + sin(X2) + cos(X1) + cos(X2) + sin(X1 * X2) + cos(X1 * X2)");
    };

    SUBCASE("f(x1, x2) = ln(x1) + x1x2 - sin(x2)") {
        std::string expr{"sin(X2) + cos(X1) + sin(2) * sin(3) - sin(4) + cos(5) + 1 + 2 + 3 - cos(6) * cos(7) * sin(8)"};
    }

    SUBCASE("random trees") {
        Operon::PrimitiveSet pset(Operon::PrimitiveSet::Arithmetic | Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Sin | Operon::NodeType::Cos);
        //Operon::PrimitiveSet pset(Operon::PrimitiveSet::Arithmetic);
        Operon::BalancedTreeCreator btc(pset, ds.Variables());
        Operon::UniformCoefficientInitializer initializer;

        constexpr auto n{100'000};
        constexpr auto maxsize{50};
        constexpr auto mindepth{1};
        constexpr auto maxdepth{1000};

        std::uniform_int_distribution<size_t> dist(1, maxsize);

        // comparison precision
        constexpr auto epsilon{1e-4};

        Operon::Interpreter interpreter;
        Operon::Range range(0, ds.Rows());
        Operon::Vector<Operon::Scalar> target(ds.Rows());

        Operon::DispatchTable<Operon::Scalar> dtable;

        for (auto i = 0; i < n; ++i) {
            auto tree = btc(rng, dist(rng), mindepth, maxdepth);
            initializer(rng, tree);

            auto parameters = tree.GetCoefficients();

            // forward mode
            Eigen::Matrix<Operon::Scalar, -1, -1> jacobian(range.Size(), parameters.size());
            Operon::ResidualEvaluator re(interpreter, tree, ds, target, range);
            auto autodiff = Operon::detail::Autodiff<decltype(re), Operon::Dual, Operon::Scalar, Eigen::ColMajor>;
            autodiff(re, parameters.data(), nullptr, jacobian.data());

            // reverse mode
            Operon::Interpreteur interpreteur{ tree, ds, range, dtable};
            Operon::DerivativeCalculator dt(interpreteur);
            dt(parameters);

            constexpr auto precision{10};
            auto finite{ std::isfinite(jacobian.sum()) };
            auto areEqual{ jacobian.array().isApprox(dt.Jacobian(), epsilon) };
            auto ok = !finite || areEqual;

            if(!ok) {
                fmt::print(fmt::fg(fmt::color::orange), "infix: {}\n", Operon::InfixFormatter::Format(tree, ds));
                std::cout << std::setprecision(precision) << "J_forward: " << jacobian << "\n";
                std::cout << std::setprecision(precision) << "J_reverse: " << dt.Jacobian() << "\n\n";
            }
            CHECK(ok);
        }
    }
}

} // namespace Operon::Test
