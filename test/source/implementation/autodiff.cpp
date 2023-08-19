// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include <random>

#include "../operon_test.hpp"

#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/core/problem.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/parser/infix.hpp"

#include <iomanip>

namespace dt = doctest;

namespace Operon::Test {

TEST_CASE("reverse mode" * dt::test_suite("[autodiff]")) {
    Operon::Dataset::Matrix values(1, 2);
    values << 2, 3; // NOLINT

    Operon::RandomGenerator rng(0UL);
    Operon::Dataset ds(values);
    ds.SetVariableNames({"x", "y"});
    Operon::Map<std::string, Operon::Hash> variables;
    for (auto&& v : ds.GetVariables()) {
        variables.insert({v.Name, v.Hash});
    }

    Operon::DispatchTable<Operon::Scalar> dtable;

    Operon::Range range{0, ds.Rows<std::size_t>()};
    Operon::Problem problem(ds, range, {0, 1});

    auto derive = [&](std::string const& expr) {
        fmt::print(fmt::emphasis::bold, "f(x, y) = {}\n", expr);
        auto tree = Operon::InfixParser::Parse(expr, variables, /*reduce=*/true);
        for (auto& n : tree.Nodes()) {
            n.Optimize = n.IsLeaf();
            //if (n.Arity > 0) { n.Value = 2; }
        }
        fmt::print(fmt::fg(fmt::color::orange), "infix: {}\n", Operon::InfixFormatter::Format(tree, ds));

        fmt::print("{}\n", Operon::DotFormatter::Format(tree, ds));

        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};
        auto parameters = tree.GetCoefficients();

        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> interpreter{dtable, ds, tree};
        auto rev = interpreter.JacRev(parameters, range);
        auto fwd = interpreter.JacFwd(parameters, range);
        std::cout << "rev: " << rev << "\n";
        std::cout << "fwd: " << rev << "\n";

        //Operon::DispatchTable<Operon::Scalar, Operon::Dual> dt;
        //Autodiff::DerivativeCalculator<decltype(dt)> dc(dt);
        //Eigen::Array<Operon::Scalar, -1, -1> jacobian(range.Size(), parameters.size());
        //std::vector<Operon::Scalar> result(range.Size());
        //dc.ForwardModeJet(tree, ds, range, parameters, {result}, {jacobian.data(), range.Size() * parameters.size()});
        //std::cout << "jet: " << jacobian << "\n";
    };

    SUBCASE("0.51 * x") { derive("0.51 * x"); }

    SUBCASE("0.51 * 0.74") { derive("0.51 * 0.74"); }

    SUBCASE("2.53 / 1.46") { derive("2.53 / 1.46"); }

    SUBCASE("3 ^ 2") { derive("3 ^ 2"); }

    SUBCASE("sin(2)") { derive("sin(2)"); }

    SUBCASE("y * sin(x) | at (x, y) = (2, 2)") { derive("sin(2)"); }

    SUBCASE("sin(x) + cos(y) | at (x, y) = (2, 3)") { derive("sin(2) + cos(3)"); }

    SUBCASE("sin(x) * cos(y) | at (x, y) = (2, 3)") { derive("sin(2) * cos(3)"); }

    SUBCASE("0.5 * sin(x) + 0.7 * cos(y) | at (x, y) = (2, 3)") { derive("0.5 * sin(2) + 0.7 * cos(3)"); }

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

    SUBCASE("Expr A") {
        std::string const expr = "((0.78 / ((-1.12) * X8)) / (((((-0.61) * X3) * 0.82) / (((-0.22) * X6) / 1.77)) / (((-0.16) - 0.50) - (((-0.46) * X4) - ((-0.03) * X9)))))";


        Operon::Dataset ds("./data/Poly-10.csv", /*hasHeader=*/true);
        Operon::Map<std::string, Operon::Hash> vars;
        for (auto const& v : ds.GetVariables()) {
            vars.insert({ v.Name, v.Hash });
        }

        auto tree = InfixParser::Parse(expr, vars);
        auto coeff = tree.GetCoefficients();
        Operon::Range range(0, 10); // NOLINT

        DispatchTable<Operon::Scalar> dt;
        auto jacrev = Operon::Interpreter<Operon::Scalar, DispatchTable<Operon::Scalar>>{dt, ds, tree}.JacRev(coeff, range); 
        auto jacfwd = Operon::Interpreter<Operon::Scalar, DispatchTable<Operon::Scalar>>{dt, ds, tree}.JacFwd(coeff, range); 

        std::cout << "jacrev:\n" << jacrev << "\n\n";
        std::cout << "jacfwd:\n" << jacfwd << "\n\n";

        auto vals = Interpreter<Operon::Scalar, decltype(dt)>::Evaluate(tree, ds, range);
        fmt::print("values: {}\n", vals);
    }

    SUBCASE("random trees") {
        using Operon::NodeType;
        Operon::PrimitiveSet pset(Operon::PrimitiveSet::Arithmetic |
                Operon::NodeType::Pow | Operon::NodeType::Aq | Operon::NodeType::Square |
                Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Abs |
                Operon::NodeType::Logabs | Operon::NodeType::Log1p |
                Operon::NodeType::Sin | Operon::NodeType::Asin |
                Operon::NodeType::Cos | Operon::NodeType::Acos |
                Operon::NodeType::Sinh | Operon::NodeType::Cosh |
                Operon::NodeType::Tan | Operon::NodeType::Atan |
                Operon::NodeType::Tanh | Operon::NodeType::Cbrt |
                Operon::NodeType::Fmin | Operon::NodeType::Fmax |
                Operon::NodeType::Sqrt | Operon::NodeType::Sqrtabs);
        //Operon::PrimitiveSet pset(Operon::PrimitiveSet::Arithmetic);
        Operon::BalancedTreeCreator btc(pset, problem.GetInputs());
        Operon::UniformCoefficientInitializer initializer;

        constexpr auto n{1'000'000};
        constexpr auto maxsize{5};
        constexpr auto mindepth{1};
        constexpr auto maxdepth{1000};

        std::uniform_int_distribution<size_t> length(1, maxsize);
        std::uniform_real_distribution<Operon::Scalar> dist(0, 1);
        std::bernoulli_distribution bernoulli(0.5); // NOLINT

        // comparison precision
        constexpr auto epsilon{1e-4};

        for (auto i = 0; i < n; ++i) {
            auto tree = btc(rng, length(rng), mindepth, maxdepth);
            for (auto& node : tree.Nodes()) {
                node.Optimize = bernoulli(rng);
                node.Value = dist(rng);
            }
            //tree.Nodes().back().Optimize = false; // it does not make sense to optimize the tree root?
            initializer(rng, tree);

            auto parameters = tree.GetCoefficients();

            Eigen::Array<Operon::Scalar, -1, -1> jfwd = Operon::Interpreter<Operon::Scalar, decltype(dtable)>{dtable, ds, tree}.JacFwd(parameters, range);
            Eigen::Array<Operon::Scalar, -1, -1> jrev = Operon::Interpreter<Operon::Scalar, decltype(dtable)>{dtable, ds, tree}.JacRev(parameters, range);

            constexpr auto precision{20};
            auto finite{ std::isfinite(jrev.sum()) };
            auto areEqual{ jrev.isApprox(jfwd, epsilon) };
            auto ok = !finite || areEqual;

            if(!ok) {
                fmt::print(fmt::fg(fmt::color::orange), "infix: {}\n", Operon::InfixFormatter::Format(tree, ds, precision));
                std::cout << std::setprecision(precision) << "J_forward: " << jfwd << "\n";
                std::cout << std::setprecision(precision) << "J_reverse: " << jrev << "\n";
            }
            CHECK(ok);
        }
    }
}

} // namespace Operon::Test
