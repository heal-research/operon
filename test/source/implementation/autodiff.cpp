// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <random>

#include "../operon_test.hpp"

#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/core/problem.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/dual.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/parser/infix.hpp"


#include <iomanip>
#include <iostream>
#include <fstream>

namespace dt = doctest;

namespace Operon::Test {

TEST_CASE("reverse mode" * dt::test_suite("[autodiff]")) {
    Operon::Dataset::Matrix values(1, 2);
    values << 1, 1; // NOLINT

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

        // fmt::print("{}\n", Operon::DotFormatter::Format(tree, ds));

        std::vector<Operon::Scalar> y(ds.Rows());
        Operon::Span<Operon::Scalar> target{y.data(), y.size()};
        auto parameters = tree.GetCoefficients();

        auto [res, jac] = Util::Autodiff(tree, ds, range);

        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> interpreter{dtable, ds, tree};
        auto est = interpreter.Evaluate(parameters, range);
        auto rev = interpreter.JacRev(parameters, range);
        auto fwd = interpreter.JacFwd(parameters, range);
        std::cout << "x = " << values(0) << ", y = " << values(1) << ", e = " << est[0] << "\n";
        // std::cout << "xad: " << ders_xad.transpose() << "\n";
        fmt::print("jac: {}\n", jac);
        std::cout << "rev: " << rev << "\n";
        std::cout << "fwd: " << rev << "\n";


        //Operon::DispatchTable<Operon::Scalar, Operon::Dual> dt;
        //Autodiff::DerivativeCalculator<decltype(dt)> dc(dt);
        //Eigen::Array<Operon::Scalar, -1, -1> jacobian(range.Size(), parameters.size());
        //std::vector<Operon::Scalar> result(range.Size());
        //dc.ForwardModeJet(tree, ds, range, parameters, {result}, {jacobian.data(), range.Size() * parameters.size()});
        //std::cout << "jet: " << jacobian << "\n";
    };

    auto generateTrees = [&](auto pset, auto n, auto l) {
        constexpr auto mindepth{1};
        constexpr auto maxdepth{1000};
        // constexpr auto pi = std::numbers::pi_v<Operon::Scalar>;
        std::uniform_int_distribution<size_t> length(1, l);
        std::bernoulli_distribution bernoulli(0.5); // NOLINT
        // std::uniform_real_distribution<Operon::Scalar> dist(-pi, +pi);
        std::uniform_real_distribution<Operon::Scalar> dist(-10.F, +10.F);
        Operon::BalancedTreeCreator btc(pset, ds.VariableHashes());

        std::vector<Operon::Tree> trees;
        trees.reserve(n);

        for (auto i = 0; i < n; ++i) {
            auto tree = btc(rng, l, mindepth, maxdepth);
            for (auto& node : tree.Nodes()) {
                node.Optimize = bernoulli(rng) || node.IsLeaf();
                // node.Optimize = node.IsLeaf();
                if (node.IsLeaf()) {
                    node.Value = dist(rng);
                }
            }
            trees.push_back(tree);
        }

        return trees;
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
        // Operon::PrimitiveSet pset(Operon::PrimitiveSet::Arithmetic |
        //         Operon::NodeType::Pow | Operon::NodeType::Aq | Operon::NodeType::Square |
        //         Operon::NodeType::Exp | Operon::NodeType::Log | Operon::NodeType::Abs |
        //         Operon::NodeType::Logabs | Operon::NodeType::Log1p |
        //         Operon::NodeType::Sin | Operon::NodeType::Asin |
        //         Operon::NodeType::Cos | Operon::NodeType::Acos |
        //         Operon::NodeType::Sinh | Operon::NodeType::Cosh |
        //         Operon::NodeType::Tan | Operon::NodeType::Atan |
        //         Operon::NodeType::Tanh | Operon::NodeType::Cbrt |
        //         Operon::NodeType::Fmin | Operon::NodeType::Fmax |
        //         Operon::NodeType::Sqrt | Operon::NodeType::Sqrtabs);
        // Operon::PrimitiveSet pset(Operon::PrimitiveSet::Arithmetic |
        //     Operon::NodeType::Exp | Operon::NodeType::Log |
        //     Operon::NodeType::Pow | Operon::NodeType::Sqrt |
        //     Operon::NodeType::Sin | Operon::NodeType::Cos |
        //     Operon::NodeType::Tanh
        // );


        Operon::PrimitiveSet pset;

        constexpr auto n{1'000'000};

        // n-ary
        pset.SetConfig(NodeType::Fmin | NodeType::Fmax);

        // unary
        pset.SetConfig(NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Sqrt | NodeType::Tanh | NodeType::Constant);
        auto trees = generateTrees(pset, n/2, 2);

        // binary
        pset.SetConfig(NodeType::Div | NodeType::Aq | NodeType::Pow | NodeType::Constant);
        auto tmp = generateTrees(pset, n/2, 3);
        std::ranges::copy(tmp, std::back_inserter(trees));

        fmt::print("{}\n", Operon::InfixFormatter::Format(tmp.front(), ds));

        // comparison precision
        auto testPrecision = [&](auto epsilon) {
            auto count{0UL};

            Operon::Map<Operon::NodeType, std::tuple<std::size_t, float>> counts;
            for (auto const& tree : trees) {
                auto parameters = tree.GetCoefficients();

                auto [res, jac] = Util::Autodiff(tree, ds, range);
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, -1>> jjet(jac.data(), range.Size(), parameters.size());

                Operon::Interpreter<Operon::Scalar, decltype(dtable)> interpreter{dtable, ds, tree};

                std::vector<Operon::Scalar> out(range.Size());
                Util::EvaluateTree(tree, ds, range, parameters.data(), out.data());
                auto res1 = interpreter.Evaluate(parameters, range);

                Eigen::Array<Operon::Scalar, -1, -1> jfwd = interpreter.JacFwd(parameters, range);
                Eigen::Array<Operon::Scalar, -1, -1> jrev = interpreter.JacRev(parameters, range);

                auto f1 = std::isfinite(jjet.sum());
                auto f2 = std::isfinite(jrev.sum());
                auto ok = (!f1) || (f1 && f2 && jrev.isApprox(jjet, epsilon));

                if(f1 && f2) {
                    for (auto const& n : tree.Nodes()) {
                        auto [it, ok] = counts.insert({n.Type, {0, 0}});
                        std::get<0>(it->second) += 1;
                        std::get<1>(it->second) += std::abs(((jjet - jrev).abs() / jjet.abs()).mean());
                    }
                }

                if(!ok) {
                    constexpr auto precision{20};
                    fmt::print(fmt::fg(fmt::color::orange), "infix: {}\n", Operon::InfixFormatter::Format(tree, ds, precision));
                    fmt::print("res0: {}\n", out);
                    fmt::print("res1: {}\n", res1);
                    // fmt::print("eps: {}, {}\n", epsilon, jrev.isApprox(jjet, epsilon));
                    std::cout << std::setprecision(precision) << "x = " << ds.GetValues("x")[0] << ", y = " << ds.GetValues("y")[0] << "\n";
                    std::cout << std::setprecision(precision) << "J_jet    : " << jjet << "\n";
                    std::cout << std::setprecision(precision) << "J_forward: " << jfwd << "\n";
                    std::cout << std::setprecision(precision) << "J_reverse: " << jrev << "\n";

                }
                count += ok;
                CHECK(ok);
            }
            fmt::print("total: {}, passed: {}, failed: {}\n", n, count, n-count);
            for (auto [k, v] : counts) {
                Node n{k};
                if (n.IsLeaf()) { continue; }
                fmt::print("eps = {}, {}: {}%\n", epsilon, n.Name(), 100.F * std::get<1>(v) / std::get<0>(v));
            }
        };

        for (auto eps : {1e-1}) {
            testPrecision(eps);
        }
    }

    SUBCASE("autodiff relative accuracy") {

        auto constexpr N{10'000};

        fmt::print("function,x,y_true,y_pred,j_true,j_pred\n");
        auto testUnary = [&](Operon::NodeType type) {
            Operon::PrimitiveSet pset(NodeType::Constant | type);
            auto trees = generateTrees(pset, N, 2);

            auto name = Node(type).Name();
            if (name == "/") { name = "inv"; }

            for (auto const& tree : trees) {
                auto parameters = tree.GetCoefficients();
                Operon::Interpreter<Operon::Scalar, decltype(dtable)> interpreter{dtable, ds, tree};
                auto est = Operon::Interpreter<float, Operon::DispatchTable<float>>::Evaluate(tree, ds, range);
                auto [res, jet] = Util::Autodiff(tree, ds, range);
                Eigen::Array<Operon::Scalar, -1, -1> jac = interpreter.JacRev(parameters, range);
                auto const x = tree.Nodes().front().Value;
                fmt::print("{},{:15.f},{:.15f},{:.15f},{:.15f},{:.15f}\n", name, x, res[0], est[0], jet[0], jac(0));
            }
        };

        auto testBinary = [&](Operon::NodeType type) {
            Operon::PrimitiveSet pset(NodeType::Constant | type);
            auto trees = generateTrees(pset, N, 3);

            auto name = Node(type).Name();

            for (auto const& tree : trees) {
                auto parameters = tree.GetCoefficients();
                Operon::Interpreter<Operon::Scalar, decltype(dtable)> interpreter{dtable, ds, tree};
                auto est = Operon::Interpreter<float, Operon::DispatchTable<float>>::Evaluate(tree, ds, range);
                auto [res, jet] = Util::Autodiff(tree, ds, range);
                Eigen::Array<Operon::Scalar, -1, -1> jac = interpreter.JacRev(parameters, range);
                auto const x = tree.Nodes().front().Value;
                fmt::print("{},{:15.f},{:.15f},{:.15f},{:.15f},{:.15f}\n", name, x, res[0], est[0], jet[0], jac(0));
                fmt::print("{},{:15.f},{:.15f},{:.15f},{:.15f},{:.15f}\n", name, x, res[0], est[0], jet[1], jac(1));
            }
        };

        for (auto type : {NodeType::Div, NodeType::Exp, NodeType::Log, NodeType::Sin, NodeType::Cos, NodeType::Sqrt, NodeType::Tanh }) {
            testUnary(type);
        }

        for (auto type : {NodeType::Div, NodeType::Aq, NodeType::Pow }) {
            testBinary(type);
        }
    };
}

} // namespace Operon::Test
