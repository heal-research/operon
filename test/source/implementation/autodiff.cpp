// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <iomanip>
#include <random>

#include "../operon_test.hpp"

#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/core/problem.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {

TEST_CASE("Autodiff specific expressions", "[autodiff]") // NOLINT(readability-function-cognitive-complexity)
{
    Operon::Dataset::Matrix values(1, 2);
    values << 1, 1; // NOLINT

    Operon::RandomGenerator rng(0UL); // NOLINT(misc-const-correctness) — captured by ref in lambda, passed to non-const distribution calls
    Operon::Dataset ds(values);
    ds.SetVariableNames({"x", "y"});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};

    auto derive = [&](std::string const& expr) -> void {
        auto tree = Operon::InfixParser::Parse(expr, ds, /*reduce=*/true);
        for (auto& n : tree.Nodes()) {
            n.Optimize = n.IsLeaf();
        }

        auto parameters = tree.GetCoefficients();
        auto [res, jac] = Util::Autodiff(tree, ds, range);

        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto rev = interpreter.JacRev(parameters, range);
        auto fwd = interpreter.JacFwd(parameters, range);

        Eigen::Map<Eigen::Array<Operon::Scalar, -1, -1>> const jjet(jac.data(), static_cast<Eigen::Index>(range.Size()), static_cast<Eigen::Index>(parameters.size()));

        auto f1 = std::isfinite(jjet.sum());
        auto f2 = std::isfinite(rev.sum());
        auto ok = !f1 || (f2 && rev.isApprox(jjet, 1e-4F));
        CHECK(ok);
    };

    SECTION("0.51 * x") { derive("0.51 * x"); }
    SECTION("0.51 * 0.74") { derive("0.51 * 0.74"); }
    SECTION("2.53 / 1.46") { derive("2.53 / 1.46"); }
    SECTION("3 ^ 2") { derive("3 ^ 2"); }
    SECTION("sin(2)") { derive("sin(2)"); }
    SECTION("sin(2) + cos(3)") { derive("sin(2) + cos(3)"); }
    SECTION("sin(2) * cos(3)") { derive("sin(2) * cos(3)"); }
    SECTION("cos(sin(3))") { derive("cos(sin(3))"); }
    SECTION("exp(sin(2))") { derive("exp(sin(2))"); }
    SECTION("exp(x) + exp(y)") { derive("exp(x) + exp(y)"); }
    SECTION("log(x) + x * y - sin(y)") { derive("log(x) + x * y - sin(y)"); }
    SECTION("1 / x") { derive("1 / x"); }
    SECTION("sqrt(x) + sqrt(y)") { derive("sqrt(x) + sqrt(y)"); }
    SECTION("exp(x)") { derive("exp(x)"); }
    SECTION("sin(exp(x))") { derive("sin(exp(x))"); }
    SECTION("tan(x)") { derive("tan(x)"); }
    SECTION("x ^ 2") { derive("x ^ 2"); }
    SECTION("x ^ 3") { derive("x ^ 3"); }
    SECTION("asin(x)") { derive("asin(x)"); }
    SECTION("acos(x)") { derive("acos(x)"); }
    SECTION("atan(x)") { derive("atan(x)"); }
}

TEST_CASE("Autodiff forward vs reverse consistency", "[autodiff]")
{
    Operon::Dataset::Matrix values(1, 2);
    values << 1, 1; // NOLINT

    Operon::RandomGenerator rng(0UL);
    Operon::Dataset ds(values);
    ds.SetVariableNames({"x", "y"});

    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};

    using Operon::NodeType;
    Operon::PrimitiveSet pset;

    auto generateTrees = [&](auto pset, auto n, auto l) -> auto {
        constexpr auto mindepth{1};
        constexpr auto maxdepth{1000};
        std::uniform_int_distribution<size_t> const length(1, l);
        std::bernoulli_distribution bernoulli(0.5); // NOLINT
        std::uniform_real_distribution<Operon::Scalar> dist(-10.F, +10.F);
        Operon::BalancedTreeCreator const btc(&pset, ds.VariableHashes(), /* bias= */ 0.0, l);

        std::vector<Operon::Tree> trees;
        trees.reserve(n);

        for (auto i = 0; i < n; ++i) {
            auto tree = btc(rng, l, mindepth, maxdepth);
            for (auto& node : tree.Nodes()) {
                node.Optimize = bernoulli(rng) || node.IsLeaf();
                if (node.IsLeaf()) {
                    node.Value = dist(rng);
                }
            }
            trees.push_back(tree);
        }
        return trees;
    };

    constexpr auto n{100'000};
    constexpr auto epsilon{1e-4F};

    // unary
    pset.SetConfig(NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Sqrt | NodeType::Tanh | NodeType::Constant);
    auto trees = generateTrees(pset, n / 2, 2);

    // binary
    pset.SetConfig(NodeType::Div | NodeType::Aq | NodeType::Pow | NodeType::Constant);
    auto tmp = generateTrees(pset, n / 2, 3);
    std::ranges::copy(tmp, std::back_inserter(trees));

    size_t finiteMismatch{0};  // f1 finite, f2 non-finite
    size_t finiteDiverge{0};   // both finite but |rev - fwd| > epsilon
    for (auto const& tree : trees) {
        auto parameters = tree.GetCoefficients();
        auto [res, jac] = Util::Autodiff(tree, ds, range);
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, -1>> const jjet(jac.data(), static_cast<Eigen::Index>(range.Size()), static_cast<Eigen::Index>(parameters.size()));

        Operon::Interpreter<Operon::Scalar, decltype(dtable)> const interpreter{&dtable, &ds, &tree};
        Eigen::Array<Operon::Scalar, -1, -1> const jrev = interpreter.JacRev(parameters, range);

        auto f1 = std::isfinite(jjet.sum());
        auto f2 = std::isfinite(jrev.sum());
        if (f1 && !f2) { ++finiteMismatch; }
        else if (f1 && f2 && !jrev.isApprox(jjet, epsilon)) { ++finiteDiverge; }
    }
    auto const total = trees.size();
    INFO("finiteness mismatches: " << finiteMismatch << " / " << total);
    INFO("finite but diverging:  " << finiteDiverge  << " / " << total);
    constexpr auto maxDivergeRate{0.02};
    CHECK(finiteMismatch == 0);
    CHECK(static_cast<double>(finiteDiverge) / static_cast<double>(total) < maxDivergeRate);
}

TEST_CASE("Autodiff poly-10 expression", "[autodiff]")
{
    std::string const expr = "((0.78 / ((-1.12) * X8)) / (((((-0.61) * X3) * 0.82) / (((-0.22) * X6) / 1.77)) / (((-0.16) - 0.50) - (((-0.46) * X4) - ((-0.03) * X9)))))";

    Operon::Dataset ds("./data/Poly-10.csv", /*hasHeader=*/true);
    auto tree = InfixParser::Parse(expr, ds);
    auto coeff = tree.GetCoefficients();
    Operon::Range range(0, 10); // NOLINT

    DispatchTable<Operon::Scalar> dt;
    auto jacrev = Operon::Interpreter<Operon::Scalar, DispatchTable<Operon::Scalar>>{&dt, &ds, &tree}.JacRev(coeff, range);
    auto jacfwd = Operon::Interpreter<Operon::Scalar, DispatchTable<Operon::Scalar>>{&dt, &ds, &tree}.JacFwd(coeff, range);

    // Forward and reverse should agree
    auto f1 = std::isfinite(jacrev.sum());
    auto f2 = std::isfinite(jacfwd.sum());
    auto ok = (!f1 && !f2) || (f1 && f2 && jacrev.isApprox(jacfwd, 0.01F));
    CHECK(ok);
}

} // namespace Operon::Test
