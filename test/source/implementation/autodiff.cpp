// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <random>

#include "../operon_test.hpp"

#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {

TEST_CASE("Autodiff specific expressions", "[autodiff]") // NOLINT(readability-function-cognitive-complexity)
{
    Operon::Dataset ds(std::vector<std::string>{"x", "y"},
                       std::vector<std::vector<Operon::Scalar>>{{1}, {1}});
    Operon::RandomGenerator rng(0UL); // NOLINT(misc-const-correctness) — captured by ref in lambda, passed to non-const distribution calls
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
    Operon::Dataset ds(std::vector<std::string>{"x", "y"},
                       std::vector<std::vector<Operon::Scalar>>{{1}, {1}});
    Operon::RandomGenerator rng(0UL);

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

TEST_CASE("Autodiff variable-wise derivative", "[autodiff]")
{
    Operon::Dataset ds(std::vector<std::string>{"x", "y"},
                       std::vector<std::vector<Operon::Scalar>>{{1.3F, -0.6F, 2.0F}, {0.4F, 1.1F, -0.9F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};
    auto const xHash = ds.GetVariable("x")->Hash;
    auto const yHash = ds.GetVariable("y")->Hash;

    // JacRevVariable/JacFwdVariable against hand-derived analytic derivatives.
    auto checkAnalytic = [&](std::string const& expr, Operon::Hash variable, auto analytic) -> void {
        auto tree = Operon::InfixParser::Parse(expr, ds, /*reduce=*/false);
        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};

        auto rev = interpreter.JacRevVariable(coeff, range, variable);
        auto fwd = interpreter.JacFwdVariable(coeff, range, variable);

        auto const xs = ds.GetValues("x");
        auto const ys = ds.GetValues("y");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto const expected = analytic(xs[i], ys[i]);
            CHECK(rev[i] == Catch::Approx(expected).margin(1e-4));
            CHECK(fwd[i] == Catch::Approx(expected).margin(1e-4));
        }
    };

    SECTION("d/dx(x^2) = 2x") {
        checkAnalytic("x ^ 2", xHash, [](auto x, auto /*y*/) { return 2 * x; });
    }
    SECTION("d/dx(sin(x)) = cos(x)") {
        checkAnalytic("sin(x)", xHash, [](auto x, auto /*y*/) { return std::cos(x); });
    }
    SECTION("d/dx(x*y) w.r.t. x = y") {
        checkAnalytic("x * y", xHash, [](auto /*x*/, auto y) { return y; });
    }
    SECTION("d/dx(x + x*x) = 1 + 2x (multiple occurrences summed)") {
        checkAnalytic("x + x * x", xHash, [](auto x, auto /*y*/) { return 1 + 2 * x; });
    }
    SECTION("d/dy(x*y) w.r.t. y = x") {
        checkAnalytic("x * y", yHash, [](auto x, auto /*y*/) { return x; });
    }

    // InfixParser produces one Variable node per textual occurrence — the
    // "multiple occurrences" case above never exercises the IsRef branch in
    // ReverseTraceGeneric/ForwardTraceGeneric (a shared-subtree occurrence
    // accumulating into the referenced node's trace/dot column instead of
    // its own). Build x * Ref(x) by hand to close that gap: node 0 is the
    // only Variable node (and thus the only one BuildColumns assigns a
    // column to); node 1 is a Ref to node 0, so its contribution must flow
    // back through the Ref-accumulation path for the result to match d/dx(x^2).
    SECTION("d/dx(x*Ref(x)) = 2x (shared subtree via Ref node)") {
        Operon::Vector<Operon::Node> nodes;
        Operon::Node v(Operon::NodeType::Variable);
        v.HashValue = v.CalculatedHashValue = xHash;
        v.Value = 1.0F;
        Operon::Node ref = Operon::Node::Ref(0);
        Operon::Node mul(Operon::NodeType::Mul);
        mul.Length = 2;
        nodes.push_back(v);
        nodes.push_back(ref);
        nodes.push_back(mul);
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto rev = interpreter.JacRevVariable(coeff, range, xHash);
        auto fwd = interpreter.JacFwdVariable(coeff, range, xHash);

        auto const xs = ds.GetValues("x");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto const expected = 2 * xs[i];
            CHECK(rev[i] == Catch::Approx(expected).margin(1e-4));
            CHECK(fwd[i] == Catch::Approx(expected).margin(1e-4));
        }
    }

    // Cross-check against central finite differences on a larger expression,
    // as a sanity net beyond the hand-derived cases above.
    SECTION("central finite-difference cross-check") {
        // Separate all-positive-x dataset — the check expression below uses
        // log(x), which is NaN for the negative x row in the outer `ds`.
        Operon::Dataset dsPos(std::vector<std::string>{"x", "y"},
                              std::vector<std::vector<Operon::Scalar>>{{1.3F, 0.6F, 2.0F}, {0.4F, 1.1F, -0.9F}});
        Operon::Range const rangePos{0, dsPos.Rows<std::size_t>()};

        auto tree = Operon::InfixParser::Parse("log(x) + x * y - sin(y) + exp(x * 0.3)", dsPos, /*reduce=*/false);
        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &dsPos, &tree};
        auto rev = interpreter.JacRevVariable(coeff, rangePos, xHash);

        constexpr Operon::Scalar h = 1e-3F;
        auto const xs = dsPos.GetValues("x");
        std::vector<Operon::Scalar> xPlus(xs.begin(), xs.end());
        std::vector<Operon::Scalar> xMinus(xs.begin(), xs.end());
        for (auto& v : xPlus) { v += h; }
        for (auto& v : xMinus) { v -= h; }
        auto const ys = dsPos.GetValues("y");
        Operon::Dataset dsPlus(std::vector<std::string>{"x", "y"}, std::vector<std::vector<Operon::Scalar>>{xPlus, std::vector<Operon::Scalar>(ys.begin(), ys.end())});
        Operon::Dataset dsMinus(std::vector<std::string>{"x", "y"}, std::vector<std::vector<Operon::Scalar>>{xMinus, std::vector<Operon::Scalar>(ys.begin(), ys.end())});

        using Interp = Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>>;
        auto fPlus = Interp::Evaluate(tree, dsPlus, rangePos, coeff);
        auto fMinus = Interp::Evaluate(tree, dsMinus, rangePos, coeff);

        for (std::size_t i = 0; i < rangePos.Size(); ++i) {
            auto const fd = (fPlus[i] - fMinus[i]) / (2 * h);
            CHECK(rev[i] == Catch::Approx(fd).margin(1e-2));
        }
    }
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
