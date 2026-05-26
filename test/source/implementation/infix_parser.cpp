// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "operon/hash/hash.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"

namespace Operon::Test {

TEST_CASE("Parser roundtrip correctness", "[parser]")
{
    constexpr int nTrees = 100'000;
    constexpr int nNodes = 20;
    constexpr int nrow = 1;
    constexpr int ncol = 10;

    Operon::RandomGenerator rng(1234);

    Eigen::Matrix<Operon::Scalar, -1, -1> values(nrow, ncol);
    for (auto& v : values.reshaped()) { v = Operon::Random::Uniform(rng, -1.F, +1.F); }
    Operon::Dataset ds(values);

    Operon::PrimitiveSet pset;
    pset.SetConfig(PrimitiveSet::Arithmetic | NodeType::Aq | NodeType::Exp | NodeType::Log | NodeType::Variable);
    Operon::BalancedTreeCreator const btc(&pset, ds.VariableHashes(), /* bias= */ 0.0, nNodes);

    Operon::Vector<Operon::Tree> trees;
    trees.reserve(nTrees);
    for (int i = 0; i < nTrees; ++i) {
        trees.push_back(btc(rng, nNodes, 1, 10));
    }

    Operon::Vector<Operon::Tree> parsedTrees;
    parsedTrees.reserve(nTrees);
    std::transform(trees.begin(), trees.end(), std::back_inserter(parsedTrees), [&](const auto& tree) -> auto {
        return InfixParser::Parse(InfixFormatter::Format(tree, ds, 50), ds);
    });

    Range range{0, 1};

    using DTable = DispatchTable<Operon::Scalar>;
    DTable const dtable;

    auto countFailures = [&](float eps) -> size_t {
        size_t count{0};
        for (int i = 0; i < nTrees; ++i) {
            auto const& t1 = trees[i];
            auto const& t2 = parsedTrees[i];
            auto v1 = Interpreter<Operon::Scalar, DTable>::Evaluate(t1, ds, range)[0];
            auto v2 = Interpreter<Operon::Scalar, DTable>::Evaluate(t2, ds, range)[0];
            if (std::isfinite(v1)) {
                count += static_cast<size_t>(!std::isfinite(v2) || std::abs(v1 - v2) > eps);
            }
        }
        return count;
    };

    constexpr auto epsLoose{1e-6F};
    constexpr auto epsStrict{1e-5F};
    constexpr auto maxFailureRateLoose{1e-2};
    constexpr auto maxFailureRateStrict{1e-3};

    CHECK(static_cast<double>(countFailures(epsLoose))  / nTrees < maxFailureRateLoose);
    CHECK(static_cast<double>(countFailures(epsStrict)) / nTrees < maxFailureRateStrict);
}

TEST_CASE("Parse specific expressions", "[parser]")
{
    SECTION("Nested unary functions") {
        const auto* str = "sin((sqrt(abs(square(sin(((-0.00191) * X6))))) - sqrt(abs(((-0.96224) / (-0.40567))))))";
        auto tree = Operon::InfixParser::Parse(str);
        CHECK(tree.Length() > 0);
    }

    SECTION("Arithmetic with constants") {
        Node c1(NodeType::Constant); c1.Value = 2;
        Node c2(NodeType::Constant); c2.Value = 3;
        Node c3(NodeType::Constant); c3.Value = 5;
        Node const sub(NodeType::Sub);
        Node const mul(NodeType::Mul);
        Operon::Vector<Node> const nodes{c1, c2, c3, sub, mul}; // 5 - 3 * 2
        Tree t(nodes);
        t.UpdateNodes();

        Dataset const ds("./data/Poly-10.csv", true);
        auto s1 = InfixFormatter::Format(t, ds, 5);
        auto t2 = InfixParser::Parse(s1);

        // Roundtrip: same number of nodes
        CHECK(t.Length() == t2.Length());
    }

    SECTION("Analytical quotient") {
        std::string const expr{"aq(3, 5)"};
        auto tree = InfixParser::Parse(expr);
        CHECK(tree.Length() > 0);
    }

    SECTION("Multiple additions") {
        const auto* modelStr = "1 + 2 + 3 + 4";
        auto tree = Operon::InfixParser::Parse(modelStr);

        using DTable = DispatchTable<Operon::Scalar>;
        DTable const dtable;
        std::string const x{"x"};
        std::vector<Operon::Scalar> const v{0};
        Operon::Dataset const ds({x}, {v});
        auto result = Interpreter<Operon::Scalar, DTable>::Evaluate(tree, ds, Range(0, 1));
        CHECK(result[0] == Catch::Approx(10.0F));
    }
}

TEST_CASE("Formatter output", "[parser]")
{
    SECTION("Balanced parentheses") {
        Operon::RandomGenerator rng(1234);
        Operon::Dataset const ds("./data/Poly-10.csv", true);
        Operon::PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic | NodeType::Exp | NodeType::Log);
        constexpr size_t maxLength = 20;
        Operon::BalancedTreeCreator const btc(&pset, ds.VariableHashes(), /* bias= */ 0.0, maxLength);

        auto validateString = [](auto const& s) -> auto {
            size_t lp{0};
            size_t rp{0};
            for (auto c : s) {
                lp += c == '(';
                rp += c == ')';
            }
            return lp == rp;
        };

        for (int i = 0; i < 100; ++i) {
            auto tree = btc(rng, 20, 1, 10);
            auto s = InfixFormatter::Format(tree, ds, 5);
            CHECK(validateString(s));
        }
    }
}

} // namespace Operon::Test
