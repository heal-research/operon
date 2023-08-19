// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <doctest/doctest.h>

#include "operon/hash/hash.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/parser/infix.hpp"
#include "../thirdparty/nanobench.h"

namespace Operon::Test {
TEST_SUITE("[implementation]")
{
    TEST_CASE("Parser correctness")
    {
        constexpr int nTrees = 1'000'000;
        constexpr int nNodes = 20;

        constexpr int nrow = 1;
        constexpr int ncol = 10;

        Operon::RandomGenerator rng(1234);

        Eigen::Matrix<Operon::Scalar, -1, -1> values(nrow, ncol);
        for (auto& v : values.reshaped()) { v = Operon::Random::Uniform(rng, -1.f, +1.f); }
        Operon::Dataset ds(values);

        Operon::PrimitiveSet pset;
        pset.SetConfig((PrimitiveSet::Arithmetic | NodeType::Aq | NodeType::Exp | NodeType::Log) & ~NodeType::Variable);
        Operon::BalancedTreeCreator btc(pset, ds.VariableHashes());

        // generate trees
        Operon::Vector<Operon::Tree> trees;
        trees.reserve(nTrees);
        for (int i = 0; i < nTrees; ++i) {
            trees.push_back(btc(rng, nNodes, 1, 10));
        }

        // prepare for parsing
        Operon::Vector<Operon::Tree> parsedTrees;
        parsedTrees.reserve(nTrees);

        // map variables
        Operon::Map<std::string, Operon::Hash> vars;
        for (auto const& v : ds.GetVariables()) {
            vars.insert({ v.Name, v.Hash });
        }

        std::transform(trees.begin(), trees.end(), std::back_inserter(parsedTrees), [&](const auto& tree) {
            try {
                auto parsed = InfixParser::Parse(InfixFormatter::Format(tree, ds, 50), vars);
                return parsed;
            } catch(std::exception& e) {
                fmt::print("unable to parse tree: {}\n{}\n", e.what(), Operon::InfixFormatter::Format(tree, ds));
                throw std::move(e);
            }
        });

        auto validateString = [](auto const& s) {
            size_t lp{0};
            size_t rp{0};
            for (auto c : s) {
                lp += c == '(';
                rp += c == ')';
            }
            return lp == rp;
        };

        // check the output of the parsed trees against the output of the original trees
        Range range{0, 1};
        size_t count{0};

        using DTable = DispatchTable<Operon::Scalar>;
        DTable dtable;

        for (int i = 0; i < nTrees; ++i) {
            bool isOk = true;
            auto const& t1 = trees[i];
            auto const& t2 = parsedTrees[i];
            auto v1 = Interpreter<Operon::Scalar, DTable>::Evaluate(t1, ds, range)[0];
            auto v2 = Interpreter<Operon::Scalar, DTable>::Evaluate(t2, ds, range)[0];

            isOk &= (!std::isfinite(v1) || !std::isfinite(v2) || std::abs(v1-v2) < 1e-6);

            auto s1 = Operon::InfixFormatter::Format(t1, ds, 5);
            auto s2 = Operon::InfixFormatter::Format(t2, ds, 5);

            if (!validateString(s1)) {
                fmt::print("warning: corrupted format string s1: {}\n", s1);
            }

            if (!validateString(s2)) {
                fmt::print("warning: corrupted format string s2: {}\n", s2);
            }

            if (!isOk) {
                ++count;
                fmt::print("warning: difference of {} (v1={}, v2={}) in the evaluations of trees:\n", std::abs(v1-v2), v1, v2);
                fmt::print("T1: {}\n", s1);
                fmt::print("T2: {}\n", s2);

                for (auto const& n : t1.Nodes()) { fmt::print("{} ", n.IsConstant() ? fmt::format("{}", n.Value) : n.Name()); } fmt::print("--> {}\n", s1);
                for (auto const& n : t2.Nodes()) { fmt::print("{} ", n.IsConstant() ? fmt::format("{}", n.Value) : n.Name()); } fmt::print("--> {}\n", s2);
            };
        }
        CHECK(count == 0);
    }

    TEST_CASE("Parser Expr 1")
    {
        auto str = "sin((sqrt(abs(square(sin(((-0.00191) * X6))))) - sqrt(abs(((-0.96224) / (-0.40567))))))";
        Operon::Map<std::string, Operon::Hash> vars;
        Operon::Map<Operon::Hash, std::string> names;
        Hasher hasher;
        for (int i = 0; i < 10; ++i) {
            auto name = fmt::format("X{}", i);
            auto hash = hasher(reinterpret_cast<uint8_t const*>(name.data()), name.size() * sizeof(char) / sizeof(uint8_t));
            vars[name] = hash;
            names[hash] = name;
        }

        auto tree = Operon::InfixParser::Parse(str, vars);
        fmt::print("{}\n", str);
        fmt::print("{}\n", Operon::InfixFormatter::Format(tree, names, 5));
    }

    TEST_CASE("Parser Expr 2")
    {
        Node c1(NodeType::Constant); c1.Value = 2;
        Node c2(NodeType::Constant); c2.Value = 3;
        Node c3(NodeType::Constant); c3.Value = 5;
        Node sub(NodeType::Sub);
        Node mul(NodeType::Mul);
        std::vector<Node> nodes { c1, c2, c3, sub, mul }; // 5 - 3 * 2
        Tree t(nodes);
        t.UpdateNodes();

        Dataset ds("./data/Poly-10.csv", true);
        auto s1 = InfixFormatter::Format(t, ds, 5);
        fmt::print("s1: {}\n", s1);
        Operon::Map<std::string, Operon::Hash> vmap;
        auto t2 = InfixParser::Parse(s1, vmap);
        auto s2 = InfixFormatter::Format(t2, ds, 5);
        fmt::print("s2: {}\n", s1);

        for (size_t i = 0; i < nodes.size(); ++i) {
            auto const& n1 = t[i];
            auto const& n2 = t2[i];
            fmt::print("{}\t{}\n", n1.IsConstant() ? fmt::format("{}", n1.Value) : n1.Name(), n2.IsConstant() ? fmt::format("{}", n2.Value) : n2.Name());
        }
    }

    TEST_CASE("Parser Expr 3")
    {
        std::string const expr{"3 aq 5"};
        Operon::Map<std::string, Operon::Hash> vars;
        auto tree = InfixParser::Parse(expr, vars);
        Operon::Map<Operon::Hash, std::string> variableNames;
        fmt::print("tree: {}\n", InfixFormatter::Format(tree, variableNames, 2));
    }

    TEST_CASE("Parser Expr 4")
    {
        auto model_str = "(((((((((-0.24762082099914550781) * X60) - ((-0.24762082099914550781) * X51)) - ((0.29588320851325988770 * X5) - ((-0.04808991029858589172) * X0))) + ((-0.34331262111663818359) / ((-0.11882954835891723633) * X23))) / ((-1.08731400966644287109) - ((-0.24762082099914550781) * X68))) + ((((-0.51293206214904785156) / ((-0.11882954835891723633) * X60)) * ((-0.24762082099914550781) * X42)) - ((-0.83979696035385131836) * X23))) * ((((-0.32350099086761474609) * X1) - ((-0.24762082099914550781) * X51)) * (0.53106397390365600586 * X38))) * ((((0.92230170965194702148 * X72) * ((-1.08731400966644287109) - ((-0.34331262111663818359) * (1.06355786323547363281 * X1)))) * ((-1.08731400966644287109) - ((-0.24762082099914550781) * X42))) + (((-0.33695843815803527832) / ((-0.11888219416141510010) * X43)) / ((-1.08523952960968017578) - ((-0.24762082099914550781) * X51)))))";

        Hasher hasher;

        Operon::Map<std::string, Operon::Hash> vars_map;
        Operon::Map<Operon::Hash, std::string> vars_names;
        for (int i = 0; i < 78; ++i) {
            auto name = fmt::format("X{}", i);
            auto hash = hasher(reinterpret_cast<uint8_t const*>(name.data()), name.size() * sizeof(char) / sizeof(uint8_t));
            vars_map[name] = hash;
            vars_names[hash] = name;
        }

        auto tree = Operon::InfixParser::Parse(model_str, vars_map);
        fmt::print("{}\n", Operon::InfixFormatter::Format(tree, vars_names));
    }

    TEST_CASE("Parser Expr 5")
    {
        auto model_str = "1 + 2 + 3 + 4";

        Operon::Map<std::string, Operon::Hash> vars_map;
        Operon::Map<Operon::Hash, std::string> vars_names;

        auto tree = Operon::InfixParser::Parse(model_str, vars_map);
        fmt::print("{}\n", Operon::InfixFormatter::Format(tree, vars_names));
        fmt::print("{}\n", Operon::PostfixFormatter::Format(tree, vars_names));
    }

    TEST_CASE("Formatter")
    {
        SUBCASE("Analytical quotient")
        {
            Node c1(NodeType::Constant); c1.Value = 2;
            Node c2(NodeType::Constant); c2.Value = 3;
            Node aq(NodeType::Aq);
            fmt::print("aq: {}\n", aq.Arity);

            Node dv(NodeType::Div);
            Tree t1({c2, c1, aq});
            Tree t2({c2, c1, dv});

            Operon::Map<Operon::Hash, std::string> map;

            Dataset::Matrix m(1,1);
            Operon::Dataset ds(m);
            Range range(0, 1);
            using DTable = DispatchTable<Operon::Scalar>;
            DTable dtable;
            auto v1 = Interpreter<Operon::Scalar, DTable>::Evaluate(t1, ds, range)[0];
            auto v2 = Interpreter<Operon::Scalar, DTable>::Evaluate(t2, ds, range)[0];

            fmt::print("{} = {}\n", InfixFormatter::Format(t1, map, 3), v1);
            fmt::print("{} = {}\n", InfixFormatter::Format(t2, map, 3), v2);
        }
    }
}

TEST_SUITE("[performance]")
{
    TEST_CASE("Parser")
    {
        constexpr int nTrees = 20'000;
        constexpr int nNodes = 50;

        Operon::Dataset ds("./data/Poly-10.csv", true);
        Operon::PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Tan);
        Operon::RandomGenerator rng(1234);
        Operon::BalancedTreeCreator btc(pset, ds.VariableHashes());

        // generate trees
        Operon::Vector<Operon::Tree> trees;
        trees.reserve(nTrees);
        for (int i = 0; i < nTrees; ++i) {
            trees.push_back(btc(rng, nNodes, 1, 10));
        }

        // format trees to infix strings
        Operon::Vector<std::string> treeStrings;
        treeStrings.reserve(nTrees);
        std::transform(trees.begin(), trees.end(), std::back_inserter(treeStrings), [&](auto const& tree) { return InfixFormatter::Format(tree, ds, 30); });

        // map dataset variables for parsing
        Operon::Map<std::string, Operon::Hash> map;
        for (auto const& v : ds.GetVariables()) {
            map.insert({ v.Name, v.Hash });
        }
        // benchmark parsing performance
        ankerl::nanobench::Bench b;
        b.performanceCounters(true).batch(nTrees * nNodes);
        b.run("parser performance", [&]() { std::for_each(treeStrings.begin(), treeStrings.end(), [&](auto const& str) { return Operon::InfixParser::Parse(str, map); }); });
    }
}
} // namespace Operon::Test
