// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <doctest/doctest.h>

#include "interpreter/interpreter.hpp"
#include "core/format.hpp"
#include "core/pset.hpp"
#include "nanobench.h"
#include "operators/creator/balanced.hpp"
#include "parser/infix.hpp"
#include "robin_hood.h"

namespace Operon::Test {
TEST_SUITE("[implementation]")
{
    TEST_CASE("Lexer")
    {
        SUBCASE("sin(PI)")
        {
            std::string str("-(1)");
            pratt::lexer<Operon::token, Operon::conv> lex(str);
            auto tokens = lex.tokenize();
            for (auto t : tokens) {
                std::cout << pratt::token_name[t.kind] << "\n";
            }
        }
    }

    TEST_CASE("Parser")
    {
        constexpr int nTrees = 100'000;
        constexpr int nNodes = 50;

        Operon::Dataset ds("../data/Poly-10.csv", true);
        Operon::PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Tan | NodeType::Square | NodeType::Sqrt | NodeType::Cbrt);
        Operon::RandomGenerator rng(1234);
        Operon::BalancedTreeCreator btc(pset, ds.Variables());

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
        robin_hood::unordered_flat_map<std::string, Operon::Hash> map;
        for (auto const& v : ds.Variables()) {
            map.insert({ v.Name, v.Hash });
        }

        std::transform(trees.begin(), trees.end(), std::back_inserter(parsedTrees), [&](const auto& tree) { return InfixParser::Parse(InfixFormatter::Format(tree, ds, 30), map); });

        fmt::print("{}\n", InfixFormatter::Format(trees.front(), ds, 2));

        DispatchTable ft;

        // check the output of the parsed trees against the output of the original trees
        bool isOk = true;
        Range range{0, 1};
        for (int i = 0; i < nTrees; ++i) {
            auto const& t1 = trees[i];
            auto const& t2 = parsedTrees[i];
            auto v1 = Interpreter::Evaluate<Operon::Scalar>(ft, t1, ds, range)[0];
            auto v2 = Interpreter::Evaluate<Operon::Scalar>(ft, t2, ds, range)[0];
            isOk &= std::abs(v1-v2) < 1e-12;

            if (!isOk) break;
        }
        CHECK(isOk);
    }

    TEST_CASE("Parser Expr")
    {
        auto model_str = "(((((((((-0.24762082099914550781) * X60) - ((-0.24762082099914550781) * X51)) - ((0.29588320851325988770 * X5) - ((-0.04808991029858589172) * X0))) + ((-0.34331262111663818359) / ((-0.11882954835891723633) * X23))) / ((-1.08731400966644287109) - ((-0.24762082099914550781) * X68))) + ((((-0.51293206214904785156) / ((-0.11882954835891723633) * X60)) * ((-0.24762082099914550781) * X42)) - ((-0.83979696035385131836) * X23))) * ((((-0.32350099086761474609) * X1) - ((-0.24762082099914550781) * X51)) * (0.53106397390365600586 * X38))) * ((((0.92230170965194702148 * X72) * ((-1.08731400966644287109) - ((-0.34331262111663818359) * (1.06355786323547363281 * X1)))) * ((-1.08731400966644287109) - ((-0.24762082099914550781) * X42))) + (((-0.33695843815803527832) / ((-0.11888219416141510010) * X43)) / ((-1.08523952960968017578) - ((-0.24762082099914550781) * X51)))))";

        Hasher<HashFunction::XXHash> hasher;

        robin_hood::unordered_flat_map<std::string, Operon::Hash> vars_map;
        std::unordered_map<Operon::Hash, std::string> vars_names;
        for (int i = 0; i < 78; ++i) {
            auto name = fmt::format("X{}", i);
            auto hash = hasher(reinterpret_cast<uint8_t const*>(name.data()), name.size() * sizeof(char) / sizeof(uint8_t));
            vars_map[name] = hash;
            vars_names[hash] = name;
        }

        auto tree = Operon::InfixParser::Parse(model_str, vars_map);
        fmt::print("{}\n", Operon::InfixFormatter::Format(tree, vars_names));
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

            std::unordered_map<Operon::Hash, std::string> map;

            Dataset::Matrix m(1,1);
            Operon::Dataset ds(m);
            Range r(0, 1);
            DispatchTable ft;
            auto v1 = Interpreter::Evaluate<Operon::Scalar>(ft, t1, ds, r)[0];
            auto v2 = Interpreter::Evaluate<Operon::Scalar>(ft, t2, ds, r)[0];

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

        Operon::Dataset ds("../data/Poly-10.csv", true);
        Operon::PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Tan);
        Operon::RandomGenerator rng(1234);
        Operon::BalancedTreeCreator btc(pset, ds.Variables());

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
        robin_hood::unordered_map<std::string, Operon::Hash> map;
        for (auto const& v : ds.Variables()) {
            map.insert({ v.Name, v.Hash });
        }

        // benchmark parsing performance
        ankerl::nanobench::Bench b;
        b.performanceCounters(true).batch(nTrees * nNodes);
        b.run("parser performance", [&]() { std::for_each(treeStrings.begin(), treeStrings.end(), [&](auto const& str) { return Operon::InfixParser::Parse(str, map); }); });
    }
}
}
