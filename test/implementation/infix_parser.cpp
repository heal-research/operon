#include <doctest/doctest.h>

#include "core/eval.hpp"
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

        // check the output of the parsed trees against the output of the original trees
        bool isOk = true;
        Range range{0, 1};
        for (int i = 0; i < nTrees; ++i) {
            auto const& t1 = trees[i];
            auto const& t2 = parsedTrees[i];
            auto v1 = Evaluate<double>(t1, ds, range)[0]; 
            auto v2 = Evaluate<double>(t2, ds, range)[0]; 
            isOk &= std::abs(v1-v2) < 1e-12;

            if (!isOk) break;
        }
        CHECK(isOk);
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
