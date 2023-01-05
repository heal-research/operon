// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <doctest/doctest.h>
#include <Eigen/Core>

#include "operon/analyzers/diversity.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {

TEST_CASE("diversity")
{
    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Full);
    Operon::RandomGenerator rd(std::random_device {}());

    Eigen::Matrix<Operon::Scalar, -1, -1> values(1, 2);
    values << 1, 1; // don't care

    Dataset ds(values);
    BalancedTreeCreator btc(grammar, ds.Variables());

    constexpr size_t minLength{1};
    constexpr size_t maxLength{100};
    constexpr size_t maxDepth{1000};

    UniformTreeInitializer treeInit(btc);
    treeInit.ParameterizeDistribution(minLength, maxLength);
    treeInit.SetMaxDepth(maxDepth);
    UniformCoefficientInitializer coeffInit;
    coeffInit.ParameterizeDistribution(Operon::Scalar{-1}, Operon::Scalar{+1});

    constexpr size_t nTrees = 1000;
    std::vector<Tree> trees;
    trees.reserve(nTrees);

    for(auto i = 0UL; i < nTrees; ++i) {
        auto tree = treeInit(rd);
        coeffInit(rd, tree);
        trees.push_back(std::move(tree));
    }

    PopulationDiversityAnalyzer<Tree> diversityAnalyzer;
    diversityAnalyzer.Prepare(trees);
}

} // namespace Operon::Test
