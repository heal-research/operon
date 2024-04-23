// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <doctest/doctest.h>
#include <fmt/core.h>


#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/distance.hpp"
#include "operon/core/pset.hpp"
#include "operon/analyzers/diversity.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

#include "nanobench.h"

namespace Operon {
namespace Test {

template<typename Callable>
struct ComputeDistanceMatrix {
    explicit ComputeDistanceMatrix(Callable&& f)
        : f_(f) { }

    template<typename T>
    inline auto operator()(std::vector<Operon::Vector<T>> const& hashes) const noexcept -> double
    {
        double d = 0;
        for (size_t i = 0; i < hashes.size() - 1; ++i) {
            for (size_t j = i+1; j < hashes.size(); ++j) {
                d += static_cast<double>(Operon::Distance::Jaccard(hashes[i], hashes[j]));
            }
        }
        return 2 * d / (hashes.size() * hashes.size()-1);
    }

    Callable f_;
};

TEST_CASE("Intersection performance")
{
    size_t n = 1000;
    size_t maxLength = 50;
    size_t maxDepth = 1000;

    Operon::RandomGenerator rd(1234);
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);

    auto variables = ds.GetVariables();

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Exp | NodeType::Log);

    std::vector<Tree> trees(n);
    BalancedTreeCreator btc(grammar, ds.VariableHashes());
    UniformCoefficientInitializer coeffInit;
    std::generate(trees.begin(), trees.end(), [&]() { auto tree = btc(rd, sizeDistribution(rd), 0, maxDepth); coeffInit(rd, tree); return tree; });

    std::vector<Operon::Vector<Operon::Hash>> hashesStrict(trees.size());
    std::vector<Operon::Vector<Operon::Hash>> hashesStruct(trees.size());


    const auto hashFunc = [](auto& tree, Operon::HashMode mode) { return MakeHashes(tree, mode); };
    std::transform(trees.begin(), trees.end(), hashesStrict.begin(), [&](Tree tree) { return hashFunc(tree, Operon::HashMode::Strict); });
    std::transform(trees.begin(), trees.end(), hashesStruct.begin(), [&](Tree tree) { return hashFunc(tree, Operon::HashMode::Relaxed); });

    std::uniform_int_distribution<size_t> dist(0U, trees.size()-1);

    auto totalOps = trees.size() * (trees.size() - 1) / 2;

    SUBCASE("Hashing performance") {
        ankerl::nanobench::Bench b;
        b.performanceCounters(true).relative(true);

        b.batch(totalOps).run("xxhash", [&]() {
            std::transform(trees.begin(), trees.end(), hashesStrict.begin(), [&](Tree tree) { return hashFunc(tree, Operon::HashMode::Strict); });
        });
    }

    SUBCASE("Performance 64-bit") {
        ankerl::nanobench::Bench b;
        b.performanceCounters(true).relative(true);

        auto s = static_cast<double>(totalOps);

        double d = 0;
        b.batch(s).run("jaccard str[i]ct", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::Jaccard(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStrict);
        });
        fmt::print("d = {}\n", d);

        b.batch(s).run("jaccard str[u]ct", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::Jaccard(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStruct);
        });
        fmt::print("d = {}\n", d);

        b.batch(s).run("sorensen-dice str[i]ct", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::SorensenDice(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStrict);
        });
        fmt::print("d = {}\n", d);

        b.batch(s).run("sorensen-dice str[u]ct", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::SorensenDice(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStruct);
        });
        fmt::print("d = {}\n", d);
    }
}
} // namespace Test
} // namespace Operon
