// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <doctest/doctest.h>

#include "core/dataset.hpp"
#include "core/format.hpp"
#include "core/stats.hpp"
#include "core/metrics.hpp"
#include "core/distance.hpp"
#include "core/pset.hpp"
#include "analyzers/diversity.hpp"
#include "operators/creator.hpp"

#include "nanobench.h"

namespace Operon {
namespace Test {

template<typename Callable>
struct ComputeDistanceMatrix {
    explicit ComputeDistanceMatrix(Callable&& f)
        : f_(f) { }

    template<typename T>
    inline double operator()(std::vector<Operon::Vector<T>> const& hashes) const noexcept
    {
        double d = 0;
        for (size_t i = 0; i < hashes.size() - 1; ++i) {
            for (size_t j = i+1; j < hashes.size(); ++j) {
                d += static_cast<double>(Operon::Distance::CountIntersect(hashes[i], hashes[j]));
            }
        }
        return d;
    }

    Callable f_;
};

TEST_CASE("Intersection performance")
{
    size_t n = 1000;
    size_t maxLength = 100;
    size_t maxDepth = 1000;

    Operon::RandomGenerator rd(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Exp | NodeType::Log);

    std::vector<Tree> trees(n);
    auto btc = BalancedTreeCreator { grammar, inputs };
    std::generate(trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), 0, maxDepth); });

    std::vector<Operon::Vector<Operon::Hash>> hashesStrict(trees.size());
    std::vector<Operon::Vector<Operon::Hash>> hashesStruct(trees.size());

    std::vector<Operon::Vector<uint32_t>> hashesStrict32(trees.size());
    std::vector<Operon::Vector<uint32_t>> hashesStruct32(trees.size());

    const auto hashFunc = [](auto& tree, Operon::HashMode mode) { return MakeHashes<Operon::HashFunction::XXHash>(tree, mode); };
    std::transform(trees.begin(), trees.end(), hashesStrict.begin(), [&](Tree tree) { return hashFunc(tree, Operon::HashMode::Strict); });
    std::transform(trees.begin(), trees.end(), hashesStruct.begin(), [&](Tree tree) { return hashFunc(tree, Operon::HashMode::Relaxed); });

    auto convertVec = [](Operon::Vector<Operon::Hash> const& vec) {
        Operon::Vector<uint32_t> vec32(vec.size());
        std::transform(vec.begin(), vec.end(), vec32.begin(), [](auto h) { return static_cast<uint32_t>(h); });
        return vec32;
    };

    std::transform(hashesStrict.begin(), hashesStrict.end(), hashesStrict32.begin(), convertVec);
    std::transform(hashesStruct.begin(), hashesStruct.end(), hashesStruct32.begin(), convertVec);

    std::uniform_int_distribution<size_t> dist(0u, trees.size()-1);

    auto avgLen = std::transform_reduce(trees.begin(), trees.end(), 0.0, std::plus<>{}, [](auto const& t) { return t.Length(); }) / (double)n;
    auto totalOps = trees.size() * (trees.size() - 1) / 2; 

    SUBCASE("Performance") {
        ankerl::nanobench::Bench b;
        b.performanceCounters(true).relative(true);

        auto s = (double)totalOps * avgLen;

        double d = 0;
        b.batch(s).run("intersect str[i]ct", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::CountIntersect(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStrict);
        });

        b.batch(s).run("intersect str[u]ct", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::CountIntersect(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStruct);
        });

        b.batch(s).run("jaccard str[i]ct", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::Jaccard(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStrict);
        });

        b.batch(s).run("jaccard str[u]ct", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::Jaccard(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStruct);
        });

        b.batch(s).run("intersect str[i]ct 32", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::CountIntersect(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStrict32);
        });

        b.batch(s).run("intersect str[u]ct 32", [&](){
            auto f = [](auto const& lhs, auto const& rhs) { return Operon::Distance::CountIntersect(lhs, rhs); };
            ComputeDistanceMatrix<decltype(f)> cdm(std::move(f));
            d = cdm(hashesStruct32);
        });

    }
}
} // namespace Test
} // namespace Operon

