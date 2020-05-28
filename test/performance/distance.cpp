/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/stats.hpp"
#include "core/metrics.hpp"
#include "core/distance.hpp"
#include "core/grammar.hpp"
#include "analyzers/diversity.hpp"
#include "operators/creator.hpp"

#include <catch2/catch.hpp>

namespace Operon {
namespace Test {
TEST_CASE("Tree distance performance", "[performance]") 
{
    size_t n = 1000;
    size_t maxLength = 100;
    size_t maxDepth = 100;

    auto rd = Operon::Random(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic | NodeType::Exp | NodeType::Log);

    std::vector<Tree> trees(n);
    auto btc = BalancedTreeCreator { grammar, inputs };
    std::generate(std::execution::unseq, trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), maxDepth); });
    Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;

    size_t reps = 100;

    SECTION("Strict diversity") {
        std::vector<Operon::Distance::HashVector> hashes(trees.size());
        std::transform(trees.begin(), trees.end(), hashes.begin(), [](Tree tree) { return MakeHashes(tree, Operon::HashMode::Strict); });
        
        // measure speed of vectorized intersection
        auto totalOps = n * (n-1) / 2;
        double tMean, tStddev, opsPerSecond;
        double diversity;

        // measured speed of vectorized intersection
        MeanVarianceCalculator elapsedCalc;
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    size_t c = Operon::Distance::CountIntersectSIMD(hashes[i], hashes[j]);
                    double s = hashes[i].size() + hashes[j].size() - c;
                    calc.Add((s - c) / s);
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6;
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = totalOps / tMean; // from ms to second
        fmt::print("strict diversity (vector): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);

        // measured speed of vectorized intersection - multi-threaded
        elapsedCalc.Reset();
        std::vector<std::pair<size_t, size_t>> pairs(n);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<double> distances(n);
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            size_t idx = 0;
            size_t total = totalOps;
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    pairs[idx++] = { i, j };

                    if (idx == std::min(n, total)) {
                        idx = 0;
                        total -= n;

                        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](auto m) { 
                                auto&& [a, b] = pairs[m];
                                size_t c = Operon::Distance::CountIntersectSIMD(hashes[a], hashes[b]);
                                double s = hashes[a].size() + hashes[b].size() - c;
                                distances[m] = (s - c) / s;
                        }); 
                        for (auto d : distances) {
                            calc.Add(d);
                        }
                    }
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6;
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = totalOps / tMean; // from ms to second
        fmt::print("strict diversity (vector): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);

        // measured speed of scalar intersection
        elapsedCalc.Reset();
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    size_t c = Operon::Distance::CountIntersect(hashes[i], hashes[j]);
                    double s = hashes[i].size() + hashes[j].size() - c;
                    calc.Add((s - c) / s);
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6;
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = totalOps / tMean;
        fmt::print("strict diversity (scalar): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);
    }

    SECTION("Relaxed diversity") {
        std::vector<Operon::Distance::HashVector> hashes(trees.size());
        std::transform(trees.begin(), trees.end(), hashes.begin(), [](Tree tree) { return MakeHashes(tree, Operon::HashMode::Relaxed); });
        
        // measure speed of vectorized intersection
        auto totalOps = n * (n-1) / 2;
        double tMean, tStddev, opsPerSecond;
        double diversity;

        // measured speed of vectorized intersection
        MeanVarianceCalculator elapsedCalc;
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    size_t c = Operon::Distance::CountIntersectSIMD(hashes[i], hashes[j]);
                    double s = hashes[i].size() + hashes[j].size() - c;
                    calc.Add((s - c) / s);
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6;
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = totalOps / tMean; // from ms to second
        fmt::print("relaxed diversity (vector): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);

        // measured speed of scalar intersection
        elapsedCalc.Reset();
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    size_t c = Operon::Distance::CountIntersect(hashes[i], hashes[j]);
                    double s = hashes[i].size() + hashes[j].size() - c;
                    calc.Add((s - c) / s);
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6;
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = totalOps / tMean;
        fmt::print("relaxed diversity (scalar): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);
    }
}
} // namespace Test
} // namespace Operon

