// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef DIVERSITY_HPP
#define DIVERSITY_HPP

#include "core/operator.hpp"
#include "core/stats.hpp"
#include "core/distance.hpp"
#include <execution>
#include <unordered_set>
#include <mutex>

namespace Operon {
namespace {
    template<Operon::HashFunction F>
    static inline Operon::Distance::HashVector MakeHashes(Tree& tree, Operon::HashMode mode) {
        Operon::Distance::HashVector hashes(tree.Length());
        tree.Hash<F>(mode);
        std::transform(std::execution::seq, tree.Nodes().begin(), tree.Nodes().end(), hashes.begin(), [](const auto& node) { return node.CalculatedHashValue; });
        std::sort(std::execution::seq, hashes.begin(), hashes.end());
        return hashes;
    }
}

template <typename T, Operon::HashFunction F = Operon::HashFunction::XXHash, typename ExecutionPolicy = std::execution::parallel_unsequenced_policy>
class PopulationDiversityAnalyzer final : PopulationAnalyzerBase<T> {
public:
    double operator()(Operon::RandomGenerator&) const
    {
        return diversity;
    }

    void Prepare(Operon::Span<const T> pop, Operon::HashMode mode = Operon::HashMode::Strict)
    {
        hashes.clear();
        hashes.resize(pop.size());

        std::vector<size_t> indices(pop.size());
        std::iota(indices.begin(), indices.end(), 0);

        ExecutionPolicy ep;

        // hybrid (strict) hashing
        std::for_each(ep, indices.begin(), indices.end(), [&](size_t i) {
            hashes[i] = MakeHashes<F>(pop[i].Genotype, mode);
        });

        std::vector<std::pair<size_t, size_t>> pairs(hashes.size());
        std::vector<Operon::Scalar> distances(hashes.size());
        size_t k = 0;
        size_t c = 0;
        size_t n = (hashes.size()-1) * hashes.size() / 2;

        MeanVarianceCalculator calc;
        for (size_t i = 0; i < hashes.size() - 1; ++i) {
            for(size_t j = i+1; j < hashes.size(); ++j) {
                pairs[k++] = { i, j }; ++c;

                if (k == distances.size() || c == n) {
                    k = 0;
                    std::for_each(ep, indices.begin(), indices.end(), [&](size_t idx) {
                        auto [a, b] = pairs[idx];
                        distances[idx] = Operon::Distance::Jaccard(hashes[a], hashes[b]);
                    });
                    calc.Add(Operon::Span(distances.data(), distances.size()));
                }
            }
        }
        diversity = calc.Mean();
    }

    private:
        double diversity;
        std::vector<Operon::Distance::HashVector> hashes; 
    };
} // namespace Operon

#endif
