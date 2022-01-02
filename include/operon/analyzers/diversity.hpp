// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef DIVERSITY_HPP
#define DIVERSITY_HPP

#include <unordered_set>
#include <mutex>

#include "pdqsort.h"
#include "vstat.hpp"

#include "core/operator.hpp"
#include "core/distance.hpp"

namespace Operon {
namespace {
    template<Operon::HashFunction F>
    static inline Operon::Vector<Operon::Hash> MakeHashes(Tree& tree, Operon::HashMode mode) {
        Operon::Vector<Operon::Hash> hashes(tree.Length());
        tree.Hash<F>(mode);
        std::transform(tree.Nodes().begin(), tree.Nodes().end(), hashes.begin(), [](const auto& node) { return node.CalculatedHashValue; });
        pdqsort(hashes.begin(), hashes.end());
        return hashes;
    }
}

template <typename T, Operon::HashFunction F = Operon::HashFunction::XXHash>
class PopulationDiversityAnalyzer final : PopulationAnalyzerBase<T> {
public:
    auto operator()(Operon::RandomGenerator&) const -> double
    {
        return diversity;
    }

    void Prepare(Operon::Span<const T> pop, Operon::HashMode mode = Operon::HashMode::Strict)
    {
        hashes.clear();
        hashes.resize(pop.size());

        std::vector<size_t> indices(pop.size());
        std::iota(indices.begin(), indices.end(), 0);

        // hybrid (strict) hashing
        std::for_each(indices.begin(), indices.end(), [&](size_t i) {
            hashes[i] = MakeHashes<F>(pop[i].Genotype, mode);
        });

        std::vector<std::pair<size_t, size_t>> pairs(hashes.size());
        std::vector<Operon::Scalar> distances(hashes.size());
        size_t k = 0;
        size_t c = 0;
        size_t n = (hashes.size()-1) * hashes.size() / 2;

        for (size_t i = 0; i < hashes.size() - 1; ++i) {
            for(size_t j = i+1; j < hashes.size(); ++j) {
                pairs[k++] = { i, j }; ++c;

                if (k == distances.size() || c == n) {
                    k = 0;
                    std::for_each(indices.begin(), indices.end(), [&](size_t idx) {
                        auto [a, b] = pairs[idx];
                        distances[idx] = Operon::Distance::Jaccard(hashes[a], hashes[b]);
                    });
                }
            }
        }
        diversity = univariate::accumulate<Operon::Scalar>(distances.data(), distances.size()).mean;
    }

    private:
        double diversity;
        std::vector<Operon::Vector<Operon::Hash>> hashes;
    };
} // namespace Operon

#endif
