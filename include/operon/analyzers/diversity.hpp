// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef DIVERSITY_HPP
#define DIVERSITY_HPP

#include <algorithm>
#include <unordered_set>
#include <vstat/vstat.hpp>

#include "operon/analyzers/analyzer_base.hpp"
#include "operon/core/operator.hpp"
#include "operon/core/distance.hpp"
#include "operon/core/tree.hpp"

namespace Operon {
namespace {
    inline auto MakeHashes(Tree& tree, Operon::HashMode m) -> Operon::Vector<Operon::Hash> {
        Operon::Vector<Operon::Hash> hashes(tree.Length());
        [[maybe_unused]] auto h = tree.Hash(m);
        std::transform(tree.Nodes().begin(), tree.Nodes().end(), hashes.begin(), [](const auto& node) { return node.CalculatedHashValue; });
        std::stable_sort(hashes.begin(), hashes.end());
        return hashes;
    }
} // namespace

template <typename T, Operon::HashMode M = Operon::HashMode::Strict>
class PopulationDiversityAnalyzer final : PopulationAnalyzerBase<T> {
public:
    auto operator()(Operon::RandomGenerator& /*unused*/) const -> double
    {
        return diversity_;
    }

    void Prepare(Operon::Span<T> pop)
    {
        std::vector<size_t> indices(pop.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<Operon::Vector<Operon::Hash>> hashes;
        hashes.reserve(pop.size());

        std::transform(indices.begin(), indices.end(), std::back_inserter(hashes),
                [&](auto i) { return MakeHashes(pop[i], M); });

        vstat::univariate_accumulator<double> acc;
        for (auto i = 0UL; i < pop.size() - 1; ++i) {
            for (auto j = i+1; j < pop.size(); ++j) {
                acc(Operon::Distance::Jaccard(hashes[i], hashes[j]));
            }
        }

        diversity_ = vstat::univariate_statistics(acc).mean;
    }

    private:
        double diversity_{};
    };
} // namespace Operon

#endif
