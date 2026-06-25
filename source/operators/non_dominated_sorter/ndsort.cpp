// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <ndsort/ndsort.hpp>

#include "operon/core/individual.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

namespace Operon {
namespace {
    auto const Proj = [](Individual const& ind) -> Vector<Scalar> const& { return ind.Fitness; };

    template<typename S>
    auto Wrap(Span<Individual const> pop, Scalar eps) -> NondominatedSorterBase::Result {
        return S{}(pop, static_cast<double>(eps), Proj, ndsort::sorted_unique);
    }
} // namespace

auto DeductiveSorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::deductive_sorter>(pop, eps); }
auto DominanceDegreeSorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::dominance_degree_sorter>(pop, eps); }
auto HierarchicalSorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::hierarchical_sorter>(pop, eps); }
auto EfficientBinarySorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::efficient_binary_sorter>(pop, eps); }
auto EfficientSequentialSorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::efficient_sequential_sorter>(pop, eps); }
auto MergeSorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::merge_sorter>(pop, eps); }
auto RankOrdinalSorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::rank_ordinal_sorter>(pop, eps); }
auto RankIntersectSorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::rank_intersect_sorter>(pop, eps); }
auto BestOrderSorter::Sort(Span<Individual const> pop, Scalar eps) const -> Result { return Wrap<ndsort::best_order_sorter>(pop, eps); }

} // namespace Operon
