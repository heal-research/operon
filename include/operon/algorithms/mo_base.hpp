// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_MO_BASE_HPP
#define OPERON_MO_BASE_HPP

#include <functional>
#include <operon/operon_export.hpp>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/ga_base.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/generator.hpp"

namespace tf { class Executor; }

namespace Operon {

class NondominatedSorterBase;
class Problem;
class ReinserterBase;
struct CoefficientInitializerBase;
struct TreeInitializerBase;

class OPERON_EXPORT MultiObjectiveGABase : public GeneticAlgorithmBase {
    gsl::not_null<NondominatedSorterBase const*> sorter_;
    Operon::Vector<Operon::Vector<size_t>> fronts_;
    Operon::Vector<Individual> best_;

    auto Sort(Operon::Span<Individual> pop) -> void;

protected:
    virtual auto UpdateDistance(Operon::Span<Individual> pop) -> void = 0;

public:
    MultiObjectiveGABase(GeneticAlgorithmConfig config, gsl::not_null<Problem const*> problem, gsl::not_null<TreeInitializerBase const*> treeInit, gsl::not_null<CoefficientInitializerBase const*> coeffInit, gsl::not_null<OffspringGeneratorBase const*> generator, gsl::not_null<ReinserterBase const*> reinserter, gsl::not_null<NondominatedSorterBase const*> sorter)
        : GeneticAlgorithmBase(config, problem, treeInit, coeffInit, generator, reinserter)
        , sorter_(sorter)
    {
        auto const n { GetGenerator()->Evaluator()->ObjectiveCount() };
        for (auto& ind : Individuals()) {
            ind.Fitness.resize(n, EvaluatorBase::ErrMax);
        }
    }

    [[nodiscard]] auto Best() const -> Operon::Span<Individual const> { return { best_.data(), best_.size() }; }
    [[nodiscard]] auto Fronts() const -> Operon::Vector<Operon::Vector<size_t>> const& { return fronts_; }

    auto Run(tf::Executor&, Operon::RandomGenerator&, std::function<void()> = nullptr, bool = false) -> void;
    auto Run(Operon::RandomGenerator&, std::function<void()> = nullptr, size_t = 0, bool = false) -> void;
};

} // namespace Operon

#endif
