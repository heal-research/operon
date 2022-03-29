// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_GENERATOR_HPP
#define OPERON_GENERATOR_HPP

#include "operon/core/operator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/selector.hpp"

namespace Operon {

class OffspringGeneratorBase : public OperatorBase<std::optional<Individual>, /* crossover prob. */ double, /* mutation prob. */ double, /* memory buffer */ Operon::Span<Operon::Scalar>> {
public:
    OffspringGeneratorBase(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : evaluator_(eval)
        , crossover_(cx)
        , mutator_(mut)
        , femaleSelector_(femSel)
        , maleSelector_(maleSel)
    {
    }

    [[nodiscard]] auto FemaleSelector() const -> SelectorBase& { return femaleSelector_.get(); }
    [[nodiscard]] auto MaleSelector() const -> SelectorBase& { return maleSelector_.get(); }
    [[nodiscard]] auto Crossover() const -> CrossoverBase& { return crossover_.get(); }
    [[nodiscard]] auto Mutator() const -> MutatorBase& { return mutator_.get(); }
    [[nodiscard]] auto Evaluator() const -> EvaluatorBase& { return evaluator_.get(); }

    // this method is necessary in order to avoid a code smell (default function arguments of virtual method)
    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation) const -> std::optional<Individual>
    {
        const auto* base = static_cast<OperatorBase<std::optional<Individual>, double, double, Operon::Span<Operon::Scalar>> const*>(this);
        return (*base)(random, pCrossover, pMutation, Operon::Span<Operon::Scalar> {});
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> override
    {
        const auto* base = static_cast<OperatorBase<std::optional<Individual>, double, double, Operon::Span<Operon::Scalar>> const*>(this);
        return (*base)(random, pCrossover, pMutation, buf);
    };

    virtual void Prepare(Operon::Span<const Individual> pop) const
    {
        this->FemaleSelector().Prepare(pop);
        this->MaleSelector().Prepare(pop);
    }
    [[nodiscard]] virtual auto Terminate() const -> bool { return evaluator_.get().BudgetExhausted(); }

private:
    std::reference_wrapper<EvaluatorBase> evaluator_;
    std::reference_wrapper<CrossoverBase> crossover_;
    std::reference_wrapper<MutatorBase> mutator_;
    std::reference_wrapper<SelectorBase> femaleSelector_;
    std::reference_wrapper<SelectorBase> maleSelector_;
};

class OPERON_EXPORT BasicOffspringGenerator final : public OffspringGeneratorBase {
public:
    explicit BasicOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> override;
};

class OPERON_EXPORT BroodOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit BroodOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
        , broodSize_(DefaultBroodSize)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> override;

    void BroodSize(size_t value) { broodSize_ = value; }
    [[nodiscard]] auto BroodSize() const -> size_t { return broodSize_; }

    static constexpr size_t DefaultBroodSize { 10 };

private:
    size_t broodSize_;
};

class OPERON_EXPORT PolygenicOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit PolygenicOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
        , broodSize_(DefaultBroodSize)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> override;

    void PolygenicSize(size_t value) { broodSize_ = value; }
    [[nodiscard]] auto PolygenicSize() const -> size_t { return broodSize_; }

    static constexpr size_t DefaultBroodSize = 10;

private:
    size_t broodSize_;
};

class OPERON_EXPORT OffspringSelectionGenerator : public OffspringGeneratorBase {
public:
    explicit OffspringSelectionGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
        , lastEvaluations_(0)
        , maxSelectionPressure_(DefaultMaxSelectionPressure)
        , comparisonFactor_(1.0)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> override;

    void MaxSelectionPressure(size_t value) { maxSelectionPressure_ = value; }
    auto MaxSelectionPressure() const -> size_t { return maxSelectionPressure_; }

    void ComparisonFactor(double value) { comparisonFactor_ = value; }
    auto ComparisonFactor() const -> double { return comparisonFactor_; }

    void Prepare(const Operon::Span<const Individual> pop) const override
    {
        OffspringGeneratorBase::Prepare(pop);
        lastEvaluations_ = this->Evaluator().EvaluationCount();
    }

    auto SelectionPressure() const -> double
    {
        auto n = this->FemaleSelector().Population().size();
        if (n == 0U) {
            return 0;
        }
        auto e = this->Evaluator().EvaluationCount() - lastEvaluations_;
        return static_cast<double>(e) / static_cast<double>(n);
    }

    auto Terminate() const -> bool override
    {
        return OffspringGeneratorBase::Terminate() || SelectionPressure() > static_cast<double>(maxSelectionPressure_);
    };

    static constexpr size_t DefaultMaxSelectionPressure { 100 };
    static constexpr double DefaultComparisonFactor { 1.0 };

private:
    mutable size_t lastEvaluations_;
    size_t maxSelectionPressure_;
    double comparisonFactor_;
};

} // namespace Operon

#endif
