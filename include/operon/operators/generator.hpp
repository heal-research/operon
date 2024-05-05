// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GENERATOR_HPP
#define OPERON_GENERATOR_HPP

#include "operon/core/operator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/selector.hpp"

namespace Operon {

struct RecombinationResult {
    std::optional<Operon::Individual> Child;
    std::optional<Operon::Individual> Parent1;
    std::optional<Operon::Individual> Parent2;

    explicit operator bool() const { return Child.has_value(); }
};

class OffspringGeneratorBase : public OperatorBase<std::optional<Individual>, /* crossover prob. */ double, /* mutation prob. */ double, /* local search prob. */ double, /* lamarckian prob. */ double, /* memory buffer */ Operon::Span<Operon::Scalar>> {
public:
    OffspringGeneratorBase(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel, OptimizerBase const* opt = nullptr)
        : evaluator_(eval)
        , crossover_(cx)
        , mutator_(mut)
        , femaleSelector_(femSel)
        , maleSelector_(maleSel)
        , optimizer_{opt}
    {
    }

    [[nodiscard]] auto FemaleSelector() const -> SelectorBase& { return femaleSelector_.get(); }
    [[nodiscard]] auto MaleSelector() const -> SelectorBase& { return maleSelector_.get(); }
    [[nodiscard]] auto Crossover() const -> CrossoverBase& { return crossover_.get(); }
    [[nodiscard]] auto Mutator() const -> MutatorBase& { return mutator_.get(); }
    [[nodiscard]] auto Evaluator() const -> EvaluatorBase& { return evaluator_.get(); }
    [[nodiscard]] auto Optimizer() const -> OptimizerBase const* { return optimizer_; }

    virtual auto Prepare(Operon::Span<Individual const> pop) const -> void
    {
        this->FemaleSelector().Prepare(pop);
        this->MaleSelector().Prepare(pop);
        this->Evaluator().Prepare(pop);
    }

    [[nodiscard]] virtual auto Terminate() const -> bool { return evaluator_.get().BudgetExhausted(); }

    auto Generate(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf, RecombinationResult& res) const -> void {
        auto doCrossover   = std::bernoulli_distribution(pCrossover)(random);
        auto doMutation    = std::bernoulli_distribution(pMutation)(random);
        auto doLocalSearch = std::bernoulli_distribution(pLocal)(random);
        auto keepChanges   = std::bernoulli_distribution(pLamarck)(random);

        if (!(doCrossover || doMutation)) {
            return;
        }

        auto pop = FemaleSelector().Population();

        if (!res.Parent1) {
            res.Parent1 = pop[ FemaleSelector()(random) ];
        }

        if (doCrossover) {
            if (!res.Parent2) {
                res.Parent2 = pop[ MaleSelector()(random) ];
            }
            res.Child = Individual{};
            ENSURE(res.Parent1->Genotype.Length() > 0);
            ENSURE(res.Parent2->Genotype.Length() > 0);
            res.Child->Genotype = this->Crossover()(random, res.Parent1->Genotype, res.Parent2->Genotype);
        }

        if (doMutation) {
            if (!res) { res.Child = Individual{}; }
            res.Child->Genotype = doCrossover
                ? this->Mutator()(random, std::move(res.Child->Genotype))
                : this->Mutator()(random, res.Parent1->Genotype);
        }

        auto coefficients = res.Child->Genotype.GetCoefficients();

        auto const* opt = Optimizer();
        if (doLocalSearch && (opt != nullptr && opt->Iterations() > 0)) {
            auto summary = opt->Optimize(random, res.Child->Genotype);
            // update budget counts in the evaluator
            Evaluator().ResidualEvaluations += summary.FunctionEvaluations;
            Evaluator().JacobianEvaluations += summary.JacobianEvaluations;
            if (summary.Success) {
                res.Child->Genotype.SetCoefficients(summary.FinalParameters);
            }
        }

        res.Child->Fitness = Evaluator()(random, res.Child.value(), buf);
        for (auto& v : res.Child->Fitness) {
            if (!std::isfinite(v)) { v = std::numeric_limits<Operon::Scalar>::max(); }
        }

        if (!keepChanges) {
            // revert tree coefficients to the previous values
            res.Child->Genotype.SetCoefficients(coefficients);
        }
    }

    auto Generate(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> RecombinationResult {
        RecombinationResult res;
        Generate(random, pCrossover, pMutation, pLocal, pLamarck, buf, res);
        return res;
    }

private:
    std::reference_wrapper<EvaluatorBase> evaluator_;
    std::reference_wrapper<CrossoverBase> crossover_;
    std::reference_wrapper<MutatorBase>   mutator_;
    std::reference_wrapper<SelectorBase>  femaleSelector_;
    std::reference_wrapper<SelectorBase>  maleSelector_;
    OptimizerBase const* optimizer_{nullptr};
};

class OPERON_EXPORT BasicOffspringGenerator final : public OffspringGeneratorBase {
public:
    explicit BasicOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel, OptimizerBase const* opt = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, opt)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> final;
};

class OPERON_EXPORT BroodOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit BroodOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel, OptimizerBase const* opt = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, opt)
        , broodSize_(DefaultBroodSize)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> final;

    void BroodSize(size_t value) { broodSize_ = value; }
    [[nodiscard]] auto BroodSize() const -> size_t { return broodSize_; }

    static constexpr size_t DefaultBroodSize { 10 };

private:
    size_t broodSize_;
};

class OPERON_EXPORT PolygenicOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit PolygenicOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel, OptimizerBase const* opt = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, opt)
        , broodSize_(DefaultBroodSize)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> final;

    void PolygenicSize(size_t value) { broodSize_ = value; }
    [[nodiscard]] auto PolygenicSize() const -> size_t { return broodSize_; }

    static constexpr size_t DefaultBroodSize = 10;

private:
    size_t broodSize_;
};

class OPERON_EXPORT OffspringSelectionGenerator : public OffspringGeneratorBase {
public:
    explicit OffspringSelectionGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel, OptimizerBase const* opt = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, opt)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> final;

    void MaxSelectionPressure(size_t value) { maxSelectionPressure_ = value; }
    auto MaxSelectionPressure() const -> size_t { return maxSelectionPressure_; }

    void ComparisonFactor(double value) { comparisonFactor_ = value; }
    auto ComparisonFactor() const -> double { return comparisonFactor_; }

    void Prepare(const Operon::Span<const Individual> pop) const override
    {
        OffspringGeneratorBase::Prepare(pop);
        lastEvaluations_ = this->Evaluator().TotalEvaluations();
    }

    auto SelectionPressure() const -> double
    {
        auto n = this->FemaleSelector().Population().size();
        if (n == 0U) {
            return 0;
        }
        auto e = this->Evaluator().TotalEvaluations() - lastEvaluations_;
        return static_cast<double>(e) / static_cast<double>(n);
    }

    auto Terminate() const -> bool override
    {
        return OffspringGeneratorBase::Terminate() || SelectionPressure() > static_cast<double>(maxSelectionPressure_);
    };

    static constexpr size_t DefaultMaxSelectionPressure { 100 };
    static constexpr double DefaultComparisonFactor { 1.0 };

private:
    mutable size_t lastEvaluations_{0};
    size_t maxSelectionPressure_{DefaultMaxSelectionPressure};
    double comparisonFactor_{0};
};

} // namespace Operon

#endif
