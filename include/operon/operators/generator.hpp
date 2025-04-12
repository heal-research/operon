// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GENERATOR_HPP
#define OPERON_GENERATOR_HPP

#include "operon/core/operator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/selector.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/hash/zobrist.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace Operon {

struct RecombinationResult {
    std::optional<Operon::Individual> Child;
    std::optional<Operon::Individual> Parent1;
    std::optional<Operon::Individual> Parent2;

    explicit operator bool() const { return Child.has_value(); }
};

class OffspringGeneratorBase : public OperatorBase<std::optional<Individual>, /* crossover prob. */ double, /* mutation prob. */ double, /* local search prob. */ double, /* lamarckian prob */ double, /* memory buffer */ Operon::Span<Operon::Scalar>> {
public:
    OffspringGeneratorBase(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut, SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : evaluator_(eval)
        , crossover_(cx)
        , mutator_(mut)
        , femaleSelector_(femSel)
        , maleSelector_(maleSel)
        , coeffOptimizer_{coeffOptimizer}
    {
    }

    [[nodiscard]] auto FemaleSelector() const -> SelectorBase const* { return femaleSelector_.get(); }
    [[nodiscard]] auto MaleSelector() const -> SelectorBase const* { return maleSelector_.get(); }
    [[nodiscard]] auto Crossover() const -> CrossoverBase const* { return crossover_.get(); }
    [[nodiscard]] auto Mutator() const -> MutatorBase const* { return mutator_.get(); }
    [[nodiscard]] auto Evaluator() const -> EvaluatorBase const* { return evaluator_.get(); }
    [[nodiscard]] auto Optimizer() const -> CoefficientOptimizer const* { return coeffOptimizer_; }

    virtual auto Prepare(Operon::Span<Individual const> pop) const -> void
    {
        FemaleSelector()->Prepare(pop);
        MaleSelector()->Prepare(pop);
        Evaluator()->Prepare(pop);
    }

    [[nodiscard]] virtual auto Terminate() const -> bool { return evaluator_->BudgetExhausted(); }

    auto Generate(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf, RecombinationResult& res) const -> void {
        auto pop = FemaleSelector()->Population();
        if (!res.Parent1) { res.Parent1 = pop[ (*FemaleSelector())(random) ]; }
        if (!res.Parent2) { res.Parent2 = pop[ (*MaleSelector())(random) ]; }

        res.Child = Individual{Evaluator()->ObjectiveCount()};
        using BernoulliTrial = std::bernoulli_distribution;

        res.Child->Genotype = BernoulliTrial{pCrossover}(random)
            ? (*Crossover())(random, res.Parent1->Genotype, res.Parent2->Genotype)
            : res.Parent1->Genotype;

        if (BernoulliTrial{pMutation}(random)) {
            res.Child->Genotype = (*Mutator())(random, std::move(res.Child->Genotype));
        }

        auto evaluate = [&]() {
            if (BernoulliTrial{pLocal}(random)) {
                auto c = res.Child->Genotype.GetCoefficients(); // save original coefficients
                auto [optimizedTree, summary] = (*Optimizer())(random, std::move(res.Child->Genotype));
                Evaluator()->ResidualEvaluations += summary.FunctionEvaluations;
                Evaluator()->JacobianEvaluations += summary.JacobianEvaluations;
                res.Child->Genotype = std::move(optimizedTree);
                res.Child->Fitness = (*Evaluator())(random, *res.Child, buf);

                if(!BernoulliTrial{pLamarck}(random)) {
                    res.Child->Genotype.SetCoefficients(c);     // restore original coefficients
                }
            } else {
                res.Child->Fitness = (*Evaluator())(random, *res.Child, buf);
            }
            for (auto& v : res.Child->Fitness) {
                if (!std::isfinite(v)) { v = std::numeric_limits<Operon::Scalar>::max(); }
            }
        };

        auto* zob       = Zobrist::GetInstance();
        auto const hash = zob->ComputeHash(res.Child->Genotype);

        auto assignCachedFitness = [&](auto const& t) {
            auto const& [ind, cnt] = t.second;
            res.Child->Fitness = ind.Fitness;
        };

        if (!(useTranspositionCache_ && zob->TranspositionTable().if_contains(hash, assignCachedFitness))) {
            evaluate();
        }

        zob->Insert(hash, *res.Child);
    }

    auto Generate(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> RecombinationResult {
        RecombinationResult res;
        Generate(random, pCrossover, pMutation, pLocal, pLamarck, buf, res);
        return res;
    }

    auto UseTranspositionCache(bool value) { useTranspositionCache_ = value; }

private:
    gsl::not_null<EvaluatorBase const*> evaluator_;
    gsl::not_null<CrossoverBase const*> crossover_;
    gsl::not_null<MutatorBase const*>   mutator_;
    gsl::not_null<SelectorBase const*>  femaleSelector_;
    gsl::not_null<SelectorBase const*>  maleSelector_;
    CoefficientOptimizer const*         coeffOptimizer_;
    bool                                useTranspositionCache_{false};
};

class OPERON_EXPORT BasicOffspringGenerator final : public OffspringGeneratorBase {
public:
    explicit BasicOffspringGenerator(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut, SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual> final;
};

class OPERON_EXPORT BroodOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit BroodOffspringGenerator(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut, SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
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
    explicit PolygenicOffspringGenerator(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut, SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
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
    explicit OffspringSelectionGenerator(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut, SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
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
        lastEvaluations_ = this->Evaluator()->TotalEvaluations();
    }

    auto SelectionPressure() const -> double
    {
        auto n = this->FemaleSelector()->Population().size();
        if (n == 0U) {
            return 0;
        }
        auto e = this->Evaluator()->TotalEvaluations() - lastEvaluations_;
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
