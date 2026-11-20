// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_GENERATOR_HPP
#define OPERON_GENERATOR_HPP

#include <cstddef>
#include <limits>
#include <optional>

#include "operon/core/operator.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/selector.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

struct RecombinationResult {
    std::optional<Operon::Individual> Child;
    std::optional<Operon::Individual> Parent1;
    std::optional<Operon::Individual> Parent2;

    explicit operator bool() const { return Child.has_value(); }
};

class OffspringGeneratorBase : public OperatorBase<std::optional<Individual>, /* crossover prob. */ double,
                                   /* mutation prob. */ double, /* local search prob. */ double,
                                   /* lamarckian prob */ double, /* memory buffer */ Operon::Span<Operon::Scalar>,
                                   /* run optimizer accounting */ OptimizerAccounting&> {
public:
    OffspringGeneratorBase(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut,
        SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : evaluator_(eval)
        , crossover_(cx)
        , mutator_(mut)
        , femaleSelector_(femSel)
        , maleSelector_(maleSel)
        , coeffOptimizer_ { coeffOptimizer }
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

    // The two independent reasons an offspring generator can ask the algorithm to stop. Subclasses can override
    // Terminate() for an additional generator-specific condition; drivers retain that virtual contract and classify
    // an otherwise-unclassified termination as a pressure limit.
    [[nodiscard]] auto BudgetExhausted() const -> bool { return evaluator_->BudgetExhausted(); }
    // Selection pressure: offspring selection has needed more than its maximum candidates per parent (never true
    // for generators without a selection-pressure limit).
    [[nodiscard]] virtual auto PressureExhausted() const -> bool { return false; }
    [[nodiscard]] virtual auto Terminate() const -> bool { return BudgetExhausted() || PressureExhausted(); }

    auto SetCache(Zobrist* cache) const { cache_ = cache; }
    [[nodiscard]] auto Cache() const -> Zobrist* { return cache_; }

    // Every optimizer invocation made while generating a candidate (kept or not) is published to `accounting`, the
    // run's optimizer accounting, before the candidate can be rejected, lose a brood/polygenic selection or be
    // discarded with its generation.
    auto Generate(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck,
        Operon::Span<Operon::Scalar> buf, OptimizerAccounting& accounting, RecombinationResult& res) const -> void
    {
        auto pop = FemaleSelector()->Population();
        if (!res.Parent1) {
            res.Parent1 = pop[(*FemaleSelector())(random)];
        }
        if (!res.Parent2) {
            res.Parent2 = pop[(*MaleSelector())(random)];
        }

        res.Child = Individual { Evaluator()->ObjectiveCount() };
        using BernoulliTrial = std::bernoulli_distribution;

        res.Child->Genotype = BernoulliTrial { pCrossover }(random)
            ? (*Crossover())(random, res.Parent1->Genotype, res.Parent2->Genotype)
            : res.Parent1->Genotype;

        if (BernoulliTrial { pMutation }(random)) {
            res.Child->Genotype = (*Mutator())(random, std::move(res.Child->Genotype));
        }

        auto evaluate = [&]() {
            ScoreIndividual(random, *res.Child, *Evaluator(), coeffOptimizer_, pLocal, pLamarck, buf, accounting);
        };

        // The transposition table keys fitness by structure only. An evaluator whose result depends on the stored
        // coefficients (e.g. ShapeConstrainedValueEvaluator, which publishes a per-genotype completion) opts out: a
        // structural hit could neither certify the recipient genotype nor publish its completion. A cache hit also
        // bypasses ScoreIndividual, hence local search: it costs no optimizer work.
        if (cache_ != nullptr && Evaluator()->StructuralFitnessReusable()) {
            auto const hash = cache_->ComputeHash(res.Child->Genotype);
            Operon::Vector<Operon::Scalar> cached(Evaluator()->ObjectiveCount());
            if (cache_->TryGet(hash, cached)) {
                res.Child->Fitness = cached;
            } else {
                evaluate();
                cache_->Insert(hash, res.Child->Fitness);
            }
        } else {
            evaluate();
        }
    }

    // Same as above, returning a fresh result.
    auto Generate(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck,
        Operon::Span<Operon::Scalar> buf, OptimizerAccounting& accounting) const -> RecombinationResult
    {
        RecombinationResult res;
        Generate(random, pCrossover, pMutation, pLocal, pLamarck, buf, accounting, res);
        return res;
    }

private:
    gsl::not_null<EvaluatorBase const*> evaluator_;
    gsl::not_null<CrossoverBase const*> crossover_;
    gsl::not_null<MutatorBase const*> mutator_;
    gsl::not_null<SelectorBase const*> femaleSelector_;
    gsl::not_null<SelectorBase const*> maleSelector_;
    CoefficientOptimizer const* coeffOptimizer_;
    mutable Zobrist* cache_ { nullptr };
};

class OPERON_EXPORT BasicOffspringGenerator final : public OffspringGeneratorBase {
public:
    explicit BasicOffspringGenerator(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut,
        SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal,
        double pLamarck, Operon::Span<Operon::Scalar> buf, OptimizerAccounting& accounting) const
        -> std::optional<Individual> final;
};

class OPERON_EXPORT BroodOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit BroodOffspringGenerator(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut,
        SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal,
        double pLamarck, Operon::Span<Operon::Scalar> buf, OptimizerAccounting& accounting) const
        -> std::optional<Individual> final;

    void BroodSize(size_t value) { broodSize_ = value; }
    [[nodiscard]] auto BroodSize() const -> size_t { return broodSize_; }

    static constexpr size_t DefaultBroodSize { 10 };

private:
    size_t broodSize_{DefaultBroodSize};
};

class OPERON_EXPORT PolygenicOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit PolygenicOffspringGenerator(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut,
        SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal,
        double pLamarck, Operon::Span<Operon::Scalar> buf, OptimizerAccounting& accounting) const
        -> std::optional<Individual> final;

    void PolygenicSize(size_t value) { broodSize_ = value; }
    [[nodiscard]] auto PolygenicSize() const -> size_t { return broodSize_; }

    static constexpr size_t DefaultBroodSize = 10;

private:
    size_t broodSize_{DefaultBroodSize};
};

class OPERON_EXPORT OffspringSelectionGenerator : public OffspringGeneratorBase {
public:
    explicit OffspringSelectionGenerator(EvaluatorBase const* eval, CrossoverBase const* cx, MutatorBase const* mut,
        SelectorBase const* femSel, SelectorBase const* maleSel, CoefficientOptimizer const* coeffOptimizer = nullptr)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
    {
    }

    auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal,
        double pLamarck, Operon::Span<Operon::Scalar> buf, OptimizerAccounting& accounting) const
        -> std::optional<Individual> final;

    void MaxSelectionPressure(size_t value) { maxSelectionPressure_ = value; }
    auto MaxSelectionPressure() const -> size_t { return maxSelectionPressure_; }

    void ComparisonFactor(double value) { comparisonFactor_ = value; }
    auto ComparisonFactor() const -> double { return comparisonFactor_; }

    void Prepare(const Operon::Span<const Individual> pop) const override
    {
        OffspringGeneratorBase::Prepare(pop);
        lastEvaluations_ = this->Evaluator()->CallCount;
    }

    [[nodiscard]] auto AttemptedCandidates() const -> std::size_t
    {
        return this->Evaluator()->CallCount - lastEvaluations_;
    }

    // Counts real offspring-candidate attempts (one CallCount increment per
    // individual scored, regardless of local search), not local search's own
    // internal residual/Jacobian evaluations - TotalEvaluations() (used here
    // previously) counts the latter, wildly overstating selection pressure
    // whenever local search runs (each attempt can cost dozens of internal
    // LM evaluations, all counted as if they were separate failed
    // candidates).
    auto SelectionPressure() const -> double
    {
        auto n = this->FemaleSelector()->Population().size();
        if (n == 0U) {
            return 0;
        }
        auto e = this->Evaluator()->CallCount - lastEvaluations_;
        return static_cast<double>(e) / static_cast<double>(n);
    }

    auto PressureExhausted() const -> bool override
    {
        return SelectionPressure() > static_cast<double>(maxSelectionPressure_);
    }

    // The selection pressure of a generation is the number of attempts (candidates generated, whether evaluated or
    // taken from the transposition cache) divided by the parents, as in offspring selection genetic algorithms. The
    // generation may make at most MaxSelectionPressure() * parents attempts (saturating); see
    // OffspringGeneratorBase::MaxAttempts() for the exact commit rule. Without a transposition cache an attempt is
    // one evaluator call, and at one thread the rule is `calls / parents > MaxSelectionPressure()` that stopped the
    // generation before the ledger existed.
    [[nodiscard]] auto MaxAttempts(std::size_t parents) const -> std::optional<std::size_t> override
    {
        constexpr auto limit = std::numeric_limits<std::size_t>::max();
        if (parents != 0 && maxSelectionPressure_ > limit / parents) {
            return limit;
        }
        return maxSelectionPressure_ * parents;
    }

    static constexpr size_t DefaultMaxSelectionPressure { 100 };
    static constexpr double DefaultComparisonFactor { 1.0 };

private:
    mutable size_t lastEvaluations_ { 0 };
    size_t maxSelectionPressure_ { DefaultMaxSelectionPressure };
    double comparisonFactor_ { 0 };
};

} // namespace Operon

#endif
