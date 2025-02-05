// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <string>
#include <utility>
#include <vector>
#include <gsl/pointers>

#include "dataset.hpp"
#include "pset.hpp"
#include "range.hpp"

namespace Operon {

class Problem {
    [[nodiscard]] auto GetVariable(auto t) const -> Operon::Variable {
        if (auto v = dataset_->GetVariable(t); v.has_value()) { return *v; }
        PANIC("cannot map argument to any known variable");
    }

    auto ValidateInputs(auto const& inputs) const {
        using T = typename std::remove_cvref_t<decltype(inputs)>::value_type;
        static_assert(std::is_same_v<T, std::string> || std::is_same_v<T, Operon::Hash>, "the inputs must be strings or hashes");
        for (auto const& x : inputs) { (void) GetVariable(x); }
    }

    std::unique_ptr<Dataset> dataset_;

    Range training_;
    Range test_;
    Range validation_;

    PrimitiveSet pset_;
    Operon::Variable target_;
    Operon::Set<Operon::Hash> inputs_;

    bool ownership_{true};

public:
    Problem(const Problem&) = delete;
    Problem(Problem&&) = delete;
    auto operator=(const Problem&) -> Problem& = delete;
    auto operator=(Problem&&) -> Problem& = delete;

    explicit Problem(std::unique_ptr<Dataset> dataset)
        : dataset_ { std::move(dataset) }
    {
        SetDefaultInputs();
    }

    explicit Problem(gsl::not_null<Dataset*> dataset)
        : Problem(std::unique_ptr<Dataset>{dataset})
    {
        ownership_ = false;
    }

    ~Problem() {
        if (!ownership_) {
            std::ignore = dataset_.release();
        }
    }

    template<typename T>
    auto SetTarget(T t) {
        target_ = GetVariable<std::remove_cvref_t<T>>(t);
    }

    auto SetTrainingRange(Operon::Range range) { training_ = range; }
    auto SetTrainingRange(int begin, int end) { training_ = Operon::Range(begin, end); }

    auto SetTestRange(Operon::Range range) { test_ = range; }
    auto SetTestRange(int begin, int end) { test_ = Operon::Range(begin, end); }

    auto SetValidationRange(Operon::Range range) { validation_ = range; }
    auto SetValidationRange(int begin, int end) { validation_ = Operon::Range(begin, end); }

    auto SetInputs(auto const& inputs) {
        ValidateInputs(inputs);
        inputs_.clear();
        for (auto const& x : inputs) {
            inputs_.insert(GetVariable(x).Hash);
        }
    }

    [[nodiscard]] auto GetInputs() const -> std::vector<Operon::Hash> const& {
        return inputs_.values();
    }

    // set all variables except the target as inputs
    auto SetDefaultInputs() -> void {
        inputs_.clear();
        for (auto const& v : dataset_->GetVariables()) {
            if (v.Hash != target_.Hash) { inputs_.insert(v.Hash); }
        }
    }

    [[nodiscard]] auto TrainingRange() const -> Range { return training_; }
    [[nodiscard]] auto TestRange() const -> Range { return test_; }
    [[nodiscard]] auto ValidationRange() const -> Range { return validation_; }

    [[nodiscard]] auto TargetVariable() const -> Variable const& { return target_; }
    [[nodiscard]] auto InputVariables() const -> std::vector<Variable>
    {
        std::vector<Variable> variables; variables.reserve(inputs_.size());
        std::transform(inputs_.values().begin(), inputs_.values().end(), std::back_inserter(variables),
            [&](auto h) { return GetVariable<Operon::Hash>(h); });
        return variables;
    }

    [[nodiscard]] auto GetPrimitiveSet() const -> PrimitiveSet const& { return pset_; }
    auto GetPrimitiveSet() -> PrimitiveSet& { return pset_; }
    auto ConfigurePrimitiveSet(Operon::PrimitiveSetConfig config) { pset_.SetConfig(config); }

    [[nodiscard]] auto GetDataset() const -> Dataset const* { return dataset_.get(); }
    auto GetDataset() -> Dataset* { return dataset_.get(); }

    [[nodiscard]] auto TargetValues() const -> Operon::Span<Operon::Scalar const> { return dataset_->GetValues(target_.Index); }
    [[nodiscard]] auto TargetValues(Operon::Range range) const -> Operon::Span<Operon::Scalar const> {
        return dataset_->GetValues(target_.Index).subspan(range.Start(), range.Size());
    }

    void StandardizeData(Range range)
    {
        for (auto const& v : inputs_) {
            dataset_->Standardize(GetVariable<Operon::Hash>(v).Index, range);
        }
    }

    void NormalizeData(Range range)
    {
        for (auto const& v : inputs_) {
            dataset_->Normalize(GetVariable<Operon::Hash>(v).Index, range);
        }
    }
};
} // namespace Operon

#endif
