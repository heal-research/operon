// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <string>
#include <utility>
#include <vector>

#include "dataset.hpp"
#include "pset.hpp"
#include "range.hpp"

namespace Operon {

class Problem {
private:
    template<typename T>
    auto GetVariable(std::conditional_t<std::is_integral_v<T>, T, T const&> t) -> Operon::Variable {
        if (auto v = dataset_.GetVariable(t); v.has_value()) { return *v; }
        throw std::runtime_error(fmt::format("a variable identified by {} {} does not exist in the dataset", typeid(t).name(), t));
    }

    template<typename T>
    auto HasVariable(std::conditional_t<std::is_integral_v<T>, T, T const&> t) -> bool {
        return dataset_.GetVariable(t).has_value();
    }

    Dataset dataset_;
    PrimitiveSet pset_;
    Range training_;
    Range test_;
    Range validation_;

    Operon::Variable target_;
    Operon::Set<Operon::Hash> inputs_;

public:
    Problem(Dataset ds, Range trainingRange, Range testRange, Range validationRange = { 0, 0 }) // NOLINT(bugprone-easily-swappable-parameters)
        : dataset_(std::move(ds))
        , training_(std::move(trainingRange))
        , test_(std::move(testRange))
        , validation_(std::move(validationRange))
    {
        target_ = dataset_.GetVariables().back();
        SetDefaultInputs();
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

    auto SetInputs(std::vector<std::string> const& inputs) {
        for (auto const& s : inputs) {
            auto v = GetVariable<std::string>(s);
        }
    }

    auto SetInputs(std::vector<Operon::Hash> const& inputs) {
        inputs_.clear();
        for (auto v : inputs) {
            inputs_.insert(GetVariable<Operon::Hash>(v).Hash);
        }
    }

    // set all variables except the target as inputs
    auto SetDefaultInputs() -> void {
        inputs_.clear();
        for (auto const& v : dataset_.GetVariables()) {
            if (v.Hash != target_.Hash) { inputs_.insert(v.Hash); }
        }
    }

    [[nodiscard]] auto TrainingRange() const -> Range { return training_; }
    [[nodiscard]] auto TestRange() const -> Range { return test_; }
    [[nodiscard]] auto ValidationRange() const -> Range { return validation_; }

    [[nodiscard]] auto TargetVariable() const -> Variable const& { return target_; }
    [[nodiscard]] auto InputVariables() const -> std::vector<Operon::Hash> const& { return inputs_.values(); }

    [[nodiscard]] auto GetPrimitiveSet() const -> PrimitiveSet const& { return pset_; }
    auto GetPrimitiveSet() -> PrimitiveSet& { return pset_; }
    auto ConfigurePrimitiveSet(Operon::PrimitiveSetConfig config) { pset_.SetConfig(config); }

    [[nodiscard]] auto GetDataset() const -> Dataset const& { return dataset_; }
    auto GetDataset() -> Dataset& { return dataset_; }

    [[nodiscard]] auto TargetValues() const -> Operon::Span<Operon::Scalar const> { return dataset_.GetValues(target_.Index); }

    void StandardizeData(Range range)
    {
        for (auto const& v : inputs_) {
            dataset_.Standardize(GetVariable<int>(v).Index, range);
        }
    }

    void NormalizeData(Range range)
    {
        for (auto const& v : inputs_) {
            dataset_.Normalize(GetVariable<int>(v).Index, range);
        }
    }
};
} // namespace Operon

#endif

