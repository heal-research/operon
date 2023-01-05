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
public:
    Problem(Dataset ds, Operon::Span<Variable const> inputs, Variable targetVariable, Range trainingRange, Range testRange, Range validationRange = { 0, 0 }) // NOLINT(bugprone-easily-swappable-parameters)
        : dataset_(std::move(ds))
        , training_(trainingRange)
        , test_(testRange)
        , validation_(validationRange)
        , target_(std::move(targetVariable))
        , inputVariables_(inputs.begin(), inputs.end())
    {
    }

    auto Target(std::string const& tgt) -> Problem& {
        auto var = dataset_.GetVariable(tgt);
        if (!var.has_value()) {
            throw std::runtime_error(fmt::format("Error: the target name {} does not exist in the dataset.\n", tgt));
        }
        target_ = var.value();
        return *this;
    }

    auto Target(Variable const& tgt) -> Problem& {
        auto var = dataset_.GetVariable(tgt.Hash);
        EXPECT(var.has_value());
        this->target_ = tgt;
        return *this;
    }

    auto TrainingRange(Range range) -> Problem& {
        training_ = range;
        return *this;
    }

    auto TestRange(Range range) -> Problem& {
        test_ = range;
        return *this;
    }

    auto ValidationRange(Range range) -> Problem& {
        validation_ = range;
        return *this;
    }

    auto Inputs(std::vector<Variable> const& inputs) -> Problem& {
        inputVariables_.clear();
        std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputVariables_));
        return *this;
    }

    auto Inputs(Operon::Span<const Variable> inputs) -> Problem& {
        inputVariables_.clear();
        std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputVariables_));
        return *this;
    }

    auto Inputs(std::vector<std::string> const& inputs) -> Problem& {
        std::vector<Variable> tmp;
        for (auto const& s : inputs) {
            auto res = dataset_.GetVariable(s);
            if (!res.has_value()) {
                throw std::runtime_error(fmt::format("The input {} does not exist in the dataset.\n", s));
            }
            tmp.push_back(res.value());
        }
        inputVariables_.swap(tmp);
        return *this;
    }

    [[nodiscard]] auto TrainingRange() const -> Range { return training_; }
    [[nodiscard]] auto TestRange() const -> Range { return test_; }
    [[nodiscard]] auto ValidationRange() const -> Range { return validation_; }

    [[nodiscard]] auto TargetVariable() const -> Variable const& { return target_; }
    [[nodiscard]] auto GetPrimitiveSet() const -> PrimitiveSet const& { return pset_; }
    auto GetPrimitiveSet() -> PrimitiveSet& { return pset_; }
    [[nodiscard]] auto GetDataset() const -> Dataset const& { return dataset_; }
    auto GetDataset() -> Dataset& { return dataset_; }

    [[nodiscard]]  auto InputVariables() const -> Operon::Span<const Variable> { return inputVariables_; }
    auto TargetValues() -> Operon::Span<const Operon::Scalar> { return dataset_.GetValues(target_.Hash); }

    void StandardizeData(Range range)
    {
        for (auto const& var : inputVariables_) {
            dataset_.Standardize(var.Index, range);
        }
    }

    void NormalizeData(Range range) {
        for (auto const& var : inputVariables_) {
            dataset_.Normalize(var.Index, range);
        }
    }

private:
    Dataset dataset_;
    PrimitiveSet pset_;
    Range training_;
    Range test_;
    Range validation_;
    Variable target_;
    std::vector<Variable> inputVariables_;
};
} // namespace Operon

#endif

