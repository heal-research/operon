// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <string>
#include <vector>

#include "dataset.hpp"
#include "pset.hpp"
#include "range.hpp"

namespace Operon {

class Problem {
public:
    Problem(Dataset const& ds) 
        : dataset(ds)
    {
        std::copy(ds.Variables().begin(), ds.Variables().end()-1, std::back_inserter(inputVariables));
        training = Range{0, ds.Rows() / 2};
        test = Range{ds.Rows() / 2, ds.Rows()};
        target = dataset.Variables().back();
    }

    Problem(Dataset const& ds, Operon::Span<const Variable> inputs, Variable const& targetVariable, Range trainingRange, Range testRange, Range validationRange = { 0, 0 })
        : dataset(ds)
        , training(trainingRange)
        , test(testRange)
        , validation(validationRange)
        , target(targetVariable)
        , inputVariables(inputs.begin(), inputs.end())
    {
    }

    Problem& Target(std::string const& tgt) {
        auto var = dataset.GetVariable(tgt);
        EXPECT(var.has_value());
        target = var.value();
        return *this;
    }

    Problem& Target(Variable const& tgt) {
        auto var = dataset.GetVariable(tgt.Hash);
        EXPECT(var.has_value());
        this->target = tgt;
        return *this;
    }

    Problem& TrainingRange(Range range) {
        training = range;
        return *this;
    }

    Problem& TestRange(Range range) {
        test = range;
        return *this;
    }

    Problem& ValidationRange(Range range) {
        validation = range;
        return *this;
    }

    Problem& Inputs(std::vector<Variable> const& inputs) {
        inputVariables.clear();
        std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputVariables));
        return *this;
    }

    Problem& Inputs(Operon::Span<const Variable> inputs) {
        inputVariables.clear();
        std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputVariables));
        return *this;
    }

    Problem& Inputs(std::vector<std::string> const& inputs) {
        std::vector<Variable> tmp;
        for (auto const& s : inputs) {
            auto res = dataset.GetVariable(s);
            if (!res.has_value()) {
                throw std::runtime_error(fmt::format("The input {} does not exist in the dataset.\n", s));
            }
            tmp.push_back(res.value());
        }
        inputVariables.swap(tmp);
        return *this;
    }

    Range TrainingRange() const { return training; }
    Range TestRange() const { return test; }
    Range ValidationRange() const { return validation; }

    const Variable& TargetVariable() const { return target; }
    const PrimitiveSet& GetPrimitiveSet() const { return pset; }
    PrimitiveSet& GetPrimitiveSet() { return pset; }
    const Dataset& GetDataset() const { return dataset; }
    Dataset& GetDataset() { return dataset; }

    const Operon::Span<const Variable> InputVariables() const { return inputVariables; }
    const Operon::Span<const Operon::Scalar> TargetValues() { return dataset.GetValues(target.Hash); }

    void StandardizeData(Range range)
    {
        for (auto const& var : inputVariables) {
            dataset.Standardize(var.Index, range);
        }
    }

    void NormalizeData(Range range) {
        for (auto const& var : inputVariables) {
            dataset.Normalize(var.Index, range);
        }
    }

private:
    Dataset dataset;
    PrimitiveSet pset;
    Range training;
    Range test;
    Range validation;
    Variable target;
    std::vector<Variable> inputVariables;
};
}

#endif
