/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <string>
#include <vector>

#include "dataset.hpp"
#include "grammar.hpp"

namespace Operon {
struct Solution {
    Tree Model;
    Operon::Scalar TrainR2;
    Operon::Scalar TestR2;
    Operon::Scalar TrainNmse;
    Operon::Scalar TestNmse;
};

class Problem {
public:
    Problem(const Dataset& ds, gsl::span<const Variable> allVariables, std::string targetVariable, Range trainingRange, Range testRange, Range validationRange = { 0, 0 })
        : dataset(ds)
        , training(trainingRange)
        , test(testRange)
        , validation(validationRange)
        , target(targetVariable)
    {
        std::copy_if(allVariables.begin(), allVariables.end(), std::back_inserter(inputVariables), [&](const auto& v) { return v.Name != targetVariable; });
        std::sort(inputVariables.begin(), inputVariables.end(), [](const auto& lhs, const auto& rhs) { return lhs.Hash < rhs.Hash; });
    }

    Range TrainingRange() const { return training; }
    Range TestRange() const { return test; }
    Range ValidationRange() const { return validation; }

    const std::string& TargetVariable() const { return target; }
    const Grammar& GetGrammar() const { return grammar; }
    Grammar& GetGrammar() { return grammar; }
    const Dataset& GetDataset() const { return dataset; }
    Dataset& GetDataset() { return dataset; }

    const gsl::span<const Variable> InputVariables() const { return inputVariables; }
    const gsl::span<const Operon::Scalar> TargetValues() const { return dataset.GetValues(target); }

    Solution CreateSolution(const Tree&) const;

    void StandardizeData(Range range)
    {
        for (const auto& var : inputVariables) {
            dataset.Standardize(var.Index, range);
        }
    }

    void NormalizeData(Range range) {
        for (const auto& var : inputVariables) {
            dataset.Normalize(var.Index, range);
        }
    }

private:
    Dataset dataset;
    Grammar grammar;
    Range training;
    Range test;
    Range validation;
    std::string target;
    std::vector<Variable> inputVariables;
};
}

#endif
