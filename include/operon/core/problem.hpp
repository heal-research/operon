/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
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
    operon::scalar_t TrainR2;
    operon::scalar_t TestR2;
    operon::scalar_t TrainNmse;
    operon::scalar_t TestNmse;
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
    const gsl::span<const operon::scalar_t> TargetValues() const { return dataset.GetValues(target); }

    Solution CreateSolution(const Tree&) const;

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
