#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <string>
#include <vector>

#include "dataset.hpp"
#include "grammar.hpp"

namespace Operon {
struct Solution {
    Tree Model;
    double TrainR2;
    double TestR2;
    double TrainNmse;
    double TestNmse;
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
    const gsl::span<const double> TargetValues() const { return dataset.GetValues(target); }

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
