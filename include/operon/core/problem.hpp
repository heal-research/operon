#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <vector>
#include <string>

#include "dataset.hpp"
#include "grammar.hpp"

namespace Operon
{
    struct Solution
    {
        Tree Model;
        double TrainR2;
        double TestR2;
        double TrainNmse;
        double TestNmse;
    };

    class Problem
    {
        public:
            Problem(const Dataset& ds, 
                    std::vector<std::string> inputVars, 
                    std::string targetVariable, 
                    Range trainingRange, 
                    Range testRange, 
                    Range validationRange = {0, 0}
                    )
                : dataset(ds), training(trainingRange), test(testRange), validation(validationRange), target(targetVariable), inputVariables(inputVars)
                {
                    std::vector<operon::hash_t> inputs;
                    std::transform(inputVariables.begin(), inputVariables.end(), std::back_inserter(inputs), [&](const std::string& name) { return dataset.GetHashValue(name); });
                    std::sort(inputs.begin(), inputs.end());
                }

            Range TrainingRange()               const { return training;   }
            Range TestRange()                   const { return test;       }
            Range ValidationRange()             const { return validation; }

            const std::string& TargetVariable() const { return target;  }
            const Grammar& GetGrammar()         const { return grammar; }
            Grammar& GetGrammar()                     { return grammar; }
            const Dataset& GetDataset()         const { return dataset; }
            Dataset& GetDataset()                     { return dataset; }
            const std::vector<std::string>& InputVariables() const { return inputVariables; }

            Solution CreateSolution(const Tree&) const;
            

        private:
            Dataset     dataset;
            Grammar     grammar;
            Range       training;
            Range       test;
            Range       validation;
            std::string target;
            std::vector<std::string> inputVariables;
    };
}

#endif
