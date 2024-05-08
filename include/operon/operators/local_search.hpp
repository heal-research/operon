// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#ifndef OPERON_LOCAL_SEARCH_HPP
#define OPERON_LOCAL_SEARCH_HPP

#include "operon/core/operator.hpp"
#include "operon/operon_export.hpp"


namespace Operon {

// forward declarations
class Tree;
class OptimizerBase;
struct OptimizerSummary;

class OPERON_EXPORT CoefficientOptimizer : public OperatorBase<OptimizerSummary, Operon::Tree&> {
public:
    explicit CoefficientOptimizer(OptimizerBase const& optimizer, double lmProb = 1.0)
        : optimizer_(optimizer)
        , lamarckianProbability_(lmProb)
    { }

    // convenience
    auto operator()(Operon::RandomGenerator& rng, Operon::Tree& tree) const -> OptimizerSummary override;

private:

    std::reference_wrapper<Operon::OptimizerBase const> optimizer_;
    double lamarckianProbability_{1.0};
};

} // namespace Operon

#endif
