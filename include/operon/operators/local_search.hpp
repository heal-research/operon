// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#ifndef OPERON_LOCAL_SEARCH_HPP
#define OPERON_LOCAL_SEARCH_HPP

#include <gsl/pointers>
#include "operon/core/operator.hpp"
#include "operon/operon_export.hpp"


namespace Operon {

// forward declarations
class Tree;
class OptimizerBase;
struct OptimizerSummary;

class OPERON_EXPORT CoefficientOptimizer : public OperatorBase<std::tuple<Operon::Tree, OptimizerSummary>, Operon::Tree> {
public:
    explicit CoefficientOptimizer(gsl::not_null<OptimizerBase const*> optimizer)
        : optimizer_(optimizer)
    { }

    // convenience
    auto operator()(Operon::RandomGenerator& rng, Operon::Tree tree) const -> std::tuple<Operon::Tree, OptimizerSummary> override;

private:
    gsl::not_null<Operon::OptimizerBase const*> optimizer_;
};

} // namespace Operon

#endif
