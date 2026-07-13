// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LOCAL_SEARCH_HPP
#define OPERON_LOCAL_SEARCH_HPP

#include <gsl/pointers>
#include <tl/expected.hpp>
#include "operon/core/operator.hpp"
#include "operon/operon_export.hpp"


namespace Operon {

// forward declarations
class Tree;
class OptimizerBase;
struct FitResult;
struct FitFailure;

class OPERON_EXPORT CoefficientOptimizer : public OperatorBase<std::tuple<Operon::Tree, tl::expected<FitResult, FitFailure>>, Operon::Tree> {
public:
    explicit CoefficientOptimizer(gsl::not_null<OptimizerBase const*> optimizer)
        : optimizer_(optimizer)
    { }

    auto operator()(Operon::RandomGenerator& rng, Operon::Tree tree) const -> std::tuple<Operon::Tree, tl::expected<FitResult, FitFailure>> override;

private:
    gsl::not_null<Operon::OptimizerBase const*> optimizer_;
};

} // namespace Operon

#endif
