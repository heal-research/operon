// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_POPULATION_LOCAL_SEARCH_HPP
#define OPERON_POPULATION_LOCAL_SEARCH_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"
#include "operon/optimizer/population_encoding.hpp"

namespace Operon {

// A population backend returns coefficients in EncodedPopulation::Coefficients
// order. Costs are the trusted objective 0.5 * sum(weight * residual^2),
// computed before and after the candidate is delivered. An Improved result must
// have a finite FinalCost strictly below InitialCost; Retained keeps the input
// coefficients and reports FinalCost == InitialCost.
enum class PopulationLocalSearchStatus : uint8_t {
    Improved,
    Retained,
    InvalidInput,
    Unsupported,
};

struct PopulationLocalSearchResult {
    std::vector<Operon::Scalar> Coefficients;
    std::vector<double> InitialCosts;
    std::vector<double> FinalCosts;
    std::vector<PopulationLocalSearchStatus> Status;
    std::vector<uint32_t> Iterations;
    std::vector<uint32_t> AcceptedSteps;
};

// Optional accelerator interface for fitting a selected, heterogeneous
// population. Inputs are contiguous [variable][row]; weights are empty or one
// finite, non-negative value per row. Backends must reject unsupported trees
// before returning a candidate and must preserve selected-tree result order.
class OPERON_EXPORT PopulationLocalSearchBackend {
public:
    virtual ~PopulationLocalSearchBackend() = default;

    [[nodiscard]] virtual auto Supports(Operon::Tree const& tree,
                                        Operon::Span<Operon::Hash const> variableHashes) const -> bool = 0;

    [[nodiscard]] virtual auto Optimize(
        PopulationOptimization::EncodedPopulation const& population,
        Operon::Span<Operon::Hash const> variableHashes,
        Operon::Span<Operon::Scalar const> columns,
        std::size_t variableCount,
        std::size_t rowCount,
        Operon::Span<Operon::Scalar const> target,
        Operon::Span<Operon::Scalar const> weights,
        uint32_t maxIterations) -> PopulationLocalSearchResult = 0;
};

} // namespace Operon

#endif // OPERON_POPULATION_LOCAL_SEARCH_HPP
