// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_SYCL_CONTEXT_HPP
#define OPERON_SYCL_CONTEXT_HPP

#include <cstddef>
#include <cstdint>

#include "operon/operon_export.hpp"
#include "operon/optimizer/population_encoding.hpp"
#include "operon/optimizer/population_local_search.hpp"

namespace Operon::PopulationOptimization::Sycl {

// Optional AdaptiveCpp implementation of the shared population-local-search
// contract. It selects a GPU explicitly and uses an in-order queue with
// reusable device-USM allocations, so it never silently executes on a CPU.
// The supported lowered instruction subset and tree limits match HIP: Constant,
// scaled Variable, Ref, Add, Sub, Mul, Div; at most 128 nodes and 16
// coefficients per tree.
class OPERON_EXPORT Context final : public Operon::PopulationLocalSearchBackend {
public:
    Context();
    Context(Context const&) = delete;
    Context(Context&&) noexcept;
    auto operator=(Context const&) -> Context& = delete;
    auto operator=(Context&&) noexcept -> Context&;
    ~Context() override;
    // Uploads supported trees for subsequent batched Evaluate calls. Inputs
    // are column-major [variable][row]; output is [tree][row].
    auto Upload(EncodedPopulation const& population, Operon::Span<Operon::Hash const> variableHashes) -> void;
    [[nodiscard]] auto Evaluate(Operon::Span<Operon::Scalar const> columns,
                                std::size_t variableCount,
                                std::size_t rowCount,
                                Operon::Span<Operon::Scalar const> coefficients) -> std::vector<Operon::Scalar>;


    [[nodiscard]] auto Supports(Operon::Tree const& tree,
                                Operon::Span<Operon::Hash const> variableHashes) const -> bool final;

    [[nodiscard]] auto Optimize(
        EncodedPopulation const& population,
        Operon::Span<Operon::Hash const> variableHashes,
        Operon::Span<Operon::Scalar const> columns,
        std::size_t variableCount,
        std::size_t rowCount,
        Operon::Span<Operon::Scalar const> target,
        Operon::Span<Operon::Scalar const> weights,
        uint32_t maxIterations) -> Operon::PopulationLocalSearchResult final;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Operon::PopulationOptimization::Sycl

#endif // OPERON_SYCL_CONTEXT_HPP
