// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_HIP_CONTEXT_HPP
#define OPERON_HIP_CONTEXT_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "operon/operon_export.hpp"
#include "operon/optimizer/population_encoding.hpp"
#include "operon/optimizer/population_local_search.hpp"
#include "operon/operators/population_scorer.hpp"




namespace Operon::PopulationOptimization::Hip {

enum class OptimizationStatus : uint8_t { Improved, Retained, InvalidInput, Unsupported };

struct OptimizationResult {
    std::vector<Operon::Scalar> Coefficients;
    std::vector<double> InitialCosts;
    std::vector<double> FinalCosts;
    std::vector<OptimizationStatus> Status;
    std::vector<uint32_t> Iterations;
    std::vector<uint32_t> AcceptedSteps;
};

// Test-only device snapshot of the fused normal-equation reduction. Layouts
// are [tree][16][16] for Normal and [tree][16] for Gradient; each tree's
// DeviceTreeRange determines the populated prefix.
struct NormalEquationDiagnostics {
    std::vector<Operon::Scalar> Normal;
    std::vector<Operon::Scalar> Gradient;
    std::vector<Operon::Scalar> Costs;
    std::vector<uint8_t> Valid;
};

// Opaque owner for a HIP stream and compact tree buffers. HIP headers remain
// private to its implementation; ordinary Operon consumers need no ROCm SDK.
class OPERON_EXPORT Context : public Operon::PopulationLocalSearchBackend {
public:
    Context();
    Context(Context const&) = delete;
    Context(Context&&) noexcept;
    auto operator=(Context const&) -> Context& = delete;
    auto operator=(Context&&) noexcept -> Context&;
    ~Context();

    // `variableHashes[column]` describes the column-major input binding used
    // by Evaluate. Upload rejects a tree whose Variable hash is not present.
    auto Upload(EncodedPopulation const&, std::span<Operon::Hash const> variableHashes) -> void;

    // Replaces only the resident flat coefficient buffer. Structural node,
    // tree-range, and packed bucket buffers uploaded by Upload remain intact.
    // This is the generation-level update boundary for unchanged topology.
    auto UpdateCoefficients(std::span<Operon::Scalar const> coefficients) -> void;

    // Inputs are contiguous [variable][row]. Coefficients replace only the
    // mutable flat coefficient buffer uploaded with the population.

    // Returns [flat coefficient][row], preserving EncodedPopulation's flat
    // coefficient order. The caller can form per-tree normal equations from
    // each TreeRange without a host-side tree walk.
    [[nodiscard]] auto JacRev(std::span<Operon::Scalar const> columns,
                              std::size_t variableCount,
                              std::size_t rowCount,
                              std::span<Operon::Scalar const> coefficients)
        -> std::vector<Operon::Scalar>;

    // Evaluates with the coefficient buffer currently resident on the device.
    [[nodiscard]] auto EvaluateResident(std::span<Operon::Scalar const> columns,
                                        std::size_t variableCount,
                                        std::size_t rowCount) -> std::vector<Operon::Scalar>;
    [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const> columns,
                                std::size_t variableCount,
                                std::size_t rowCount,
                                std::span<Operon::Scalar const> coefficients)
        -> std::vector<Operon::Scalar>;

    // Returns 0.5 * sum(weight * residual^2) per resident tree, with a
    // validity byte per tree. Predictions stay device-resident.
    [[nodiscard]] auto GaussianCosts(std::span<Operon::Scalar const> columns,
                                     std::size_t variableCount,
                                     std::size_t rowCount,
                                     std::span<Operon::Scalar const> target,
                                     std::span<Operon::Scalar const> weights = {})
        -> std::pair<std::vector<Operon::Scalar>, std::vector<uint8_t>>;


    // Runs all LM iterations on the device. Inputs are contiguous [variable][row];
    // targets and optional non-negative weights are row-local. Only final
    // coefficients and diagnostics cross back to the host.
    [[nodiscard]] auto OptimizeGaussian(std::span<Operon::Scalar const> columns,
                                        std::size_t variableCount,
                                        std::size_t rowCount,
                                        std::span<Operon::Scalar const> target,
                                        std::span<Operon::Scalar const> weights = {},
                                        uint32_t maxIterations = 16) -> OptimizationResult;

    // Test-only fused-reduction seam. This is intentionally confined to the
    // optional HIP context rather than the scalar optimizer API.
    [[nodiscard]] auto NormalEquations(std::span<Operon::Scalar const> columns,
                                       std::size_t variableCount,
                                       std::size_t rowCount,
                                       std::span<Operon::Scalar const> target,
                                       std::span<Operon::Scalar const> weights = {}) -> NormalEquationDiagnostics;


    [[nodiscard]] auto Supports(Operon::Tree const& tree, Operon::Span<Operon::Hash const> variableHashes) const -> bool final;
    [[nodiscard]] auto Optimize(EncodedPopulation const& population,
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

// Opt-in generation-level scorer for Evaluator<ScalarDispatch> with SSE,
// MSE, RMSE, or NMSE and no linear scaling/non-finite omission. It keeps the
// scalar API untouched and rejects every unsupported evaluator configuration.
class OPERON_EXPORT GaussianPopulationOffspringScorer final : public Operon::PopulationOffspringScorer {
public:
    explicit GaussianPopulationOffspringScorer(Context& context)
        : context_(context)
    {
    }

    void Score(std::span<Operon::Individual> candidates,
               std::span<Operon::RandomGenerator> random,
               Operon::EvaluatorBase const& evaluator,
               Operon::CoefficientOptimizer const* optimizer,
               double localSearchProbability,
               double lamarckianProbability,
               std::span<Operon::Vector<Operon::Scalar>> scratch) final;

private:
    Context& context_;
};

} // namespace Operon::PopulationOptimization::Hip

#endif // OPERON_HIP_CONTEXT_HPP
