// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#ifndef OPERON_CLI_RESOLVED_CONFIG_HPP
#define OPERON_CLI_RESOLVED_CONFIG_HPP

#include "operator_factory.hpp"
#include "operon/algorithms/ga_base.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/selector.hpp"

// The `operon.resolved_config` document written by `--print-config` and `--config-out PATH` (schema:
// docs/schemas/resolved-config.schema.json): the configuration the program resolved, with every default explicit.
namespace Operon::Cli {
struct ResolvedObjects {
    std::string_view Program;
    GeneticAlgorithmConfig const& Config;
    Problem const& ProblemRef;
    std::size_t Threads;
    CreatorBase const& Creator;
    std::size_t CreatorMaxLength;
    std::size_t CreatorMinDepth;
    std::size_t CreatorMaxDepth;
    SubtreeCrossover const& Crossover;
    Span<MutatorWeight const> Mutators;
    SelectorBase const& Female;
    SelectorBase const& Male;
    OffspringGeneratorBase const& Generator;
    ReinserterBase const& Reinserter;
    EvaluatorBase const& Evaluator;
    OptimizerBase const& Optimizer;
    std::optional<ShapeConstraintSpec> const& Shape;
    Span<char* const> Argv;
};
auto BuildResolvedConfig(cxxopts::ParseResult const& result, ResolvedObjects const& objects) -> std::string;
// Returns true for --print-config; --config-out also writes when both are requested.
auto ExportResolvedConfig(cxxopts::ParseResult const& result, ResolvedObjects const& objects) -> bool;
} // namespace Operon::Cli
#endif
