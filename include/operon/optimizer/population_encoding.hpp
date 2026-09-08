// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_POPULATION_ENCODING_HPP
#define OPERON_POPULATION_ENCODING_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <tl/expected.hpp>

#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon::PopulationOptimization {

// Stable, backend-neutral tree representation for population coefficient
// optimization. Device backends lower Opcode separately: `NodeType` and a
// Function's hash are preserved here so the host can reject unsupported
// primitives without baking a device ABI into Operon's core node enum.
struct EncodedNode {
    Operon::Hash HashValue{};
    Operon::Scalar Value{};
    uint32_t RefTo{};
    uint16_t Arity{};
    uint16_t Length{};
    NodeType Type{};
    bool IsEnabled{};
    bool Optimize{};
};

struct TreeRange {
    uint32_t NodeOffset{};
    uint32_t NodeCount{};
    uint32_t CoefficientOffset{};
    uint32_t CoefficientCount{};
    uint32_t OriginalIndex{};
};

struct EncodedPopulation {
    std::vector<EncodedNode> Nodes;
    std::vector<TreeRange> Trees;
    std::vector<Operon::Scalar> Coefficients;
};

enum class EncodingError : uint8_t {
    InvalidTree,
    NodeBufferTooLarge,
    CoefficientBufferTooLarge,
    IndividualIndexTooLarge,
};

// Encodes the selected individuals in selection order. OriginalIndex remains
// the index in `population`, so a backend may bucket Trees without changing
// the caller-visible result order. Structural instructions never contain the
// mutable coefficient values used by optimization; those live only in the
// flat Coefficients buffer.
[[nodiscard]] inline auto EncodePopulation(
    Operon::Span<Operon::Individual const> population,
    Operon::Span<std::size_t const> selected) -> tl::expected<EncodedPopulation, EncodingError>
{
    EncodedPopulation encoded;
    encoded.Trees.reserve(selected.size());

    for (auto const originalIndex : selected) {
        if (originalIndex >= population.size() || originalIndex > UINT32_MAX) {
            return tl::make_unexpected(EncodingError::IndividualIndexTooLarge);
        }

        auto const& tree = population[originalIndex].Genotype;
        if (!tree.Validate()) { return tl::make_unexpected(EncodingError::InvalidTree); }

        auto const& source = tree.Nodes();
        auto const coefficients = tree.GetCoefficients();
        if (source.size() > UINT32_MAX - encoded.Nodes.size()) {
            return tl::make_unexpected(EncodingError::NodeBufferTooLarge);
        }
        if (coefficients.size() > UINT32_MAX - encoded.Coefficients.size()) {
            return tl::make_unexpected(EncodingError::CoefficientBufferTooLarge);
        }

        encoded.Trees.push_back(TreeRange{
            .NodeOffset = static_cast<uint32_t>(encoded.Nodes.size()),
            .NodeCount = static_cast<uint32_t>(source.size()),
            .CoefficientOffset = static_cast<uint32_t>(encoded.Coefficients.size()),
            .CoefficientCount = static_cast<uint32_t>(coefficients.size()),
            .OriginalIndex = static_cast<uint32_t>(originalIndex),
        });

        encoded.Nodes.reserve(encoded.Nodes.size() + source.size());
        for (auto const& node : source) {
            encoded.Nodes.push_back(EncodedNode{
                .HashValue = node.HashValue,
                .Value = node.Optimize ? Operon::Scalar{} : node.Value,
                .RefTo = node.RefTo,
                .Arity = node.Arity,
                .Length = node.Length,
                .Type = node.Type,
                .IsEnabled = node.IsEnabled,
                .Optimize = node.Optimize,
            });
        }
        encoded.Coefficients.insert(encoded.Coefficients.end(), coefficients.begin(), coefficients.end());
    }

    return encoded;
}

// Reconstructs a Tree from one encoded range. This is deliberately a CPU
// oracle, not a device evaluator: interpreter evaluation/JacRev over the
// decoded tree verifies that lowering preserves current Operon semantics
// (including Ref and coefficient order) before a GPU opcode ABI exists.
[[nodiscard]] inline auto DecodeTree(EncodedPopulation const& encoded, std::size_t treeIndex) -> tl::expected<Operon::Tree, EncodingError>
{
    if (treeIndex >= encoded.Trees.size()) { return tl::make_unexpected(EncodingError::InvalidTree); }
    auto const range = encoded.Trees[treeIndex];
    auto const nodeEnd = static_cast<std::size_t>(range.NodeOffset) + range.NodeCount;
    auto const coefficientEnd = static_cast<std::size_t>(range.CoefficientOffset) + range.CoefficientCount;
    if (nodeEnd > encoded.Nodes.size() || coefficientEnd > encoded.Coefficients.size()) {
        return tl::make_unexpected(EncodingError::InvalidTree);
    }

    Operon::Vector<Operon::Node> nodes;
    nodes.reserve(range.NodeCount);
    for (auto const& encodedNode : std::span{encoded.Nodes}.subspan(range.NodeOffset, range.NodeCount)) {
        if (encodedNode.Type > NodeType::Function) { return tl::make_unexpected(EncodingError::InvalidTree); }
        auto node = Operon::Node{encodedNode.Type, encodedNode.HashValue};
        node.Value = encodedNode.Value;
        node.Length = encodedNode.Length;
        node.Arity = encodedNode.Arity;
        node.RefTo = static_cast<uint16_t>(encodedNode.RefTo);
        node.IsEnabled = encodedNode.IsEnabled;
        node.Optimize = encodedNode.Optimize;
        nodes.push_back(node);
    }

    auto tree = Operon::Tree{std::move(nodes)}.UpdateNodes();
    if (!tree.Validate() || tree.CoefficientsCount() != range.CoefficientCount) {
        return tl::make_unexpected(EncodingError::InvalidTree);
    }
    tree.SetCoefficients(std::span{encoded.Coefficients}.subspan(range.CoefficientOffset, range.CoefficientCount));
    return tree;
}

} // namespace Operon::PopulationOptimization

#endif // OPERON_POPULATION_ENCODING_HPP
