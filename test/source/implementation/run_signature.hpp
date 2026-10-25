// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_TEST_RUN_SIGNATURE_HPP
#define OPERON_TEST_RUN_SIGNATURE_HPP

#include <bit>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"

// Bit-for-bit signatures of the committed state of a run, for the tests that compare runs at different thread counts.
namespace Operon::Test {

template <typename T> inline auto Bits(T value) -> std::uint64_t
{
    using Int = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    return std::bit_cast<Int>(value);
}

// Every field of Node, floating point values bit for bit. (Node::operator== compares the hash only.) Keep in step
// with Node; the signature test below changes each field in turn.
inline void Append(std::vector<std::uint64_t>& sig, Node const& node)
{
    sig.push_back(node.HashValue);
    sig.push_back(node.CalculatedHashValue);
    sig.push_back(Bits(node.Value));
    sig.push_back(node.Arity);
    sig.push_back(node.Length);
    sig.push_back(node.Depth);
    sig.push_back(node.Level);
    sig.push_back(node.Parent);
    sig.push_back(static_cast<std::uint64_t>(node.Type));
    sig.push_back(static_cast<std::uint64_t>(node.IsEnabled));
    sig.push_back(static_cast<std::uint64_t>(node.Optimize));
    sig.push_back(node.RefTo);
}

inline auto Signature(Operon::Span<Individual const> parents) -> std::vector<std::uint64_t>
{
    std::vector<std::uint64_t> sig;
    for (auto const& ind : parents) {
        for (auto const& node : ind.Genotype.Nodes()) {
            Append(sig, node);
        }
        sig.push_back(~std::uint64_t { 0 }); // separates the individuals
        for (auto f : ind.Fitness) {
            sig.push_back(Bits(f));
        }
        sig.push_back(ind.Rank);
        sig.push_back(Bits(ind.Distance));
    }
    return sig;
}

inline auto Signature(std::vector<Operon::RandomGenerator> const& rngs) -> std::vector<std::uint64_t>
{
    std::vector<std::uint64_t> sig;
    for (auto const& rng : rngs) {
        for (auto const word : rng.state()) {
            sig.push_back(word);
        }
    }
    return sig;
}

} // namespace Operon::Test

#endif
