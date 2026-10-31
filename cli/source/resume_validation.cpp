// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "resume_validation.hpp"

#include <algorithm>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <stdexcept>
#include <utility>

#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/hash/hash.hpp"

namespace Operon {

auto HashBytes(void const* data, std::size_t size) -> std::uint64_t
{
    return Hasher {}(static_cast<std::uint8_t const*>(data), size);
}

auto HashText(std::string_view text) -> std::uint64_t { return Hasher {}(text); }

auto CombineHashes(std::uint64_t seed, std::uint64_t value) -> std::uint64_t
{
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

auto BuildIdentityHash(BuildInfo const& info) -> std::uint64_t
{
    std::uint64_t h = HashText("build");
    for (auto const* part : { &info.Revision, &info.Precision, &info.Arch, &info.IsaLevel }) {
        h = CombineHashes(h, HashText(*part));
    }
    return CombineHashes(h, sizeof(Scalar));
}

auto ShapeConstraintsHash(ShapeConstraintSet const* constraints) -> std::uint64_t
{
    if (constraints == nullptr) {
        return HashText("shape.constraints: none");
    }
    auto const scalar
        = [](std::uint64_t h, Scalar value) { return CombineHashes(h, HashBytes(&value, sizeof(value))); };

    std::uint64_t h = HashText("shape.constraints: set");
    // the domains of a hash map in a fixed order, so that the order of the keys in the file does not matter
    std::vector<std::pair<std::string, std::pair<Scalar, Scalar>>> domains(
        constraints->Domains.begin(), constraints->Domains.end());
    std::ranges::sort(domains, [](auto const& a, auto const& b) { return a.first < b.first; });
    h = CombineHashes(h, domains.size());
    for (auto const& [name, box] : domains) {
        h = CombineHashes(h, HashText(name));
        h = scalar(h, box.first);
        h = scalar(h, box.second);
    }
    h = CombineHashes(h, constraints->Constraints.size());
    for (auto const& constraint : constraints->Constraints) {
        h = CombineHashes(h, static_cast<std::uint64_t>(constraint.Op));
        h = CombineHashes(h, HashText(constraint.Variable));
        h = CombineHashes(h, constraint.Sign ? static_cast<std::uint64_t>(*constraint.Sign + 2) : 0ULL);
        if (constraint.Bound) {
            h = CombineHashes(h, 1);
            h = scalar(h, constraint.Bound->first);
            h = scalar(h, constraint.Bound->second);
        } else {
            h = CombineHashes(h, 0);
        }
    }
    return h;
}

auto FingerprintDifferences(Serialization::RunFingerprint const& saved, Serialization::RunFingerprint const& current)
    -> std::vector<std::string>
{
    auto const label = [](std::string const& entry) -> std::string {
        return entry.starts_with("option.") ? "--" + entry.substr(7) : entry;
    };
    std::vector<std::string> differences;
    for (std::size_t i = 0; i < current.Groups.size(); ++i) {
        auto const it = std::ranges::find(saved.Groups, current.Groups[i]);
        if (it == saved.Groups.end()) {
            differences.push_back(label(current.Groups[i]) + " (not recorded in the checkpoint)");
        } else if (saved.Hashes[static_cast<std::size_t>(it - saved.Groups.begin())] != current.Hashes[i]) {
            differences.push_back(label(current.Groups[i]));
        }
    }
    for (auto const& entry : saved.Groups) {
        if (std::ranges::find(current.Groups, entry) == current.Groups.end()) {
            differences.push_back(label(entry) + " (recorded in the checkpoint only)");
        }
    }
    return differences;
}

auto FingerprintMismatchMessage(std::string const& path, Serialization::RunFingerprint const& saved,
    Serialization::RunFingerprint const& current) -> std::string
{
    auto const differences = FingerprintDifferences(saved, current);
    if (differences.empty()) {
        return {};
    }
    if (std::ranges::find_if(differences, [](std::string const& d) { return d.starts_with(BuildIdentityGroup); })
        != differences.end()) {
        return fmt::format("checkpoint '{}' was written by another build (Operon revision, scalar precision, "
                           "architecture or x86-64 ISA level differ): resume with the build that wrote it",
            path);
    }
    return fmt::format("checkpoint '{}' was written by a run with different settings, differing: {}; resume with the "
                       "dataset and options of the checkpointed run (only --threads, --generations, --evaluations, "
                       "--timelimit, --seed without a cache, the data paths and what is only reported or written may "
                       "change)",
        path, fmt::join(differences, ", "));
}

auto CheckGenerationLimit(std::string const& path, std::uint64_t checkpointGeneration, std::size_t generationLimit)
    -> void
{
    if (checkpointGeneration > generationLimit) {
        throw std::runtime_error(fmt::format("checkpoint '{}' is at generation {} but --generations is {}: raise "
                                             "--generations to at least the generation of the checkpoint",
            path, checkpointGeneration, generationLimit));
    }
}

auto CheckCachePresence(std::string const& path, bool checkpointHasCache, bool runHasCache) -> void
{
    if (checkpointHasCache && !runHasCache) {
        throw std::runtime_error(
            fmt::format("checkpoint '{}' carries a transposition cache: resume with --transposition-cache and the "
                        "--cache-visibility, --cache-max-age and --seed of the checkpointed run",
                path));
    }
    if (!checkpointHasCache && runHasCache) {
        throw std::runtime_error(
            fmt::format("checkpoint '{}' has no transposition cache (it was written by a run without "
                        "--transposition-cache): resume without --transposition-cache, or start a new run",
                path));
    }
}

auto CheckPopulation(std::string const& path, Operon::Vector<Individual> const& population, Problem const& problem,
    std::size_t maxLength) -> void
{
    auto inputs = problem.GetInputs();
    std::ranges::sort(inputs);
    auto const& pset = problem.GetPrimitiveSet();

    for (std::size_t i = 0; i < population.size(); ++i) {
        auto const nodes = population[i].Genotype.Nodes();
        if (nodes.empty()) {
            throw std::runtime_error(fmt::format("checkpoint '{}': parent {} has an empty genotype", path, i));
        }
        if (nodes.size() > maxLength) {
            throw std::runtime_error(fmt::format(
                "checkpoint '{}': parent {} has {} nodes, more than --maxlength {}", path, i, nodes.size(), maxLength));
        }
        for (std::size_t j = 0; j < nodes.size(); ++j) {
            auto const& node = nodes[j];
            if (node.Type == NodeType::Variable && !std::ranges::binary_search(inputs, node.HashValue)) {
                throw std::runtime_error(fmt::format("checkpoint '{}': node {} of parent {} uses the variable with "
                                                     "hash {:#x}, which is not an input of this run (the checkpoint "
                                                     "was written with another dataset, --target or --inputs)",
                    path, j, i, node.HashValue));
            }
            if (node.Type == NodeType::Function
                && !(pset.Contains(node.HashValue) && pset.IsEnabled(node.HashValue)
                    && pset.Frequency(node.HashValue) > 0)) {
                throw std::runtime_error(fmt::format("checkpoint '{}': node {} of parent {} uses the function with "
                                                     "hash {:#x}, which is not an enabled primitive of this run (the "
                                                     "checkpoint was written with another primitive set)",
                    path, j, i, node.HashValue));
            }
        }
    }
}

} // namespace Operon
