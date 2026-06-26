// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_SERIALIZATION_HPP
#define OPERON_SERIALIZATION_HPP

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "operon/operon_export.hpp"
#include "operon/core/individual.hpp"

namespace Operon::Serialization {

// ---- JSON ----

OPERON_EXPORT auto ToJson(Tree const& tree) -> std::string;
OPERON_EXPORT auto ToJson(Individual const& individual) -> std::string;
OPERON_EXPORT auto ToJson(std::span<Individual const> front) -> std::string;

OPERON_EXPORT auto TreeFromJson(std::string_view json) -> Tree;
OPERON_EXPORT auto IndividualFromJson(std::string_view json) -> Individual;

// ---- BEVE (binary) ----

OPERON_EXPORT auto ToBeve(Tree const& tree) -> std::string;
OPERON_EXPORT auto ToBeve(Individual const& individual) -> std::string;
OPERON_EXPORT auto ToBeve(std::span<Individual const> front) -> std::string;

OPERON_EXPORT auto TreeFromBeve(std::string_view data) -> Tree;
OPERON_EXPORT auto IndividualFromBeve(std::string_view data) -> Individual;

// ---- Checkpoint (algorithm save / resume) ----

struct OPERON_EXPORT Checkpoint {
    std::array<uint64_t, 4>                   RngState{};
    uint64_t                                  Generation{0};
    Operon::Vector<Individual>                Population;
    std::vector<std::array<uint64_t, 4>>      WorkerRngStates;
};

OPERON_EXPORT auto ToBeve(Checkpoint const&) -> std::string;
OPERON_EXPORT auto CheckpointFromBeve(std::string_view data) -> Checkpoint;

OPERON_EXPORT auto SaveCheckpoint(Checkpoint const&, std::string_view path) -> void;
OPERON_EXPORT auto LoadCheckpoint(std::string_view path) -> Checkpoint;

} // namespace Operon::Serialization

#endif
