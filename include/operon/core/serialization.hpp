// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_SERIALIZATION_HPP
#define OPERON_SERIALIZATION_HPP

#include <span>
#include <string>
#include <string_view>

#include "operon/operon_export.hpp"
#include "operon/core/individual.hpp"

namespace Operon::Serialization {

OPERON_EXPORT auto ToJson(Tree const& tree) -> std::string;
OPERON_EXPORT auto ToJson(Individual const& individual) -> std::string;
OPERON_EXPORT auto ToJson(std::span<Individual const> front) -> std::string;

OPERON_EXPORT auto TreeFromJson(std::string_view json) -> Tree;
OPERON_EXPORT auto IndividualFromJson(std::string_view json) -> Individual;

} // namespace Operon::Serialization

#endif
