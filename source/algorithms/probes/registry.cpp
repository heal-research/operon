// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/probes/registry.hpp"

#include <utility>

namespace Operon {

auto ProbeRegistry::Register(std::string type, ProbeFactory factory) -> void
{
    factories_.insert_or_assign(std::move(type), std::move(factory));
}

auto ProbeRegistry::Contains(std::string const& type) const -> bool
{
    return factories_.contains(type);
}

auto ProbeRegistry::Create(std::string const& type, ProbeParams const& params) const -> std::unique_ptr<GenerationProbe>
{
    auto it = factories_.find(type);
    if (it == factories_.end()) { return nullptr; }
    return it->second(params);
}

} // namespace Operon
