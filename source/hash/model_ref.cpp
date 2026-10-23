// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/hash/model_ref.hpp"

#include <fmt/format.h>

#include "operon/core/node.hpp"
#include "operon/hash/sha256.hpp"

namespace Operon {

namespace {
    constexpr auto TypeLetter(NodeType type) -> char
    {
        switch (type) {
        case NodeType::Constant:
            return 'C';
        case NodeType::Variable:
            return 'V';
        case NodeType::Ref:
            return 'R';
        case NodeType::Function:
            return 'F';
        }
        return '?';
    }
} // namespace

auto ModelSerialization(Operon::Tree const& tree) -> std::string
{
    auto const& nodes = tree.Nodes();
    std::string out;
    out.reserve(32 + (nodes.size() * 48));
    fmt::format_to(std::back_inserter(out), "operon.model.v1\nnodes {}\n", nodes.size());
    for (auto const& node : nodes) {
        fmt::format_to(std::back_inserter(out), "{} {:x} {} {} {} {} {}\n", TypeLetter(node.Type),
            static_cast<std::uint64_t>(node.HashValue), node.Arity, node.IsEnabled ? 1 : 0, node.Optimize ? 1 : 0,
            node.IsRef() ? node.RefTo : 0, static_cast<double>(node.Value));
    }
    return out;
}

auto ModelRef(Operon::Tree const& tree) -> std::string { return Sha256Hex(ModelSerialization(tree)); }

} // namespace Operon
