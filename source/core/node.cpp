// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/node.hpp"

#include <iterator>       // for pair
#include <utility>        // for make_pair, pair
#include <string>         // for string

#include "operon/core/standard_library.hpp"
#include "operon/core/types.hpp"

using std::pair;
using std::string;

namespace Operon {
namespace {
    auto Descriptions() -> Operon::Map<Operon::Hash, pair<string, string>>&
    {
        static Operon::Map<Operon::Hash, pair<string, string>> desc; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
        return desc;
    }

    auto LookupHash(Node const& node) -> Operon::Hash
    {
        return node.IsVariable() ? Node(NodeType::Variable).HashValue : node.HashValue;
    }
} // namespace

    void Node::RegisterName(Operon::Hash hash, string name, string desc)
    {
        Descriptions()[hash] = { std::move(name), std::move(desc) };
    }

    auto Node::Name() const noexcept -> std::string const& // NOLINT(bugprone-exception-escape)
    {
        StandardLibrary::RegisterNames();
        return Descriptions().at(LookupHash(*this)).first;
    }

    auto Node::Desc() const noexcept -> std::string const& // NOLINT(bugprone-exception-escape)
    {
        StandardLibrary::RegisterNames();
        return Descriptions().at(LookupHash(*this)).second;
    }

} // namespace Operon
