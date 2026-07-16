// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/node.hpp"

#include <gtl/phmap.hpp>

#include <stdexcept>      // for invalid_argument, out_of_range
#include <utility>        // for make_pair, pair, move
#include <string>         // for string

#include "operon/core/standard_library.hpp"
#include "operon/core/types.hpp"

using std::pair;
using std::string;

namespace Operon {
namespace {
    // Thread-safe: Name()/Desc() are read on every tree format/print, which
    // can happen concurrently with RegisterName() calls registering
    // user-defined Dynamic functions during setup.
    auto Descriptions() -> gtl::parallel_flat_hash_map_m<Operon::Hash, pair<string, string>>&
    {
        static gtl::parallel_flat_hash_map_m<Operon::Hash, pair<string, string>> desc; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
        return desc;
    }

    auto LookupHash(Node const& node) -> Operon::Hash
    {
        return node.IsVariable() ? Node(NodeType::Variable).HashValue : node.HashValue;
    }

    // Returns by value: a reference into the map would not be safe to hand
    // back once the lookup's internal lock (gtl::parallel_flat_hash_map_m
    // shards its locking per bucket) has been released.
    auto LookupDescription(Node const& node) -> pair<string, string>
    {
        auto& d = Descriptions();
        auto const h = LookupHash(node);
        pair<string, string> result;
        if (d.if_contains(h, [&](auto const& kv) { result = kv.second; })) {
            return result;
        }
        auto const fallback = node.IsDynamic() ? Node(NodeType::Dynamic).HashValue : h;
        if (d.if_contains(fallback, [&](auto const& kv) { result = kv.second; })) {
            return result;
        }
        throw std::out_of_range("Node: no description registered for this hash");
    }
} // namespace

    void Node::RegisterName(Operon::Hash hash, string name, string desc)
    {
        // Also used internally by StandardLibrary::RegisterNames() to seed
        // built-in entries, whose hashes ARE in [0, NodeTypes::Count) by
        // construction (Node(NodeType) ctor) - so this can't reject that
        // range itself. User-facing callers (symbol_library.hpp) guard
        // against landing in the reserved range before calling this.
        Descriptions().lazy_emplace_l(
            hash,
            [&](auto& kv)         { kv.second = { std::move(name), std::move(desc) }; },
            [&](auto const& ctor) { ctor(hash, std::make_pair(std::move(name), std::move(desc))); }
        );
    }

    auto Node::Name() const -> std::string
    {
        StandardLibrary::RegisterNames();
        return LookupDescription(*this).first;
    }

    auto Node::Desc() const -> std::string
    {
        StandardLibrary::RegisterNames();
        return LookupDescription(*this).second;
    }

} // namespace Operon
