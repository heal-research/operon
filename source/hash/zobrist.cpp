// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/hash/zobrist.hpp"

#include <algorithm>
#include <gtl/phmap.hpp>

namespace Operon {

struct Zobrist::TranspositionTable {
    gtl::parallel_flat_hash_map<Operon::Hash, std::pair<Zobrist::Value, std::size_t>> Map;
};

Zobrist::Zobrist(Operon::RandomGenerator& rng, int maxLength)
    : table_(static_cast<int>(NodeTypes::Count), maxLength)
    , tt_(std::make_unique<TranspositionTable>())
{
    std::generate(table_.container().begin(), table_.container().end(), std::ref(rng));
}

Zobrist::~Zobrist() = default;

auto Zobrist::TryGet(Operon::Hash hash, Value& val) const -> bool
{
    bool found{false};
    tt_->Map.if_contains(hash, [&](auto const& v) {
        val = v.second.first;
        found = true;
    });
    if (found) { ++hits_; }
    return found;
}

auto Zobrist::Insert(Operon::Hash hash, Value const& val) -> void
{
    tt_->Map.lazy_emplace_l(
        hash,
        [](auto& v) { ++v.second.second; },          // already present: bump count
        [&](auto const& ctor) { ctor(hash, std::make_pair(val, std::size_t{1})); }  // new entry
    );
}

auto Zobrist::Clear() -> void
{
    tt_->Map.clear();
    hits_.store(0);
}

auto Zobrist::Size() const -> std::size_t
{
    return tt_->Map.size();
}

} // namespace Operon
