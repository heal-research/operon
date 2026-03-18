// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/hash/zobrist.hpp"

#include <algorithm>
#include <gtl/phmap.hpp>

namespace Operon {

struct Zobrist::TranspositionTable {
    using Key   = Operon::Hash;
    using Value = std::pair<Operon::Individual, std::size_t>;
    gtl::parallel_flat_hash_map<Key, Value> map;
};

Zobrist::Zobrist(Operon::RandomGenerator& rng, int maxLength)
    : table_(static_cast<int>(NodeTypes::Count), maxLength)
    , tt_(std::make_unique<TranspositionTable>())
{
    std::generate(table_.container().begin(), table_.container().end(), std::ref(rng));
}

Zobrist::~Zobrist() = default;

auto Zobrist::TryGet(Operon::Hash hash, Operon::Individual& ind) const -> bool
{
    bool found{false};
    tt_->map.if_contains(hash, [&](auto const& v) {
        ind = v.second.first;
        found = true;
    });
    if (found) { ++hits_; }
    return found;
}

auto Zobrist::Insert(Operon::Hash hash, Operon::Individual const& ind) -> void
{
    tt_->map.lazy_emplace_l(
        hash,
        [](auto& v) { ++v.second.second; },          // already present: bump count
        [&](auto const& ctor) { ctor(hash, std::make_pair(ind, std::size_t{1})); }  // new entry
    );
}

auto Zobrist::Clear() -> void
{
    tt_->map.clear();
    hits_.store(0);
}

auto Zobrist::Size() const -> std::size_t
{
    return tt_->map.size();
}

} // namespace Operon
