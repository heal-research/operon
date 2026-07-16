// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/hash/zobrist.hpp"

#include <algorithm>
#include <utility>

namespace Operon {

struct Zobrist::TranspositionTable {
    ZobristCache<FitnessEntry> Cache;
};

Zobrist::Zobrist(Operon::RandomGenerator& rng, int maxLength, Operon::Span<Operon::Hash const> variableHashes)
    : table_(static_cast<int>(variableHashes.size()) + 1, maxLength)
    , tt_(std::make_unique<TranspositionTable>())
{
    std::generate(table_.container().begin(), table_.container().end(), std::ref(rng));
    for (int i = 0; std::cmp_less(i, variableHashes.size()); ++i) {
        varIndex_[variableHashes[i]] = i;
    }
}

Zobrist::~Zobrist() = default;

auto Zobrist::TryGet(Operon::Hash hash, Value& val) const -> bool
{
    // relaxed: these are statistics counters with no ordering requirement
    // on anything else, and TryGet is in the per-individual evaluation hot
    // path - the default seq_cst RMW would tax every lookup for no benefit.
    lookups_.fetch_add(1, std::memory_order_relaxed);
    bool const found = tt_->Cache.IfContains(hash, [&](FitnessEntry const& e) -> void { val = e.Value; });
    if (found) { hits_.fetch_add(1, std::memory_order_relaxed); }
    return found;
}

auto Zobrist::Insert(Operon::Hash hash, Value const& val) -> void
{
    tt_->Cache.LazyEmplace(hash,
        [](FitnessEntry& e) -> void { ++e.Visits; },
        [&](FitnessEntry& e) -> void { e.Value = val; }
    );
}

auto Zobrist::Clear() -> void
{
    tt_->Cache.Clear();
    hits_.store(0, std::memory_order_relaxed);
    lookups_.store(0, std::memory_order_relaxed);
}

auto Zobrist::Size() const -> std::size_t
{
    return tt_->Cache.Size();
}

} // namespace Operon
