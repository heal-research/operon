// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/hash/zobrist.hpp"

#include <algorithm>

namespace Operon {

struct Zobrist::TranspositionTable {
    ZobristCache<FitnessEntry> Cache;
};

Zobrist::Zobrist(Operon::RandomGenerator& rng, int maxLength, Operon::Span<Operon::Hash const> variableHashes)
    : table_(static_cast<int>(NodeTypes::Count) + static_cast<int>(variableHashes.size()) + 1, maxLength)
    , tt_(std::make_unique<TranspositionTable>())
{
    std::generate(table_.container().begin(), table_.container().end(), std::ref(rng));
    for (int i = 0; i < static_cast<int>(variableHashes.size()); ++i) {
        varIndex_[variableHashes[i]] = i;
    }
}

Zobrist::~Zobrist() = default;

auto Zobrist::TryGet(Operon::Hash hash, Value& val) const -> bool
{
    bool const found = tt_->Cache.IfContains(hash, [&](FitnessEntry const& e) { val = e.Value; });
    if (found) { ++hits_; }
    return found;
}

auto Zobrist::Insert(Operon::Hash hash, Value const& val) -> void
{
    tt_->Cache.LazyEmplace(hash,
        [](FitnessEntry& e) { ++e.Visits; },
        [&](FitnessEntry& e) { e.Value = val; }
    );
}

auto Zobrist::Clear() -> void
{
    tt_->Cache.Clear();
    hits_.store(0);
}

auto Zobrist::Size() const -> std::size_t
{
    return tt_->Cache.Size();
}

} // namespace Operon
