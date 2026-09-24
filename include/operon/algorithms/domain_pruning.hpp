// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_DOMAIN_PRUNING_HPP
#define OPERON_ALGORITHMS_DOMAIN_PRUNING_HPP

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Result of conservative domain analysis. Unknown is always retained by pruning.
enum class DomainStatus : uint8_t {
    Valid,
    Invalid,
    Unknown,
};

// Policy used to turn per-row evidence into a candidate decision.
enum class DomainPolicy : uint8_t {
    // Reject when any selected row is proven invalid and no row is unknown.
    // Mixed invalid and unknown evidence remains Unknown (and is retained).
    AllRowsFinite,
    // Reject only when every selected row is proven invalid; one proven finite
    // row is sufficient to establish that a finite result exists.
    NoFiniteRows,
};

// Non-owning immutable view of the selected dataset rows. The caller owns the
// dataset and must keep it alive for the duration of analysis/enumeration.
struct DomainContext {
    Dataset const* Data{};
    Range Rows{};

    DomainContext(Dataset const& data, Range rows)
        : Data(&data)
        , Rows(rows)
    {
    }
};

// Conservative, coefficient-independent analysis of a tree over selected rows.
// Optimize=true constants/weights are Unknown, never fixed to Node::Value.
[[nodiscard]] OPERON_EXPORT auto AnalyzeDomain(
    Tree const& tree, DomainContext const& context, DomainPolicy policy = DomainPolicy::AllRowsFinite) -> DomainStatus;

struct DomainPruningConfig {
    bool Enabled{false};
    DomainPolicy Policy{DomainPolicy::AllRowsFinite};
    DomainContext const* Context{};
};

} // namespace Operon

#endif


