// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_RANGE_TIGHTENING_HPP
#define OPERON_RANGE_TIGHTENING_HPP

#include <gsl/pointers>

#include <memory>

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Cache for TightenRange/TightenRangeBisected results, keyed by structural
// hash (via a caller-supplied Zobrist instance, reusing its hash table) mixed
// with the actual coeff values and domain bounds. Unlike the fitness cache
// (deliberately coefficient-independent, since a fitness value is reusable
// across any structurally-identical individual with the same trainable
// parameter *count*), an interval result genuinely depends on the exact
// coefficient *values* and the exact domain box - both are folded into the
// key so a cache hit is always for the exact same computation, never a
// coefficient-stale substitute. That's a correctness requirement here, not
// just a quality one: TightenRange's entire value proposition is being a
// *sound* bound, so a stale cache hit would silently reintroduce
// unsoundness. Thread-safe (backed by the same gtl concurrent map the
// fitness cache uses).
class OPERON_EXPORT RangeCache {
public:
    using Interval = IntervalEvaluator::Interval;

    explicit RangeCache(Zobrist const& zobrist);
    ~RangeCache();
    RangeCache(RangeCache const&) = delete;
    RangeCache(RangeCache&&) = delete;
    auto operator=(RangeCache const&) -> RangeCache& = delete;
    auto operator=(RangeCache&&) -> RangeCache& = delete;

    // `variant` distinguishes cache spaces that would otherwise collide on
    // the same (tree, coeff, domains) key but must not share entries - e.g.
    // TightenRange's flat result vs. TightenRangeBisected's tighter one at a
    // given remaining depth. Internal to TightenRange/TightenRangeBisected;
    // callers normally don't need to pass this.
    [[nodiscard]] auto TryGet(
        Tree const& tree, Operon::Span<Operon::Scalar const> coeff,
        IntervalEvaluator::DomainMap const& domains, Interval& out, Operon::Hash variant = 0
    ) const -> bool;

    auto Insert(
        Tree const& tree, Operon::Span<Operon::Scalar const> coeff,
        IntervalEvaluator::DomainMap const& domains, Interval const& val, Operon::Hash variant = 0
    ) -> void;

    [[nodiscard]] auto Size() const -> std::size_t;
    auto Clear() -> void;

private:
    struct Entry;

    [[nodiscard]] auto ComputeKey(
        Tree const& tree, Operon::Span<Operon::Scalar const> coeff,
        IntervalEvaluator::DomainMap const& domains, Operon::Hash variant
    ) const -> Operon::Hash;

    gsl::not_null<Zobrist const*> zobrist_;
    std::unique_ptr<ZobristCache<Entry>> cache_;
};

// Mean-value-form (first-order Taylor) enclosure of a tree's output range,
// intersected with IntervalEvaluator's naive enclosure of the same tree:
//
//   F([a,b]) ⊆ F(m) + ∇F([a,b]) · ([a,b] − m)
//
// `coeff` follows the same convention as IntervalEvaluator::Evaluate: one
// entry per node with Node::Optimize == true, consumed in node order.
//
// Falls back to the naive enclosure alone (no intersection) if the tree
// contains any op Deriv() can't symbolically differentiate, or if a
// gradient column evaluates to IntervalEvaluator::Interval's empty(). If
// naive itself is empty, it stays empty.
//
// `cache`, if supplied, is consulted first and populated on a miss.
OPERON_EXPORT auto TightenRange(
    Tree const& tree,
    IntervalEvaluator::DomainMap const& domains,
    Operon::Span<Operon::Scalar const> coeff,
    RangeCache* cache = nullptr
) -> IntervalEvaluator::Interval;

// Prototype: recursively bisects the domain on the variable whose gradient
// interval straddles zero the most (sign-ambiguous, where TightenRange is
// loosest), taking the union of both sub-box results, intersected with the
// whole-box TightenRange result. Never less sound, and never worse than
// TightenRange alone; stops when maxDepth is reached or no variable's
// gradient is sign-ambiguous.
//
// `cache`, if supplied, is consulted/populated both for this call's own
// result and (via the same cache) for every internal TightenRange call the
// recursion makes - the latter is often the bigger win, since different
// individuals sharing the same tree/coeff/domains recurse through identical
// sub-box sequences.
OPERON_EXPORT auto TightenRangeBisected(
    Tree const& tree,
    IntervalEvaluator::DomainMap domains,
    Operon::Span<Operon::Scalar const> coeff,
    int maxDepth = 4,
    RangeCache* cache = nullptr
) -> IntervalEvaluator::Interval;

} // namespace Operon

#endif
