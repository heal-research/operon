// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/interpreter/range_tightening.hpp"

#include <bit>
#include <limits>
#include <memory>

#include "operon/core/tree_diff.hpp"

namespace Operon {

namespace {
    constexpr std::size_t NoGrad = std::numeric_limits<std::size_t>::max();
    using Scalar = Operon::Scalar;
    using Interval = IntervalEvaluator::Interval;

    // Mirrors Deriv()'s dispatch in tree_diff.cpp; must stay in sync with it.
    auto IsSymbolicallyDifferentiable(Node const& n) -> bool
    {
        // Add/Mul/Sub handle arbitrary arity in Deriv(); Div only arity 1-2
        // (Deriv() explicitly returns Zero for arity > 2); Pow is hardcoded
        // for exactly arity 2 (indexes children[1] unconditionally).
        if (n.IsAddition() || n.IsMultiplication() || n.IsSubtraction()) { return true; }
        if (n.IsDivision()) { return n.Arity <= 2; }
        if (n.IsPow()) { return n.Arity == 2; }
        if (n.IsAq() || n.IsPowabs() || n.IsOp<BuiltinOp::Fmin, BuiltinOp::Fmax>()) {
            return false;
        }
        if (n.Arity == 1) { return HasUnarySymbolicDeriv(n.HashValue); }
        if (n.Arity == 2) { return HasBinarySymbolicDeriv(n.HashValue); }
        // Deriv() itself falls through to Zero here (no hardcoded or
        // registered rule can apply to arity >= 3 beyond Add/Mul/Sub/Div
        // above) - sound-by-default, matching that.
        return false;
    }

    auto EvaluateGradientColumn(
        VariableGradientDag const& gdag, std::size_t root,
        IntervalEvaluator::DomainMap const& domains, Operon::Span<Operon::Scalar const> coeff
    ) -> Interval
    {
        Operon::Vector<Node> subnodes(
            gdag.Nodes.cbegin(), gdag.Nodes.cbegin() + static_cast<std::ptrdiff_t>(root) + 1);
        Tree const gradTree{std::move(subnodes)};
        return IntervalEvaluator(&gradTree, domains).Evaluate(coeff);
    }

    auto MixHash(std::uint64_t a, std::uint64_t b) -> std::uint64_t
    {
        return a ^ (b * 0x9e3779b97f4a7c15ULL + (a << 6U) + (a >> 2U));
    }
} // namespace

struct RangeCache::Entry {
    Interval Value;
};

RangeCache::RangeCache(Zobrist const& zobrist)
    : zobrist_(&zobrist)
    , cache_(std::make_unique<ZobristCache<Entry>>())
{ }

RangeCache::~RangeCache() = default;

auto RangeCache::ComputeKey(
    Tree const& tree, Operon::Span<Operon::Scalar const> coeff,
    IntervalEvaluator::DomainMap const& domains, Operon::Hash variant
) const -> Operon::Hash
{
    auto h = zobrist_->ComputeHash(tree);
    // Zobrist::ComputeHash intentionally excludes Node::Value (it hashes
    // structure/identity only). coeff below only carries Optimize==true
    // values, so a non-optimized node's baked-in Value (composed-function
    // constants, fixed variable/function weights) must be folded in here,
    // or two structurally-identical trees differing only in such a value
    // collide on the same key and return each other's stale interval.
    for (auto const& n : tree.Nodes()) {
        if (!n.Optimize) {
            h = MixHash(h, std::bit_cast<std::uint64_t>(static_cast<double>(n.Value)));
        }
    }
    for (auto v : coeff) {
        h = MixHash(h, std::bit_cast<std::uint64_t>(static_cast<double>(v)));
    }
    // Domain contribution is order-independent (XOR-folded per entry) since
    // DomainMap has no fixed iteration order.
    Operon::Hash domainMix{0};
    for (auto const& [varHash, dom] : domains) {
        auto const entry = MixHash(
            MixHash(varHash, std::bit_cast<std::uint64_t>(static_cast<double>(dom.first))),
            std::bit_cast<std::uint64_t>(static_cast<double>(dom.second)));
        domainMix ^= entry;
    }
    return MixHash(MixHash(h, domainMix), variant);
}

auto RangeCache::TryGet(
    Tree const& tree, Operon::Span<Operon::Scalar const> coeff,
    IntervalEvaluator::DomainMap const& domains, Interval& out, Operon::Hash variant
) const -> bool
{
    auto const key = ComputeKey(tree, coeff, domains, variant);
    return cache_->IfContains(key, [&](Entry const& e) { out = e.Value; });
}

auto RangeCache::Insert(
    Tree const& tree, Operon::Span<Operon::Scalar const> coeff,
    IntervalEvaluator::DomainMap const& domains, Interval const& val, Operon::Hash variant
) -> void
{
    auto const key = ComputeKey(tree, coeff, domains, variant);
    cache_->LazyEmplace(key,
        [](Entry&) { }, // first writer wins on a genuine race, same as the fitness cache
        [&](Entry& e) { e.Value = val; });
}

auto RangeCache::Size() const -> std::size_t { return cache_->Size(); }
auto RangeCache::Clear() -> void { cache_->Clear(); }

auto TightenRange(
    Tree const& tree,
    IntervalEvaluator::DomainMap const& domains,
    Operon::Span<Operon::Scalar const> coeff,
    RangeCache* cache
) -> Interval
{
    constexpr Operon::Hash flatVariant = 0;
    if (cache != nullptr) {
        Interval cached;
        if (cache->TryGet(tree, coeff, domains, cached, flatVariant)) { return cached; }
    }

    auto const finish = [&](Interval const& result) -> Interval {
        if (cache != nullptr) { cache->Insert(tree, coeff, domains, result, flatVariant); }
        return result;
    };

    auto const naive = IntervalEvaluator(&tree, domains).Evaluate(coeff);

    // A variable can occur multiple times with only some occurrences behind
    // an undifferentiated op (e.g. X + abs(X)); the root would then come
    // back nonzero but understate the true partial, so this is checked
    // structurally rather than via BuildVariableGradientDag's root value.
    for (auto const& n : tree.Nodes()) {
        if (n.IsLeaf() || n.IsRef()) { continue; }
        if (!IsSymbolicallyDifferentiable(n)) { return finish(naive); }
    }

    auto const gdag = BuildVariableGradientDag(tree, coeff);
    if (gdag.Variables.empty()) { return finish(naive); } // no input variables: naive is already exact

    // F(m) via a degenerate (lo == hi) domain map, reusing IntervalEvaluator.
    IntervalEvaluator::DomainMap midpoints;
    midpoints.reserve(domains.size());
    for (auto const& [hash, domain] : domains) {
        auto const m = Interval{domain.first, domain.second}.mid();
        midpoints.insert_or_assign(hash, IntervalEvaluator::Domain{m, m});
    }
    auto const fm = IntervalEvaluator(&tree, midpoints).Evaluate(coeff);

    auto meanValue = fm;
    for (std::size_t k = 0; k < gdag.Variables.size(); ++k) {
        auto const root = gdag.Roots[k];
        if (root == NoGrad) { continue; } // genuinely zero, given the pre-check above

        auto const hash = gdag.Variables[k];
        auto const dit  = domains.find(hash);
        if (dit == domains.end()) { return finish(naive); } // defensive: naive would already have thrown
        auto const& [lo, hi] = dit->second;
        auto const m = Interval{lo, hi}.mid();

        auto const gradInterval = EvaluateGradientColumn(gdag, root, domains, coeff);
        auto const xkMinusM = pappus::ops::sub<Scalar>(
            pappus::ops::variable<Scalar>(lo, hi), pappus::ops::constant<Scalar>(m));
        meanValue = pappus::ops::add<Scalar>(meanValue, pappus::ops::mul<Scalar>(gradInterval, xkMinusM));
    }

    if (meanValue.is_empty()) { return finish(naive); }
    return finish(naive & meanValue);
}

auto TightenRangeBisected(
    Tree const& tree,
    IntervalEvaluator::DomainMap domains,
    Operon::Span<Operon::Scalar const> coeff,
    int maxDepth,
    RangeCache* cache
) -> Interval
{
    // Distinguishes this call's own (tighter, depth-dependent) result from
    // TightenRange's flat one in the same cache - same (tree, coeff,
    // domains) key would otherwise collide across the two functions.
    auto const bisectedVariant = MixHash(0x62697365637465ULL /* "bisecte" */, static_cast<std::uint64_t>(maxDepth));
    if (cache != nullptr) {
        Interval cached;
        if (cache->TryGet(tree, coeff, domains, cached, bisectedVariant)) { return cached; }
    }

    auto const finish = [&](Interval const& result) -> Interval {
        if (cache != nullptr) { cache->Insert(tree, coeff, domains, result, bisectedVariant); }
        return result;
    };

    auto const result = TightenRange(tree, domains, coeff, cache);
    if (maxDepth <= 0 || result.is_empty()) { return finish(result); }

    auto const gdag = BuildVariableGradientDag(tree, coeff);
    if (gdag.Variables.empty()) { return finish(result); }

    // Pick the variable whose gradient interval straddles zero with the
    // largest diameter: it's both sign-ambiguous (mean-value form is
    // loosest there) and contributes the most to that ambiguity.
    Operon::Hash splitVar{};
    Scalar bestDiameter{0};
    bool found = false;
    for (std::size_t k = 0; k < gdag.Variables.size(); ++k) {
        auto const root = gdag.Roots[k];
        if (root == NoGrad) { continue; }
        auto const gradInterval = EvaluateGradientColumn(gdag, root, domains, coeff);
        if (!gradInterval.contains(Scalar{0})) { continue; }
        auto const d = gradInterval.diameter();
        if (d > bestDiameter) { bestDiameter = d; splitVar = gdag.Variables[k]; found = true; }
    }
    if (!found) { return finish(result); } // every gradient is sign-definite already

    auto const dit = domains.find(splitVar);
    if (dit == domains.end()) { return finish(result); }
    auto const [lo, hi] = dit->second;
    auto const mid = Interval{lo, hi}.mid();

    auto leftDomains = domains;
    leftDomains[splitVar] = {lo, mid};
    auto rightDomains = domains;
    rightDomains[splitVar] = {mid, hi};

    auto const left  = TightenRangeBisected(tree, leftDomains, coeff, maxDepth - 1, cache);
    auto const right = TightenRangeBisected(tree, rightDomains, coeff, maxDepth - 1, cache);
    auto const unioned = left | right;

    if (unioned.is_empty()) { return finish(result); }
    return finish(result & unioned);
}

} // namespace Operon
