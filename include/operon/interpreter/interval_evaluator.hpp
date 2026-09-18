// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERVAL_EVALUATOR_HPP
#define OPERON_INTERVAL_EVALUATOR_HPP

#include <fmt/format.h>
#include <functional>
#include <gsl/pointers>
#include <stdexcept>
#include <tl/expected.hpp>

#include <utility>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/hash_registry.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/core/postorder_evaluator.hpp"
#include "operon/operon_export.hpp"

#include <pappus/pappus.hpp>

namespace Operon {

// Interval callback registries for unary/binary built-in and user-defined
// functions, keyed by Node::HashValue. Definitions live in
// interval_evaluator.cpp, explicitly instantiated per T, so all shared
// libraries share one registry instance.
template<typename T> using IntervalUnaryFn  = std::function<pappus::interval<T>(pappus::interval<T> const&)>;
template<typename T> using IntervalBinaryFn = std::function<pappus::interval<T>(pappus::interval<T> const&, pappus::interval<T> const&)>;

template<typename T> using IntervalUnaryRegistry  = HashRegistry<IntervalUnaryFn<T>>;
template<typename T> using IntervalBinaryRegistry = HashRegistry<IntervalBinaryFn<T>>;

// Direct registry access, mainly for tests. Prefer
// RegisterUnaryInterval/RegisterBinaryInterval to register a rule -- calling
// .Register() here skips built-in lazy-init, so a colliding hash is
// silently accepted instead of throwing. Explicitly instantiated per T in
// interval_evaluator.cpp so all shared libraries share one registry.
template<typename T> auto IntervalUnaryRules() -> IntervalUnaryRegistry<T>&;
template<typename T> auto IntervalBinaryRules() -> IntervalBinaryRegistry<T>&;

// Registers the built-in interval rules exactly once. Free function (not a
// member) so RegisterUnaryInterval/RegisterBinaryInterval can call it
// before writing, so a hash colliding with a built-in throws at the
// caller's own registration site instead of later inside Evaluate().
template<typename T> void RegisterIntervalBuiltins();

// Registers an interval callback for a unary function (built-in or
// user-defined). Throws if `hash` is already registered, including a
// collision with a built-in.
template<typename T> void RegisterUnaryInterval(Operon::Hash hash, IntervalUnaryFn<T> fn);

// Registers an interval callback for a binary function. See
// RegisterUnaryInterval.
template<typename T> void RegisterBinaryInterval(Operon::Hash hash, IntervalBinaryFn<T> fn);

// Whether an interval callback is registered for `hash`, forcing built-in
// registration first.
template<typename T> auto HasUnaryInterval(Operon::Hash hash) -> bool;
template<typename T> auto HasBinaryInterval(Operon::Hash hash) -> bool;

extern template auto IntervalUnaryRules<Operon::Scalar>() -> IntervalUnaryRegistry<Operon::Scalar>&;
extern template auto IntervalBinaryRules<Operon::Scalar>() -> IntervalBinaryRegistry<Operon::Scalar>&;
extern template void RegisterIntervalBuiltins<Operon::Scalar>();
extern template void RegisterUnaryInterval<Operon::Scalar>(Operon::Hash, IntervalUnaryFn<Operon::Scalar>);
extern template void RegisterBinaryInterval<Operon::Scalar>(Operon::Hash, IntervalBinaryFn<Operon::Scalar>);
extern template auto HasUnaryInterval<Operon::Scalar>(Operon::Hash) -> bool;
extern template auto HasBinaryInterval<Operon::Scalar>(Operon::Hash) -> bool;

extern template auto IntervalUnaryRules<eve::wide<Operon::Scalar>>() -> IntervalUnaryRegistry<eve::wide<Operon::Scalar>>&;
extern template auto IntervalBinaryRules<eve::wide<Operon::Scalar>>() -> IntervalBinaryRegistry<eve::wide<Operon::Scalar>>&;
extern template void RegisterIntervalBuiltins<eve::wide<Operon::Scalar>>();

namespace detail {
template<typename T>
struct IntervalPostOrderPolicy {
    using Scalar = T;
    using Value = pappus::interval<Scalar>;
    struct Context {};

    static void RegisterBuiltins() { RegisterIntervalBuiltins<Scalar>(); }
    static auto UnaryRules() -> IntervalUnaryRegistry<Scalar> const& { return IntervalUnaryRules<Scalar>(); }
    static auto BinaryRules() -> IntervalBinaryRegistry<Scalar> const& { return IntervalBinaryRules<Scalar>(); }
    static auto EmptyTree() -> std::string { return "IntervalEvaluator: empty tree"; }
    static auto MissingNode(Node const& node) -> std::string { return fmt::format("IntervalEvaluator: node kind `{}` not yet mapped", node.Name()); }

    static auto MakeConstant(Context const&, Scalar value) -> Value { return pappus::ops::constant<Scalar>(value); }
    static auto Add(Context const&, Value const& lhs, Value const& rhs) -> Value { return pappus::ops::add<Scalar>(lhs, rhs); }
    static auto Mul(Context const&, Value const& lhs, Value const& rhs) -> Value { return pappus::ops::mul<Scalar>(lhs, rhs); }
    static auto Sub(Context const&, Value const& lhs, Value const& rhs) -> Value { return pappus::ops::sub<Scalar>(lhs, rhs); }
    static auto Div(Context const&, Value const& lhs, Value const& rhs) -> Value { return pappus::ops::div<Scalar>(lhs, rhs); }
    static auto Min(Context const&, Value const& lhs, Value const& rhs) -> Value { return pappus::ops::min<Scalar>(lhs, rhs); }
    static auto Max(Context const&, Value const& lhs, Value const& rhs) -> Value { return pappus::ops::max<Scalar>(lhs, rhs); }
    static auto Neg(Context const&, Value const& value) -> Value { return pappus::ops::neg<Scalar>(value); }
    static auto Inv(Context const&, Value const& value) -> Value { return pappus::ops::inv<Scalar>(value); }
    static auto CallUnary(Context const&, IntervalUnaryFn<Scalar> const& function, Value const& value) -> Value { return function(value); }
    static auto CallBinary(Context const&, IntervalBinaryFn<Scalar> const& function, Value const& lhs, Value const& rhs) -> Value { return function(lhs, rhs); }
    static auto Scale(Value value, Scalar scale) -> Value { return value * scale; }
};
} // namespace detail

// Forward interval bounds for a tree over a single input domain. Walks the
// tree post-order, computing a `pappus::interval<T>` per node; variables
// are bound via `domains`, constants/weights via `coeff`.
//
// T defaults to `Operon::Scalar`. The `eve::wide<Operon::Scalar>`
// instantiation supports built-ins only (SIMD bisection); user-registered
// rules require the scalar evaluator.
//
// Domain errors (e.g. log of a negative interval) return `interval::empty()`
// rather than throwing; callers must check `result.is_empty()`.
template<typename T = Operon::Scalar>
class IntervalEvaluator {
public:
    using Scalar = T;
    using Interval = pappus::interval<Scalar>;
    // (lower, upper) bound for a variable, always Operon::Scalar-typed
    // regardless of T.
    using Domain = std::pair<Operon::Scalar, Operon::Scalar>;

    using DomainMap = Operon::Map<Operon::Hash, Domain>;

    // Compiles hash-keyed bounds into slots indexed by Tree::Nodes().
    IntervalEvaluator(gsl::not_null<Operon::Tree const*> tree, DomainMap const& domains)
        : tree_(tree)
        , domains_(domains)
    {
        RebuildDomainSlots();
    }

    [[nodiscard]] auto GetTree() const noexcept -> Operon::Tree const* { return tree_.get(); }
    [[nodiscard]] auto Domains() const noexcept -> DomainMap const& { return domains_; }

    // Retargets this evaluator at a different tree, reusing `domains_` (the
    // same map passed at construction -- every constraint in a
    // ShapeConstraintSet shares one domain box, only the tree differs
    // between Identity and a derivative constraint's sliced tree) and this
    // object's already-grown domainSlots_/primal_ vector capacity. Mirrors
    // AffineEvaluator::SetTree/Domains -- the interval-only bound path
    // previously built a fresh IntervalEvaluator (and its domainSlots_
    // allocation) per constraint per individual; profiling showed that
    // allocation churn as a real, non-trivial share of shape-constrained
    // Measure() cost.
    void SetTree(gsl::not_null<Operon::Tree const*> tree)
    {
        tree_ = tree;
        RebuildDomainSlots();
    }

    // Non-owning lane bounds for one variable. Lo and Hi must remain valid through
    // TryEvaluate; evaluators copy their values and never retain pointers.
    struct LaneOverride {
        Operon::Hash Hash;
        Operon::Scalar const* Lo;
        Operon::Scalar const* Hi;
    };

    // Non-throwing evaluation with caller-provided per-lane variable bounds -- including exceptions from
    // user-registered interval callbacks, caught in TryEvaluateImpl; never propagates one.
    [[nodiscard]] auto TryEvaluate(Operon::Span<Operon::Scalar const> coeff, std::span<LaneOverride const> overrides) const
        -> tl::expected<Interval, std::string>
    {
        return TryEvaluateImpl(coeff, overrides);
    }
    [[nodiscard]] auto TryEvaluate(Operon::Span<Operon::Scalar const> coeff, Operon::Hash hash,
                                   Scalar const& lo, Scalar const& hi) const
        -> tl::expected<Interval, std::string>
    {
        LaneOverride const override { hash, reinterpret_cast<Operon::Scalar const*>(&lo), reinterpret_cast<Operon::Scalar const*>(&hi) };
        return TryEvaluate(coeff, std::span { &override, 1 });
    }


    // Evaluates the tree. `coeff` has one entry per node with
    // `Node::Optimize == true`, in node order, always Operon::Scalar-typed.
    [[nodiscard]] auto Evaluate(Operon::Span<Operon::Scalar const> coeff) const -> Interval
    {
        auto result = TryEvaluate(coeff);
        if (!result) { throw std::runtime_error(result.error()); }
        return std::move(*result);
    }
    // Non-throwing evaluation, including from user-registered callbacks -- see TryEvaluateImpl. Callers should
    // never need their own try/catch around this; use it directly instead of Evaluate() + try/catch.
    [[nodiscard]] auto TryEvaluate(Operon::Span<Operon::Scalar const> coeff) const -> tl::expected<Interval, std::string>
    {
        return TryEvaluateImpl(coeff, {});
    }

private:
    [[nodiscard]] auto TryEvaluateImpl(Operon::Span<Operon::Scalar const> coeff, std::span<LaneOverride const> overrides) const
        -> tl::expected<Interval, std::string>
    {
        std::size_t coefficientIndex = 0;
        auto const weight = [&](Node const& node) {
            if (node.Optimize) {
                EXPECT(coefficientIndex < coeff.size());
                return static_cast<Scalar>(coeff[coefficientIndex++]);
            }
            return static_cast<Scalar>(node.Value);
        };
        auto const bindLeaf = [&](Node const& node, std::size_t index, Scalar scale) -> tl::expected<Interval, std::string> {
            if (node.IsConstant()) {
                return pappus::ops::constant<Scalar>(scale);
            }

            auto const override = std::ranges::find(overrides, node.HashValue, &LaneOverride::Hash);
            if (override != overrides.end()) {
                return pappus::ops::variable<Scalar>(LoadOverride(override->Lo), LoadOverride(override->Hi)) * scale;
            }

            auto const& slot = domainSlots_[index];
            if (!slot.Present) {
                return tl::unexpected(fmt::format("IntervalEvaluator: no domain bound for variable hash {}", node.HashValue));
            }
            return pappus::ops::variable<Scalar>(static_cast<Scalar>(slot.Bounds.first), static_cast<Scalar>(slot.Bounds.second)) * scale;
        };

        // EvaluatePostOrder invokes user-registered interval callbacks (RegisterUnaryInterval/
        // RegisterBinaryInterval) directly; a callback that throws must not escape here, or TryEvaluate would
        // silently stop being non-throwing despite its name and tl::expected contract, forcing every caller to
        // wrap it in its own try/catch to compensate. Caught once, here, instead.
        try {
            return detail::EvaluatePostOrder<detail::IntervalPostOrderPolicy<Scalar>>(
                tree_->Nodes(), primal_, typename detail::IntervalPostOrderPolicy<Scalar>::Context {}, weight, bindLeaf);
        } catch (std::exception const& error) {
            return tl::unexpected(std::string(error.what()));
        }
    }

    [[nodiscard]] static auto LoadOverride(Operon::Scalar const* value) -> Scalar
    {
        if constexpr (std::same_as<Scalar, Operon::Scalar>) {
            return *value;
        } else {
            return eve::load(value, eve::as<Scalar> {});
        }
    }

    void RebuildDomainSlots()
    {
        auto const& nodes = tree_->Nodes();
        domainSlots_.clear();
        domainSlots_.reserve(nodes.size());
        for (auto const& node : nodes) {
            auto const it = node.Type == NodeType::Variable ? domains_.find(node.HashValue) : domains_.end();
            domainSlots_.push_back(it == domains_.end() ? DomainSlot{} : DomainSlot{ it->second, true });
        }
    }

    struct DomainSlot {
        Domain Bounds{};
        bool Present{false};
    };

    gsl::not_null<Operon::Tree const*> tree_;
    DomainMap domains_;
    std::vector<DomainSlot> domainSlots_;
    mutable std::vector<Interval> primal_; // reused across Evaluate calls
};


} // namespace Operon

#endif
