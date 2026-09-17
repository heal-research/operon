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
    {
        auto const& nodes = tree_->Nodes();
        domainSlots_.reserve(nodes.size());
        for (auto const& node : nodes) {
            auto const it = node.Type == NodeType::Variable ? domains.find(node.HashValue) : domains.end();
            domainSlots_.push_back(it == domains.end() ? DomainSlot{} : DomainSlot{ it->second, true });
        }
    }

    [[nodiscard]] auto GetTree() const noexcept -> Operon::Tree const* { return tree_.get(); }

    // Non-owning lane bounds for one variable. Lo and Hi must remain valid through
    // TryEvaluate; evaluators copy their values and never retain pointers.
    struct LaneOverride {
        Operon::Hash Hash;
        Operon::Scalar const* Lo;
        Operon::Scalar const* Hi;
    };

    // Non-throwing evaluation with caller-provided per-lane variable bounds.
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
    // Non-throwing evaluation for callers that cannot unwind an active SIMD frame.
    [[nodiscard]] auto TryEvaluate(Operon::Span<Operon::Scalar const> coeff) const -> tl::expected<Interval, std::string>
    {
        return TryEvaluateImpl(coeff, {});
    }

private:
    [[nodiscard]] auto TryEvaluateImpl(Operon::Span<Operon::Scalar const> coeff, std::span<LaneOverride const> overrides) const
        -> tl::expected<Interval, std::string>
    {
        RegisterIntervalBuiltins<Scalar>();

        auto const& nodes = tree_->Nodes();
        auto const n = nodes.size();
        if (n == 0) { return tl::unexpected("IntervalEvaluator: empty tree"); }

        primal_.resize(n);
        std::size_t ci = 0;

        // Folds over node i's immediate children, reading from `primal_`.
        auto const addFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(nodes, i)) { acc = pappus::ops::add<Scalar>(acc, primal_[j]); }
            return acc;
        };
        auto const mulFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{1}};
            for (auto j : Tree::Indices(nodes, i)) { acc = pappus::ops::mul<Scalar>(acc, primal_[j]); }
            return acc;
        };
        // first - (rest[0] + rest[1] + ...), matching Operon's n-ary Sub.
        auto const subFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}}; // overwritten on first child
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = primal_[j]; first = false; }
                else       { acc = pappus::ops::sub<Scalar>(acc, primal_[j]); }
            }
            EXPECT(!first); // arity > 0 — malformed tree otherwise
            return acc;
        };
        // first / (rest[0] * rest[1] * ...), matching Operon's n-ary Div.
        auto const divFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{1}}; // overwritten on first child
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = primal_[j]; first = false; }
                else       { acc = pappus::ops::div<Scalar>(acc, primal_[j]); }
            }
            EXPECT(!first); // arity > 0 — malformed tree otherwise
            return acc;
        };
        // min([a1,b1], [a2,b2], ...) = [min(a1,a2,...), min(b1,b2,...)]
        auto const minFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = primal_[j]; first = false; }
                else       { acc = pappus::ops::min<Scalar>(acc, primal_[j]); }
            }
            EXPECT(!first);
            return acc;
        };
        // max([a1,b1], [a2,b2], ...) = [max(a1,a2,...), max(b1,b2,...)]
        auto const maxFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = primal_[j]; first = false; }
                else       { acc = pappus::ops::max<Scalar>(acc, primal_[j]); }
            }
            EXPECT(!first);
            return acc;
        };

        for (std::size_t i = 0; i < n; ++i) {
            auto const& node = nodes[i];
            // Leaves use v as value (Constant) or weight (Variable);
            // non-leaves apply it as a post-multiply.
            Scalar v;
            if (node.Optimize) {
                EXPECT(ci < coeff.size());
                v = static_cast<Scalar>(coeff[ci++]);
            } else {
                v = static_cast<Scalar>(node.Value);
            }

            if (node.Type == NodeType::Constant) {
                primal_[i] = pappus::ops::constant<Scalar>(v);
            } else if (node.Type == NodeType::Variable) {
                Scalar lo{};
                Scalar hi{};
                auto const override = std::ranges::find(overrides, node.HashValue, &LaneOverride::Hash);
                if (override != overrides.end()) {
                    lo = LoadOverride(override->Lo);
                    hi = LoadOverride(override->Hi);
                } else {
                    auto const& slot = domainSlots_[i];
                    if (!slot.Present) {
                        return tl::unexpected(fmt::format(
                            "IntervalEvaluator: no domain bound for variable hash {}",
                            node.HashValue));
                    }
                    lo = static_cast<Scalar>(slot.Bounds.first);
                    hi = static_cast<Scalar>(slot.Bounds.second);
                }
                primal_[i] = pappus::ops::variable<Scalar>(lo, hi) * v;
            } else if (node.Type == NodeType::Ref) {
                EXPECT(static_cast<std::size_t>(node.RefTo) < i);
                primal_[i] = primal_[node.RefTo];
            } else {
                // Add/Mul/Sub/Div/Fmin/Fmax are n-ary folds handled directly;
                // every other op goes through the unary/binary registry.
                switch (node.HashValue) {
                case Operon::Hash(BuiltinOp::Add):
                    primal_[i] = addFold(i) * v;
                    break;
                case Operon::Hash(BuiltinOp::Mul):
                    primal_[i] = mulFold(i) * v;
                    break;
                case Operon::Hash(BuiltinOp::Sub):
                    primal_[i] = (node.Arity == 1 ? pappus::ops::neg<Scalar>(primal_[i - 1])
                                                  : subFold(i)) * v;
                    break;
                case Operon::Hash(BuiltinOp::Div):
                    primal_[i] = (node.Arity == 1 ? pappus::ops::inv<Scalar>(primal_[i - 1])
                                                  : divFold(i)) * v;
                    break;
                case Operon::Hash(BuiltinOp::Fmin):
                    primal_[i] = minFold(i) * v;
                    break;
                case Operon::Hash(BuiltinOp::Fmax):
                    primal_[i] = maxFold(i) * v;
                    break;
                default:
                    // Gated on arity so a hash registered under the wrong
                    // registry falls through to the throw below instead of
                    // reading/dropping the wrong operands.
                    if (node.Arity == 1) {
                        if (auto const* unary = IntervalUnaryRules<Scalar>().TryGet(node.HashValue)) {
                            primal_[i] = (*unary)(primal_[i - 1]) * v;
                            break;
                        }
                    } else if (node.Arity == 2) {
                        if (auto const* binary = IntervalBinaryRules<Scalar>().TryGet(node.HashValue)) {
                            auto const j = static_cast<std::size_t>(i - 1);
                            auto const k = j - (nodes[j].Length + 1);
                            primal_[i] = (*binary)(primal_[j], primal_[k]) * v;
                            break;
                        }
                    }
                    return tl::unexpected(fmt::format(
                        "IntervalEvaluator: node kind `{}` not yet mapped",
                        node.Name()));
                }
            }
        }
        return primal_.back();
    }

    [[nodiscard]] static auto LoadOverride(Operon::Scalar const* value) -> Scalar
    {
        if constexpr (std::same_as<Scalar, Operon::Scalar>) {
            return *value;
        } else {
            return eve::load(value, eve::as<Scalar> {});
        }
    }

    struct DomainSlot {
        Domain Bounds{};
        bool Present{false};
    };

    gsl::not_null<Operon::Tree const*> tree_;
    std::vector<DomainSlot> domainSlots_;
    mutable std::vector<Interval> primal_; // reused across Evaluate calls
};


} // namespace Operon

#endif
