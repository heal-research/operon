// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERVAL_EVALUATOR_HPP
#define OPERON_INTERVAL_EVALUATOR_HPP

#include <fmt/format.h>
#include <functional>
#include <gsl/pointers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/hash_registry.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

// pappus headers unconditionally redefine EXPECT/ENSURE (operon defines them
// as ASSERT wrappers) and rely on a deprecated implicit copy ctor for
// `interval<T>` (user-provided copy assignment without a copy ctor). Silence
// both for the pappus includes only; the rest of the TU keeps operon warnings.
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wmacro-redefined"
#  pragma clang diagnostic ignored "-Wdeprecated-copy-with-user-provided-copy"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmacro-redefined"
#  pragma GCC diagnostic ignored "-Wdeprecated-copy-with-user-provided-copy"
#endif
#include <pappus/pappus.hpp>
#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

namespace Operon {

// Registered interval callbacks for unary/binary built-in or user-defined
// functions, keyed by Node::HashValue. Global function-local-static
// singletons (`inline` for one-definition-rule safety, since
// IntervalEvaluator is a header-only class with no .cpp) mirroring the
// pattern used by tree_diff.cpp's SymbolicDerivRegistry: no registry
// parameter needs to thread through IntervalEvaluator's constructor or
// Evaluate(). All writes (built-in registration below, plus any user calls
// to RegisterUnaryInterval/RegisterBinaryInterval) happen before Evaluate()
// is first called from a GP worker thread, so the plain (non-locking)
// Operon::Map HashRegistry already uses is sufficient — no sharded-lock map
// type needed, same read-only-after-setup contract DispatchTable relies on.
//
// Only two argument shapes are needed (not three, despite pappus's
// PAPPUS_DEFINE_UNARY_OP macro generating three overloads per op): interval
// calls never take a context, unlike affine's context-threaded finalize
// overload (see affine_evaluator.hpp).
using IntervalUnaryFn  = std::function<pappus::interval<Operon::Scalar>(pappus::interval<Operon::Scalar> const&)>;
using IntervalBinaryFn = std::function<pappus::interval<Operon::Scalar>(pappus::interval<Operon::Scalar> const&, pappus::interval<Operon::Scalar> const&)>;

using IntervalUnaryRegistry  = HashRegistry<IntervalUnaryFn>;
using IntervalBinaryRegistry = HashRegistry<IntervalBinaryFn>;

// Direct registry access — needed by RegisterUnaryInterval/RegisterBinaryInterval
// below (same TU) and by tests that assert on registration state. Prefer
// RegisterUnaryInterval/RegisterBinaryInterval for registering a rule: calling
// .Register() on the registry returned here directly skips the built-in
// lazy-init those functions trigger first, reopening the ordering hazard they
// exist to close (a user hash colliding with a built-in would be silently
// accepted instead of throwing immediately).
inline auto IntervalUnaryRules() -> IntervalUnaryRegistry&
{
    static IntervalUnaryRegistry registry;
    return registry;
}

inline auto IntervalBinaryRules() -> IntervalBinaryRegistry&
{
    static IntervalBinaryRegistry registry;
    return registry;
}

// Registers the built-in unary/binary interval rules exactly once, mirroring
// StandardLibrary::RegisterNames()'s lazy-static-lambda-once pattern. Every
// rule is lifted verbatim out of IntervalEvaluator's old switch statement;
// each lambda calls `pappus::ops::xxx` fully qualified rather than relying on
// ADL, sidestepping the pappus::ops sibling-namespace ADL gap entirely (that
// gap only bites a generic lambda expecting implicit lookup — these are
// written inside operon's own code and can just spell out the qualified
// name, same as the switch bodies already did). A free function (not an
// IntervalEvaluator member) so both IntervalEvaluator::Evaluate() and the
// public Register{Unary,Binary}Interval() entry points below can call it —
// the latter must trigger it before writing, so that a user hash colliding
// with a built-in throws immediately at the user's own call site instead of
// being accepted now and only discovered (as a confusing, differently-hashed
// throw) the first time Evaluate() runs.
inline void RegisterIntervalBuiltins()
{
    using Scalar = Operon::Scalar;
    using Interval = pappus::interval<Scalar>;
    static auto const registered = [] {
        auto& unary  = IntervalUnaryRules();
        auto& binary = IntervalBinaryRules();

        unary.Register(Operon::Hash(BuiltinOp::Square),  [](Interval const& v) { return pappus::ops::square<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrt),    [](Interval const& v) { return pappus::ops::sqrt<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Exp),     [](Interval const& v) { return pappus::ops::exp<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Log),     [](Interval const& v) { return pappus::ops::log<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sin),     [](Interval const& v) { return pappus::ops::sin<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cos),     [](Interval const& v) { return pappus::ops::cos<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Tan),     [](Interval const& v) { return pappus::ops::tan<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Asin),    [](Interval const& v) { return pappus::ops::asin<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Acos),    [](Interval const& v) { return pappus::ops::acos<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Atan),    [](Interval const& v) { return pappus::ops::atan<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sinh),    [](Interval const& v) { return pappus::ops::sinh<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cosh),    [](Interval const& v) { return pappus::ops::cosh<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Tanh),    [](Interval const& v) { return pappus::ops::tanh<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Abs),     [](Interval const& v) { return pappus::ops::abs<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrtabs), [](Interval const& v) { return pappus::ops::sqrtabs<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Logabs),  [](Interval const& v) { return pappus::ops::logabs<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cbrt),    [](Interval const& v) { return pappus::ops::cbrt<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Log1p),   [](Interval const& v) { return pappus::ops::log1p<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Floor),   [](Interval const& v) { return pappus::ops::floor<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Ceil),    [](Interval const& v) { return pappus::ops::ceil<Scalar>(v); });

        binary.Register(Operon::Hash(BuiltinOp::Pow), [](Interval const& a, Interval const& b) {
            return pappus::ops::pow<Scalar>(a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Aq), [](Interval const& a, Interval const& b) {
            return pappus::ops::aq<Scalar>(a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Powabs), [](Interval const& a, Interval const& b) {
            return pappus::ops::pow<Scalar>(pappus::ops::abs<Scalar>(a), b);
        });

        return true;
    }();
    static_cast<void>(registered);
}

// Register an interval callback for a unary function (built-in or
// user-defined), keyed by the same hash the function's Node::HashValue
// carries. A miss at evaluation time is not "not differentiable" the way a
// missing tree_diff rule is — it means the tree cannot be bounded at all,
// and IntervalEvaluator::Evaluate() still throws on a miss, unchanged from
// today. Throws if `hash` is already registered (write-once) — including
// when `hash` collides with a built-in, since RegisterIntervalBuiltins()
// above always runs first.
inline void RegisterUnaryInterval(Operon::Hash hash, IntervalUnaryFn fn)
{
    RegisterIntervalBuiltins();
    IntervalUnaryRules().Register(hash, std::move(fn));
}

// Register an interval callback for a binary function. See
// RegisterUnaryInterval for the miss-behavior and built-in-collision notes.
inline void RegisterBinaryInterval(Operon::Hash hash, IntervalBinaryFn fn)
{
    RegisterIntervalBuiltins();
    IntervalBinaryRules().Register(hash, std::move(fn));
}

// Forward rigorous bounds for an Operon tree over a single input domain.
//
// Walks the tree in post-order (the same order used by the Operon interpreter)
// and computes a `pappus::interval<Operon::Scalar>` enclosure for each node.
// Variables are bound to a user-supplied domain map keyed by `Node::HashValue`;
// constants and node weights come from the coefficient span (mirroring
// `Tree::GetCoefficients()` / `Node::Optimize`).
//
// Batch size is 1: one evaluation produces one interval enclosure for the whole
// domain. This is intentional -- interval/affine arithmetic are single-value
// computations, not per-row fitness evaluations.
//
// Domain-error policy: operations on out-of-domain inputs (e.g. log of a
// negative interval, sqrt of a negative interval, log1p of an interval
// entirely below -1) return `interval::empty()` — a NaN-bounds interval.
// This empty result propagates silently through subsequent operations (e.g.
// exp(empty()) = empty()) and is returned from Evaluate() without throwing.
// The caller must check `result.is_empty()` if domain validity matters.
// This differs from AffineEvaluator, which throws std::invalid_argument for
// the same out-of-domain inputs. See the pappus handoff doc for the rationale:
// "interval and affine do not have identical domain semantics".
class IntervalEvaluator {
public:
    using Scalar = Operon::Scalar;
    using Interval = pappus::interval<Scalar>;
    // (lower, upper) bound for a variable identified by its hash.
    using Domain = std::pair<Scalar, Scalar>;
    using DomainMap = Operon::Map<Operon::Hash, Domain>;

    IntervalEvaluator(gsl::not_null<Operon::Tree const*> tree, DomainMap domains)
        : tree_(tree), domains_(std::move(domains)) {}

    [[nodiscard]] auto GetTree() const noexcept -> Operon::Tree const* { return tree_.get(); }
    [[nodiscard]] auto Domains() const noexcept -> DomainMap const& { return domains_; }

    // Evaluate the tree over the supplied domains. `coeff` follows the same
    // convention as `Interpreter::Evaluate`: one entry per node with
    // `Node::Optimize == true`, consumed in node order.
    [[nodiscard]] auto Evaluate(Operon::Span<Scalar const> coeff) const -> Interval
    {
        RegisterIntervalBuiltins();

        auto const& nodes = tree_->Nodes();
        auto const n = nodes.size();
        if (n == 0) { throw std::runtime_error("IntervalEvaluator: empty tree"); }

        primal_.resize(n);
        std::size_t ci = 0;

        // Folds over the immediate children of node `i`, reading from `primal_`.
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
        // `first - (rest[0] + rest[1] + ...)` matching Operon's n-ary Sub.
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
        // `first / (rest[0] * rest[1] * ...)` matching Operon's n-ary Div.
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
            // Per-node weight/value: leaves use it as value (Constant) or weight
            // (Variable); non-leaves apply it as a post-multiply (matches the
            // Operon `EvaluateTree` reference). Non-leaf nodes default to
            // `Optimize == false`, so this is usually 1.0.
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
                auto it = domains_.find(node.HashValue);
                if (it == domains_.end()) {
                    throw std::runtime_error(fmt::format(
                        "IntervalEvaluator: no domain bound for variable hash {}",
                        node.HashValue));
                }
                auto const& [lo, hi] = it->second;
                primal_[i] = pappus::ops::variable<Scalar>(lo, hi) * v;
            } else if (node.Type == NodeType::Ref) {
                EXPECT(static_cast<std::size_t>(node.RefTo) < i);
                primal_[i] = primal_[node.RefTo];
            } else {
                // Add/Mul/Sub/Div/Fmin/Fmax stay hardcoded: each is a verified
                // n-ary fold over an arbitrary number of children, not a
                // single-hash unary/binary registry entry. Every other op —
                // unary and binary alike — goes through the registry,
                // built-in or user-defined.
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
                    // Gated on the node's actual arity (exactly 1 for unary,
                    // exactly 2 for binary) so a hash mistakenly registered
                    // under the wrong registry falls through to the "not yet
                    // mapped" throw below, rather than reading a nonexistent
                    // operand (arity 1 through the binary path), silently
                    // ignoring one (arity 2 through the unary path), or —
                    // for arity 0 or arity >= 3 registered as binary —
                    // reading unrelated primal_ entries or dropping operands
                    // beyond the first two.
                    if (node.Arity == 1) {
                        if (auto const* unary = IntervalUnaryRules().TryGet(node.HashValue)) {
                            primal_[i] = (*unary)(primal_[i - 1]) * v;
                            break;
                        }
                    } else if (node.Arity == 2) {
                        if (auto const* binary = IntervalBinaryRules().TryGet(node.HashValue)) {
                            auto const j = static_cast<std::size_t>(i - 1);
                            auto const k = j - (nodes[j].Length + 1);
                            primal_[i] = (*binary)(primal_[j], primal_[k]) * v;
                            break;
                        }
                    }
                    throw std::runtime_error(fmt::format(
                        "IntervalEvaluator: node kind `{}` not yet mapped",
                        node.Name()));
                }
            }
        }
        return primal_.back();
    }

private:
    gsl::not_null<Operon::Tree const*> tree_;
    DomainMap domains_;
    mutable std::vector<Interval> primal_; // reused across Evaluate calls
};

} // namespace Operon

#endif
