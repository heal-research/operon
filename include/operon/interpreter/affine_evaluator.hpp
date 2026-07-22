// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_AFFINE_EVALUATOR_HPP
#define OPERON_AFFINE_EVALUATOR_HPP

#include <fmt/format.h>
#include <functional>
#include <gsl/pointers>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/hash_registry.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

// See interval_evaluator.hpp for the rationale behind these pragmas.
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

// Registered affine callbacks for unary/binary built-in or user-defined
// functions, keyed by Node::HashValue. Exported `.cpp`-backed singletons
// (see affine_evaluator.cpp) — see interval_evaluator.hpp's matching comment
// for the full rationale (no registry parameter threaded through the
// constructor/Evaluate(), plain non-locking HashRegistry since all writes
// finish before Evaluate() is first called from a GP worker thread). Only
// real difference from IntervalUnaryFn/IntervalBinaryFn: every call here
// takes the shared affine_context first — the context-threaded finalize
// overload of PAPPUS_DEFINE_UNARY_OP is the only one either evaluator ever
// calls (the bare affine_form<T>-only overload, without a context, is never
// used here).
using AffineUnaryFn  = std::function<pappus::affine_form<Operon::Scalar>(
    pappus::ops::affine_context<Operon::Scalar> const&, pappus::affine_form<Operon::Scalar> const&)>;
using AffineBinaryFn = std::function<pappus::affine_form<Operon::Scalar>(
    pappus::ops::affine_context<Operon::Scalar> const&,
    pappus::affine_form<Operon::Scalar> const&, pappus::affine_form<Operon::Scalar> const&)>;

using AffineUnaryRegistry  = HashRegistry<AffineUnaryFn>;
using AffineBinaryRegistry = HashRegistry<AffineBinaryFn>;

// Direct registry access — see interval_evaluator.hpp's matching comment.
// Prefer RegisterUnaryAffine/RegisterBinaryAffine over calling .Register()
// on the registry returned here directly.
OPERON_EXPORT auto AffineUnaryRules() -> AffineUnaryRegistry&;
OPERON_EXPORT auto AffineBinaryRules() -> AffineBinaryRegistry&;

// Registers the built-in unary/binary affine rules exactly once, mirroring
// interval_evaluator.hpp's RegisterIntervalBuiltins(). A free function (not
// an AffineEvaluator member) so both AffineEvaluator::Evaluate() and the
// public Register{Unary,Binary}Affine() entry points below can call it — the
// latter must trigger it before writing, for the same reason
// RegisterIntervalBuiltins() does: a user hash colliding with a built-in
// should throw immediately at the user's own call site, not later inside
// Evaluate().
OPERON_EXPORT void RegisterAffineBuiltins();

// Register an affine callback for a unary function (built-in or
// user-defined), keyed by the same hash the function's Node::HashValue
// carries. A miss at evaluation time means the tree cannot be bounded at
// all — AffineEvaluator::Evaluate() still throws on a miss, unchanged from
// today. Throws if `hash` is already registered (write-once) — including
// when `hash` collides with a built-in, since RegisterAffineBuiltins() above
// always runs first.
OPERON_EXPORT void RegisterUnaryAffine(Operon::Hash hash, AffineUnaryFn fn);

// Register an affine callback for a binary function. See
// RegisterUnaryAffine for the miss-behavior and built-in-collision notes.
OPERON_EXPORT void RegisterBinaryAffine(Operon::Hash hash, AffineBinaryFn fn);

// Query whether an affine callback is registered for `hash` (built-in or
// user-defined), forcing built-in registration first. See
// HasUnaryInterval/HasBinaryInterval for the coverage-check use case.
OPERON_EXPORT auto HasUnaryAffine(Operon::Hash hash) -> bool;
OPERON_EXPORT auto HasBinaryAffine(Operon::Hash hash) -> bool;

// Forward affine-arithmetic bounds for an Operon tree over a single input
// domain. Mirrors `IntervalEvaluator` but every `affine_form` shares one
// `pappus::ops::affine_context<Operon::Scalar>` owned by this evaluator.
//
// Domain-error policy: unlike interval arithmetic, `affine_form::inv()` throws
// when the form's interval contains zero (affine forms cannot represent
// unbounded values). Similarly, `log()`, `log1p()`, and `sqrt()` throw
// std::invalid_argument for out-of-domain inputs. This evaluator lets such
// exceptions propagate -- the caller decides whether to catch or treat as a
// hard failure.
//
// This differs from IntervalEvaluator, where out-of-domain operations return
// `interval::empty()` (NaN bounds) silently. See the pappus handoff doc:
// "interval and affine do not have identical domain semantics".
//
// `max_terms` policy (Phase 8c): defaults to `0` (unbounded / exact). There is
// no build-time default — the right budget depends on tree depth and how many
// Evaluate() calls are composed, which varies per GP individual. Expose it only
// as a per-instance constructor argument. Set a finite budget only after
// profiling confirms term growth is a bottleneck (TermCount() helps measure).
class AffineEvaluator {
public:
    using Scalar = Operon::Scalar;
    using Affine = pappus::affine_form<Scalar>;
    using Interval = pappus::interval<Scalar>;
    using Domain = std::pair<Scalar, Scalar>;
    using DomainMap = Operon::Map<Operon::Hash, Domain>;
    using Context = pappus::ops::affine_context<Scalar>;

    AffineEvaluator(gsl::not_null<Operon::Tree const*> tree, DomainMap domains,
                    std::size_t maxTerms = 0)
        : tree_(tree), domains_(std::move(domains))
    {
        ctx_.max_terms = maxTerms;
    }

    [[nodiscard]] auto GetTree() const noexcept -> Operon::Tree const* { return tree_.get(); }
    [[nodiscard]] auto Domains() const noexcept -> DomainMap const& { return domains_; }
    [[nodiscard]] auto GetContext() const noexcept -> Context const& { return ctx_; }
    // Number of noise terms in the last root result (0 if Evaluate has not been called).
    // Useful for profiling affine term growth under different max_terms settings.
    [[nodiscard]] auto TermCount() const noexcept -> std::size_t {
        return primal_.empty() ? 0 : primal_.back().size();
    }

    // The affine interval enclosure of the root.
    [[nodiscard]] auto Evaluate(Operon::Span<Scalar const> coeff) const -> Affine
    {
        // The shared noise-symbol counter grows monotonically across Evaluate
        // calls — it is NOT reset. This ensures that forms from different
        // evaluations get distinct noise-symbol indices, so combining them
        // (e.g. r1 - r2) treats the uncertainties as independent and produces
        // sound (conservative) bounds. Resetting the counter would reuse
        // indices, causing pappus to merge terms from independent evaluations
        // and produce falsely-narrow enclosures.

        RegisterAffineBuiltins();

        auto const& nodes = tree_->Nodes();
        auto const n = nodes.size();
        if (n == 0) { throw std::runtime_error("AffineEvaluator: empty tree"); }

        // affine_form is non-default-constructible (it binds to a context), so
        // we can't resize — clear() preserves capacity and push_back reuses it.
        primal_.clear();
        primal_.reserve(n);
        std::size_t ci = 0;

        // Add/Mul: identity-seeded folds — no spurious affine_form copy.
        auto const addFold = [&](std::size_t i) {
            auto acc = pappus::ops::constant<Scalar>(ctx_, Scalar{0});
            for (auto j : Tree::Indices(nodes, i)) { acc = pappus::ops::add<Scalar>(ctx_, acc, primal_[j]); }
            EXPECT(nodes[i].Arity > 0);
            return acc;
        };
        auto const mulFold = [&](std::size_t i) {
            auto acc = pappus::ops::constant<Scalar>(ctx_, Scalar{1});
            for (auto j : Tree::Indices(nodes, i)) { acc = pappus::ops::mul<Scalar>(ctx_, acc, primal_[j]); }
            EXPECT(nodes[i].Arity > 0);
            return acc;
        };
        // Sub/Div: first-child-seeded via std::optional — no placeholder copy.
        auto const subFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(nodes, i)) {
                if (!acc) { acc = primal_[j]; }
                else      { acc = pappus::ops::sub<Scalar>(ctx_, *acc, primal_[j]); }
            }
            EXPECT(acc.has_value()); // arity > 0 — malformed tree otherwise
            return std::move(*acc);
        };
        auto const divFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(nodes, i)) {
                if (!acc) { acc = primal_[j]; }
                else      { acc = pappus::ops::div<Scalar>(ctx_, *acc, primal_[j]); }
            }
            EXPECT(acc.has_value()); // arity > 0 — malformed tree otherwise
            return std::move(*acc);
        };
        // Fmin/Fmax are n-ary (Node::IsNaryOp<BuiltinOp> covers Add..Fmax). Fold
        // all children — do NOT hardcode binary j/k indexing.
        auto const minFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(nodes, i)) {
                if (!acc) { acc = primal_[j]; }
                else      { acc = pappus::ops::min<Scalar>(ctx_, *acc, primal_[j]); }
            }
            EXPECT(acc.has_value());
            return std::move(*acc);
        };
        auto const maxFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(nodes, i)) {
                if (!acc) { acc = primal_[j]; }
                else      { acc = pappus::ops::max<Scalar>(ctx_, *acc, primal_[j]); }
            }
            EXPECT(acc.has_value());
            return std::move(*acc);
        };

        for (std::size_t i = 0; i < n; ++i) {
            auto const& node = nodes[i];
            Scalar v;
            if (node.Optimize) {
                EXPECT(ci < coeff.size());
                v = static_cast<Scalar>(coeff[ci++]);
            } else {
                v = static_cast<Scalar>(node.Value);
            }

            if (node.Type == NodeType::Constant) {
                primal_.push_back(pappus::ops::constant<Scalar>(ctx_, v));
            } else if (node.Type == NodeType::Variable) {
                auto it = domains_.find(node.HashValue);
                if (it == domains_.end()) {
                    throw std::runtime_error(fmt::format(
                        "AffineEvaluator: no domain bound for variable hash {}",
                        node.HashValue));
                }
                auto const& [lo, hi] = it->second;
                primal_.push_back(pappus::ops::variable<Scalar>(ctx_, lo, hi) * v);
            } else if (node.Type == NodeType::Ref) {
                EXPECT(static_cast<std::size_t>(node.RefTo) < primal_.size());
                primal_.push_back(primal_[node.RefTo]);
            } else {
                // Add/Mul/Sub/Div/Fmin/Fmax stay hardcoded: verified n-ary
                // folds, same scope boundary as IntervalEvaluator. Every
                // other op goes through the registry.
                switch (node.HashValue) {
                case Operon::Hash(BuiltinOp::Add):
                    primal_.push_back(addFold(i) * v);
                    break;
                case Operon::Hash(BuiltinOp::Mul):
                    primal_.push_back(mulFold(i) * v);
                    break;
                case Operon::Hash(BuiltinOp::Sub):
                    primal_.push_back((node.Arity == 1 ? pappus::ops::neg<Scalar>(primal_[i - 1])
                                                      : subFold(i)) * v);
                    break;
                case Operon::Hash(BuiltinOp::Div):
                    // May throw if the denominator form contains zero (affine inv
                    // is stricter than interval inv).
                    primal_.push_back((node.Arity == 1 ? pappus::ops::inv<Scalar>(ctx_, primal_[i - 1])
                                                      : divFold(i)) * v);
                    break;
                case Operon::Hash(BuiltinOp::Fmin):
                    primal_.push_back(minFold(i) * v);
                    break;
                case Operon::Hash(BuiltinOp::Fmax):
                    primal_.push_back(maxFold(i) * v);
                    break;
                default:
                    // Gated on the node's actual arity (exactly 1 for unary,
                    // exactly 2 for binary) — see interval_evaluator.hpp's
                    // matching comment for the full rationale, including why
                    // arity 0 / arity >= 3 must also fall through to the
                    // throw rather than only guarding against 1-vs-2.
                    if (node.Arity == 1) {
                        if (auto const* unary = AffineUnaryRules().TryGet(node.HashValue)) {
                            primal_.push_back((*unary)(ctx_, primal_[i - 1]) * v);
                            break;
                        }
                    } else if (node.Arity == 2) {
                        if (auto const* binary = AffineBinaryRules().TryGet(node.HashValue)) {
                            auto const j = static_cast<std::size_t>(i - 1);
                            auto const k = j - (nodes[j].Length + 1);
                            primal_.push_back((*binary)(ctx_, primal_[j], primal_[k]) * v);
                            break;
                        }
                    }
                    throw std::runtime_error(fmt::format(
                        "AffineEvaluator: node kind `{}` not yet mapped",
                        node.Name()));
                }
            }
        }
        return primal_.back();
    }

private:
    gsl::not_null<Operon::Tree const*> tree_;
    DomainMap domains_;
    mutable Context ctx_; // shared by all forms; counter grows monotonically
    mutable std::vector<Affine> primal_; // reused across Evaluate calls
};

} // namespace Operon

#endif
