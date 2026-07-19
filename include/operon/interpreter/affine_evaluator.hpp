// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_AFFINE_EVALUATOR_HPP
#define OPERON_AFFINE_EVALUATOR_HPP

#include <fmt/format.h>
#include <gsl/pointers>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

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
                case Operon::Hash(BuiltinOp::Square):
                    primal_.push_back(pappus::ops::square<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Sqrt):
                    primal_.push_back(pappus::ops::sqrt<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Exp):
                    primal_.push_back(pappus::ops::exp<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Log):
                    primal_.push_back(pappus::ops::log<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Sin):
                    primal_.push_back(pappus::ops::sin<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Cos):
                    primal_.push_back(pappus::ops::cos<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Tan):
                    primal_.push_back(pappus::ops::tan<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Asin):
                    primal_.push_back(pappus::ops::asin<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Acos):
                    primal_.push_back(pappus::ops::acos<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Atan):
                    primal_.push_back(pappus::ops::atan<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Sinh):
                    primal_.push_back(pappus::ops::sinh<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Cosh):
                    primal_.push_back(pappus::ops::cosh<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Tanh):
                    primal_.push_back(pappus::ops::tanh<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Pow): {
                    auto const j = static_cast<std::size_t>(i - 1);
                    auto const k = j - (nodes[j].Length + 1);
                    primal_.push_back(pappus::ops::pow<Scalar>(ctx_, primal_[j], primal_[k]) * v);
                    break;
                }
                case Operon::Hash(BuiltinOp::Aq): {
                    auto const j = static_cast<std::size_t>(i - 1);
                    auto const k = j - (nodes[j].Length + 1);
                    primal_.push_back(pappus::ops::aq<Scalar>(ctx_, primal_[j], primal_[k]) * v);
                    break;
                }
                case Operon::Hash(BuiltinOp::Abs):
                    // May throw if the domain crosses zero (requires Chebyshev V-shape).
                    primal_.push_back(pappus::ops::abs<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Sqrtabs):
                    primal_.push_back(pappus::ops::sqrtabs<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Logabs):
                    primal_.push_back(pappus::ops::logabs<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Powabs): {
                    auto const j = static_cast<std::size_t>(i - 1);
                    auto const k = j - (nodes[j].Length + 1);
                    auto absBase = pappus::ops::abs<Scalar>(ctx_, primal_[j]);
                    primal_.push_back(pappus::ops::pow<Scalar>(ctx_, absBase, primal_[k]) * v);
                    break;
                }
                case Operon::Hash(BuiltinOp::Fmin):
                    primal_.push_back(minFold(i) * v);
                    break;
                case Operon::Hash(BuiltinOp::Fmax):
                    primal_.push_back(maxFold(i) * v);
                    break;
                case Operon::Hash(BuiltinOp::Cbrt):
                    primal_.push_back(pappus::ops::cbrt<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Log1p):
                    // May throw if the domain includes values <= -1.
                    primal_.push_back(pappus::ops::log1p<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Floor):
                    primal_.push_back(pappus::ops::floor<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                case Operon::Hash(BuiltinOp::Ceil):
                    primal_.push_back(pappus::ops::ceil<Scalar>(ctx_, primal_[i - 1]) * v);
                    break;
                default:
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
