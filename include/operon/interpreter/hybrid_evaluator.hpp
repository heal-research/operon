// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// EXPERIMENTAL, branch-only prototype for the per-node adaptive
// affine-to-interval collapse design:
// operon-publications/papers/interval-range-tightening/per-node-collapse-design.md
//
// Walks affine and interval arithmetic over the same tree in lockstep.
// At every internal node, if the node's affine bound is both (a) certified
// (not ill-conditioned, per the same MaxAbsCenter*eps > threshold*radius
// check TryAffineBoundDirect already uses at the root, applied here with a
// LOCAL per-subtree max center instead of a whole-tree one) and (b) no
// tighter than the interval bound, keep it; otherwise DISCARD the affine
// form and replace it with a FRESH, independently-noise-symbol'd one built
// from the (already node-weight-applied) interval bound via
// `affine_form(ctx, interval)`. This never narrows an existing noise
// symbol's domain -- see the design doc's "Why this is sound" section.
//
// Explicitly NOT production code. Not wired into ShapeConstrainedEvaluator,
// ShapeBoundMode, or any CLI flag. Exists to measure root-level tightness
// impact on real trees before deciding whether to build this for real.
//
// Revision history (this file has been through two rounds of implementation
// review, both finding real bugs in the first draft -- kept here since it's
// directly relevant to trusting this file's own correctness):
// - localMax_ (the per-subtree ill-conditioning tracker) did not propagate
//   from children into parents -- a fold/unary/binary op only tracked its
//   own accumulator's center, never read a child's already-computed
//   localMax_. A child with a catastrophic intermediate that later
//   cancelled down to a small final center could hide that history from
//   every ancestor's own certification check. Fixed: every op now seeds
//   localMax_[i] from every child's localMax_[j] before/while folding.
// - The node weight `v` was applied to the affine value `a` uniformly
//   after the dispatch switch, but only to `ivl` inside the four hardcoded
//   n-ary fold cases (Add/Mul/Sub/Div/Fmin/Fmax) -- the unary/binary
//   registry dispatch path (exp, sin, sqrt, user-defined ops, ...) never
//   applied `v` to `ivl` at all. Any weighted registry-op node had its
//   collapse decision comparing a weighted affine width against an
//   unweighted interval width, and a triggered collapse built a fresh
//   affine form from the wrong (unweighted) interval. Fixed: `v` is now
//   applied to both `a` and `ivl` in exactly one place, after computing
//   both as raw (unweighted) op results -- removed from every individual
//   case body.
#ifndef OPERON_HYBRID_EVALUATOR_HPP
#define OPERON_HYBRID_EVALUATOR_HPP

#include <cmath>
#include <cstdlib>
#include <fmt/format.h>
#include <gsl/pointers>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"

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

class HybridEvaluator {
public:
    using Scalar = Operon::Scalar;
    using Affine = pappus::affine_form<Scalar>;
    using Interval = pappus::interval<Scalar>;
    using Domain = std::pair<Scalar, Scalar>;
    using DomainMap = Operon::Map<Operon::Hash, Domain>;
    using Context = pappus::ops::affine_context<Scalar>;

    // Threshold matching ShapeAffineIllConditionedThreshold()'s default
    // (4.0) -- not shared with that (unexported) function since this is a
    // standalone experimental evaluator, kept parallel deliberately.
    static auto IllConditionedThreshold() -> Scalar {
        static auto const threshold = [] {
            auto const* env = std::getenv("OPERON_HYBRID_ILL_THRESHOLD");
            if (!env) { return Scalar{4}; }
            auto v = static_cast<Scalar>(std::atof(env));
            return (std::isfinite(v) && v > Scalar{0}) ? v : Scalar{4};
        }();
        return threshold;
    }

    // Collapse margin: only collapse when interval is tighter by more than
    // this relative margin (0 = collapse on any tighter-ness). Open
    // question in the design doc -- exposed as a knob for the pilot.
    static auto CollapseMargin() -> Scalar {
        static auto const margin = [] {
            auto const* env = std::getenv("OPERON_HYBRID_COLLAPSE_MARGIN");
            if (!env) { return Scalar{0}; }
            auto v = static_cast<Scalar>(std::atof(env));
            return (std::isfinite(v) && v >= Scalar{0} && v < Scalar{1}) ? v : Scalar{0};
        }();
        return margin;
    }

    HybridEvaluator(gsl::not_null<Operon::Tree const*> tree, DomainMap domains)
        : tree_(tree), domains_(std::move(domains)) {}

    void SetTree(gsl::not_null<Operon::Tree const*> tree) noexcept { tree_ = tree; }

    [[nodiscard]] auto CollapseCount() const noexcept -> std::size_t { return collapseCount_; }
    [[nodiscard]] auto InternalNodeCount() const noexcept -> std::size_t { return internalCount_; }

    [[nodiscard]] auto Evaluate(Operon::Span<Scalar const> coeff) const -> Affine
    {
        RegisterAffineBuiltins();
        RegisterIntervalBuiltins();

        auto const& nodes = tree_->Nodes();
        auto const n = nodes.size();
        if (n == 0) { throw std::runtime_error("HybridEvaluator: empty tree"); }

        aprimal_.clear(); aprimal_.reserve(n);
        iprimal_.resize(n);
        localMax_.assign(n, Scalar{0});
        variableCache_.clear();
        collapseCount_ = 0;
        internalCount_ = 0;
        std::size_t ci = 0;

        // Seed node i's localMax_ from child j's already-computed localMax_ --
        // must run for every child before/while folding, so a node's
        // ill-conditioning history correctly reflects everything in its own
        // subtree, not just its own accumulator's center. See file header.
        auto seedFromChild = [&](std::size_t i, std::size_t j) {
            localMax_[i] = std::max(localMax_[i], localMax_[j]);
        };
        auto trackLocal = [&](std::size_t i, Affine const& f) {
            localMax_[i] = std::max(localMax_[i], std::fabs(f.center()));
        };

        auto const aAddFold = [&](std::size_t i) {
            auto acc = pappus::ops::constant<Scalar>(ctx_, Scalar{0});
            for (auto j : Tree::Indices(nodes, i)) {
                seedFromChild(i, j);
                acc = pappus::ops::add<Scalar>(ctx_, acc, aprimal_[j]);
                trackLocal(i, acc);
            }
            return acc;
        };
        auto const aMulFold = [&](std::size_t i) {
            auto acc = pappus::ops::constant<Scalar>(ctx_, Scalar{1});
            for (auto j : Tree::Indices(nodes, i)) {
                seedFromChild(i, j);
                acc = pappus::ops::mul<Scalar>(ctx_, acc, aprimal_[j]);
                trackLocal(i, acc);
            }
            return acc;
        };
        auto const aSubFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(nodes, i)) {
                seedFromChild(i, j);
                if (!acc) { acc = aprimal_[j]; } else { acc = pappus::ops::sub<Scalar>(ctx_, *acc, aprimal_[j]); }
                trackLocal(i, *acc);
            }
            EXPECT(acc.has_value());
            return std::move(*acc);
        };
        auto const aDivFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(nodes, i)) {
                seedFromChild(i, j);
                if (!acc) { acc = aprimal_[j]; } else { acc = pappus::ops::div<Scalar>(ctx_, *acc, aprimal_[j]); }
                trackLocal(i, *acc);
            }
            EXPECT(acc.has_value());
            return std::move(*acc);
        };
        auto const aMinFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(nodes, i)) {
                seedFromChild(i, j);
                if (!acc) { acc = aprimal_[j]; } else { acc = pappus::ops::min<Scalar>(ctx_, *acc, aprimal_[j]); }
                trackLocal(i, *acc);
            }
            EXPECT(acc.has_value());
            return std::move(*acc);
        };
        auto const aMaxFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(nodes, i)) {
                seedFromChild(i, j);
                if (!acc) { acc = aprimal_[j]; } else { acc = pappus::ops::max<Scalar>(ctx_, *acc, aprimal_[j]); }
                trackLocal(i, *acc);
            }
            EXPECT(acc.has_value());
            return std::move(*acc);
        };

        auto const iAddFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(nodes, i)) { acc = pappus::ops::add<Scalar>(acc, iprimal_[j]); }
            return acc;
        };
        auto const iMulFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{1}};
            for (auto j : Tree::Indices(nodes, i)) { acc = pappus::ops::mul<Scalar>(acc, iprimal_[j]); }
            return acc;
        };
        auto const iSubFold = [&](std::size_t i) {
            bool first = true; auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = iprimal_[j]; first = false; } else { acc = pappus::ops::sub<Scalar>(acc, iprimal_[j]); }
            }
            return acc;
        };
        auto const iDivFold = [&](std::size_t i) {
            bool first = true; auto acc = Interval{Scalar{1}};
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = iprimal_[j]; first = false; } else { acc = pappus::ops::div<Scalar>(acc, iprimal_[j]); }
            }
            return acc;
        };
        auto const iMinFold = [&](std::size_t i) {
            bool first = true; auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = iprimal_[j]; first = false; } else { acc = pappus::ops::min<Scalar>(acc, iprimal_[j]); }
            }
            return acc;
        };
        auto const iMaxFold = [&](std::size_t i) {
            bool first = true; auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = iprimal_[j]; first = false; } else { acc = pappus::ops::max<Scalar>(acc, iprimal_[j]); }
            }
            return acc;
        };

        for (std::size_t i = 0; i < n; ++i) {
            auto const& node = nodes[i];
            Scalar v;
            if (node.Optimize) { EXPECT(ci < coeff.size()); v = static_cast<Scalar>(coeff[ci++]); }
            else { v = static_cast<Scalar>(node.Value); }

            Affine a = pappus::ops::constant<Scalar>(ctx_, Scalar{0});
            Interval ivl;
            bool isLeafOrRef = false;

            if (node.Type == NodeType::Constant) {
                // Constant bakes v into the value directly (matches both
                // AffineEvaluator/IntervalEvaluator's own convention) -- no
                // separate weight-apply step for this case.
                a = pappus::ops::constant<Scalar>(ctx_, v);
                ivl = pappus::ops::constant<Scalar>(v);
                localMax_[i] = std::fabs(v);
                isLeafOrRef = true;
            } else if (node.Type == NodeType::Variable) {
                auto it = domains_.find(node.HashValue);
                if (it == domains_.end()) {
                    throw std::runtime_error(fmt::format("HybridEvaluator: no domain bound for variable hash {}", node.HashValue));
                }
                auto const& [lo, hi] = it->second;
                auto cacheIt = variableCache_.find(node.HashValue);
                if (cacheIt == variableCache_.end()) {
                    cacheIt = variableCache_.emplace(node.HashValue, pappus::ops::variable<Scalar>(ctx_, lo, hi)).first;
                }
                a = cacheIt->second;
                if (v != Scalar{1}) { a *= v; }
                localMax_[i] = std::fabs(a.center());
                ivl = pappus::ops::variable<Scalar>(lo, hi) * v;
                isLeafOrRef = true;
            } else if (node.Type == NodeType::Ref) {
                EXPECT(static_cast<std::size_t>(node.RefTo) < i);
                a = aprimal_[node.RefTo];
                ivl = iprimal_[node.RefTo];
                localMax_[i] = localMax_[node.RefTo];
                isLeafOrRef = true;
            } else {
                // Every case below computes RAW (unweighted) a/ivl and seeds
                // localMax_ from children; `v` is applied to both uniformly
                // in exactly one place after the switch (see file header --
                // this used to be per-case and inconsistent between the two
                // backends).
                switch (node.HashValue) {
                case Operon::Hash(BuiltinOp::Add): a = aAddFold(i); ivl = iAddFold(i); break;
                case Operon::Hash(BuiltinOp::Mul): a = aMulFold(i); ivl = iMulFold(i); break;
                case Operon::Hash(BuiltinOp::Sub):
                    if (node.Arity == 1) {
                        seedFromChild(i, i - 1);
                        a = -aprimal_[i - 1];
                        ivl = pappus::ops::neg<Scalar>(iprimal_[i - 1]);
                        trackLocal(i, a);
                    } else {
                        a = aSubFold(i);
                        ivl = iSubFold(i);
                    }
                    break;
                case Operon::Hash(BuiltinOp::Div):
                    if (node.Arity == 1) {
                        seedFromChild(i, i - 1);
                        a = aprimal_[i - 1].inv();
                        ivl = pappus::ops::inv<Scalar>(iprimal_[i - 1]);
                        trackLocal(i, a);
                    } else {
                        a = aDivFold(i);
                        ivl = iDivFold(i);
                    }
                    break;
                case Operon::Hash(BuiltinOp::Fmin): a = aMinFold(i); ivl = iMinFold(i); break;
                case Operon::Hash(BuiltinOp::Fmax): a = aMaxFold(i); ivl = iMaxFold(i); break;
                default:
                    if (node.Arity == 1) {
                        auto const* aunary = AffineUnaryRules().TryGet(node.HashValue);
                        auto const* iunary = IntervalUnaryRules().TryGet(node.HashValue);
                        if (aunary && iunary) {
                            seedFromChild(i, i - 1);
                            a = (*aunary)(ctx_, aprimal_[i - 1]);
                            ivl = (*iunary)(iprimal_[i - 1]);
                            trackLocal(i, a);
                            break;
                        }
                    } else if (node.Arity == 2) {
                        auto const j = i - 1;
                        auto const k = j - (nodes[j].Length + 1);
                        auto const* abinary = AffineBinaryRules().TryGet(node.HashValue);
                        auto const* ibinary = IntervalBinaryRules().TryGet(node.HashValue);
                        if (abinary && ibinary) {
                            seedFromChild(i, j);
                            seedFromChild(i, k);
                            a = (*abinary)(ctx_, aprimal_[j], aprimal_[k]);
                            ivl = (*ibinary)(iprimal_[j], iprimal_[k]);
                            trackLocal(i, a);
                            break;
                        }
                    }
                    throw std::runtime_error(fmt::format("HybridEvaluator: node kind `{}` not yet mapped", node.Name()));
                }
                if (v != Scalar{1}) { a *= v; }
                ivl = ivl * v;
                localMax_[i] = std::max(localMax_[i], std::fabs(a.center()));
            }

            if (!isLeafOrRef) {
                ++internalCount_;
                // --- Collapse decision (only for internal, non-leaf/Ref nodes) ---
                // Domain-error safety: never compare with NaN/empty. If affine
                // is invalid, or interval is empty/non-finite, do not attempt a
                // numeric collapse comparison -- keep affine's own (already
                // NaN-poisoned, if applicable) convention. See design doc gap 1.
                auto aiv = a.to_interval();
                bool const aValid = std::isfinite(aiv.inf()) && std::isfinite(aiv.sup());
                bool const iValid = !ivl.is_empty() && std::isfinite(ivl.inf()) && std::isfinite(ivl.sup());

                bool certified = aValid;
                if (certified) {
                    auto const r = a.radius();
                    auto const impliedErrorFloor = localMax_[i] * std::numeric_limits<Scalar>::epsilon();
                    if (r > Scalar{0} && impliedErrorFloor > IllConditionedThreshold() * r) { certified = false; }
                }

                bool shouldCollapse = false;
                if (!aValid && iValid) {
                    shouldCollapse = true; // affine unusable, interval is our only option
                } else if (!certified && iValid) {
                    shouldCollapse = true; // per design doc: force collapse on uncertified affine
                } else if (aValid && iValid) {
                    auto const aw = static_cast<Scalar>(aiv.sup() - aiv.inf());
                    auto const iw = static_cast<Scalar>(ivl.sup() - ivl.inf());
                    auto const margin = CollapseMargin();
                    if (iw < aw * (Scalar{1} - margin)) { shouldCollapse = true; }
                }
                // !aValid && !iValid: both backends agree this node is a domain
                // error; keep affine's own invalid()/NaN-poisoned convention,
                // do not collapse into an equally-invalid interval-derived form.

                if (shouldCollapse) {
                    ++collapseCount_;
                    // Fresh, independent noise symbol -- see design doc "Why
                    // this is sound." `ivl` already has `v` applied above
                    // (exactly once), so no double-scaling here. Resetting
                    // localMax_[i] to just the replacement's own center is
                    // deliberate, not a bug: the replacement is a genuinely
                    // fresh, sound value with none of the discarded form's
                    // residual numerical risk, so nothing downstream needs
                    // to remember the pre-collapse history.
                    a = Affine(ctx_.state, ivl);
                    localMax_[i] = std::fabs(a.center());
                }
            }

            aprimal_.push_back(std::move(a));
            iprimal_[i] = ivl;
        }
        return aprimal_.back();
    }

private:
    gsl::not_null<Operon::Tree const*> tree_;
    DomainMap domains_;
    mutable Context ctx_;
    mutable std::vector<Affine> aprimal_;
    mutable std::vector<Interval> iprimal_;
    mutable std::vector<Scalar> localMax_;
    mutable Operon::Map<Operon::Hash, Affine> variableCache_;
    mutable std::size_t collapseCount_{0};
    mutable std::size_t internalCount_{0};
};

} // namespace Operon

#endif
