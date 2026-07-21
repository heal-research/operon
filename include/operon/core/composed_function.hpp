// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_COMPOSED_FUNCTION_HPP
#define OPERON_CORE_COMPOSED_FUNCTION_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fmt/format.h>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dispatch.hpp"
#include "node.hpp"
#include "pset.hpp"
#include "symbol_library.hpp"
#include "tree.hpp"
#include "tree_diff.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/parser/infix.hpp"

// Derives DispatchTable Callable<T>/CallableDiff<T> entries for a composed
// function (see operon-planning/designs/composed-functions.md) from its
// already-parsed body Tree, by walking the body's own private node buffer
// and recursing into each constituent built-in's own already-registered
// Callable/CallableDiff — no reimplementation of the underlying math.
//
// v1 scope: body Trees only ever reference built-ins (ParseFunctionBody's
// grammar), so every Function node's Callable/CallableDiff is guaranteed
// resolvable against `dt` at construction time.

namespace Operon {

namespace detail {
    // Resolve each unique built-in hash appearing in `body` to its Callable
    // (or CallableDiff) from `dt`, once, by value — the derived
    // Callable/CallableDiff must not hold a reference/pointer to `dt`
    // itself (which could dangle or go stale if the table is later copied),
    // same "don't bind to one specific object instance" principle as the
    // rejected dataset-variable-inside-a-body idea.
    template<typename Fn, typename DTable, typename Getter>
    auto ResolveOps(DTable const& dt, Tree const& body, Getter&& get) -> Operon::Map<Operon::Hash, Fn>
    {
        Operon::Map<Operon::Hash, Fn> ops;
        for (auto const& n : body.Nodes()) {
            if (n.Type == Operon::NodeType::Function && !ops.contains(n.HashValue)) {
                ops.emplace(n.HashValue, get(dt, n.HashValue));
            }
        }
        return ops;
    }

    // Binds the composed function's formal-parameter index to the outer
    // call site's actual argument node index. Tree::Indices yields children
    // nearest-first (the *last*-pushed/last-written argument first — see
    // e.g. MakeBinaryCallable's "first child is at i-1" comment in
    // symbol_library.hpp, and Dispatch::BinaryOp/NaryOp, which all consume
    // it in that same near-to-far order without reinterpreting it as
    // textual order). params[0] is the *first textually-written* formal
    // parameter, i.e. the *farthest*/earliest-pushed child — the reverse of
    // Tree::Indices' own enumeration order. Confirmed empirically (not just
    // by re-deriving the parse convention): an order-sensitive composed
    // body (`a - b`) evaluated against `sub2(x, y)` must equal `x - y`, and
    // only reversing Tree::Indices' order here produces that.
    inline auto BindArgIndices(Operon::Vector<Node> const& outerNodes, std::size_t i, std::size_t arity)
        -> std::array<std::int64_t, kMaxComposedFunctionArity>
    {
        std::array<std::int64_t, kMaxComposedFunctionArity> childIdx{};
        std::size_t k = 0;
        for (auto j : Tree::Indices(outerNodes, i)) {
            if (k == arity) { break; }
            childIdx[k++] = static_cast<std::int64_t>(j);
        }
        std::reverse(childIdx.begin(), childIdx.begin() + static_cast<std::ptrdiff_t>(arity));
        return childIdx;
    }
} // namespace detail

// A v1 composed-function body only ever references built-ins (see
// ParseFunctionBody), but not every built-in has symbolic-diff coverage —
// tree_diff.cpp explicitly leaves Aq/Powabs/Fmin/Fmax and Abs/Sqrtabs/
// Floor/Ceil undifferentiated (silently degrading to a Zero gradient
// through the JIT path). Registering a composed function whose body uses
// one of these would silently produce a model whose coefficients under
// that composed function never move when fit via the JIT path — exactly
// the "silent wrong gradient" failure class this design has avoided
// everywhere else. Policy (design-doc decided): reject at registration.
inline auto HasSymbolicDerivCoverage(Operon::Node const& n) -> bool
{
    if (n.IsLeaf()) { return true; } // param/constant leaves: trivial, always covered
    // Mirrors Deriv()'s own hash-based dispatch order exactly (tree_diff.cpp)
    // — arity must NOT be checked first: Sub/Div both have real arity==1
    // rules (unary minus, reciprocal) special-cased by hash before Deriv()
    // ever reaches its generic "arity==1, consult the unary registry"
    // fallback, so checking arity==1 up front would wrongly flag them as
    // uncovered.
    if (n.IsAddition() || n.IsMultiplication()) { return true; } // n-ary, always covered
    if (n.IsSubtraction()) { return true; } // arity 1 (unary minus), 2, and >2 all handled
    if (n.IsDivision()) { return n.Arity <= 2; } // arity>2 not yet supported, per Deriv()'s own comment
    if (n.IsPow()) { return true; } // binary-only from infix grammar
    if (n.IsAq() || n.IsPowabs() || n.IsOp<Operon::BuiltinOp::Fmin, Operon::BuiltinOp::Fmax>()) { return false; }
    if (n.Arity == 1) { return Operon::HasUnarySymbolicDeriv(n.HashValue); }
    return false; // matches Deriv()'s final fallthrough `return Zero;`
}

// Throws std::invalid_argument naming the first offending node's hash if
// `body` references any built-in lacking symbolic-diff coverage. Call this
// at composed-function registration time, before deriving/registering the
// backend hooks — never silently degrade.
inline void ValidateSymbolicDiffCoverage(Tree const& body)
{
    for (auto const& n : body.Nodes()) {
        if (!HasSymbolicDerivCoverage(n)) {
            throw std::invalid_argument(fmt::format(
                "composed function body references a built-in (hash {}) with no symbolic-diff rule — "
                "its JIT-path gradient would silently be zero",
                n.HashValue));
        }
    }
}

namespace detail {
    // Hand-rolled port of tree_diff.cpp's private hash-consing convention
    // (Mix/Push/AppendRef/GetConst/MakeUnary/MakeBinary — anonymous-
    // namespace internals there, not exported) so a composed function's
    // symbolic-derivative rule can build/extend the *same* live dag/memo/h
    // Deriv() threads through the outer tree's own differentiation, using
    // only public Node factories — the same "external caller hand-rolls the
    // convention" precedent test/source/implementation/user_defined_registries.cpp
    // already established. Matching Mix/Push exactly, rather than inventing
    // a different scheme, keeps hash-consing behavior between
    // composed-function-injected nodes and the rest of the dag consistent.
    using DiffNodes  = Operon::Vector<Node>;
    using DiffHashes = Operon::Vector<Operon::Hash>;
    using DiffMemo   = Operon::Map<Operon::Hash, std::size_t>;
    inline constexpr std::size_t DiffZero = std::numeric_limits<std::size_t>::max();

    // Salted relative to tree_diff.cpp's own Mix(): that version is safe
    // within tree_diff.cpp itself because Deriv() only ever combines
    // *derivative* dag indices this way (fresh nodes, whose own hashes are
    // already mixed), never a raw original-tree index directly — but
    // DiffCopyBody (below) *does* mix directly against tree_diff.cpp's
    // `h[i] = i` identity hashes for the original tree's prefix (needed to
    // reconstruct the composed body's primal value inline). `h[0] == 0`
    // there is completely normal, but combined with `BuiltinOp::Add == 0`
    // it makes the unsalted Mix(0,0) == 0 a real fixed point: found by
    // direct reproduction, not by inspection — `exp(x+x+x)` composed and
    // differentiated came out as `3*exp(2x)` instead of `3*exp(3x)`,
    // because Mix(Add, h[0]=0) collapsed to exactly 0 and stayed 0 through
    // a second Add combining with it, causing DiffMakeBinary's memo to
    // spuriously treat `Add(innerSum, x)` as identical to `Add(x, x)`. The
    // salt breaks that fixed point without needing bit-for-bit agreement
    // with tree_diff.cpp's own Mix (the two memo maps are never compared
    // against each other, so nothing requires the formulas to match).
    inline auto DiffMix(std::uint64_t a, std::uint64_t b) -> std::uint64_t
    {
        a += 0x9e3779b97f4a7c15ULL;
        return a ^ (b * 0x9e3779b97f4a7c15ULL + (a << 6U) + (a >> 2U));
    }

    inline void DiffPush(DiffNodes& dag, DiffHashes& h, Node n, std::uint64_t hash)
    {
        n.Optimize = false;
        dag.push_back(n);
        h.push_back(hash);
    }

    inline auto DiffAppendRef(DiffNodes& dag, DiffHashes& h, std::size_t target) -> std::size_t
    {
        auto hash = DiffMix(static_cast<std::uint64_t>(Operon::NodeType::Ref), h[target]);
        DiffPush(dag, h, Node::Ref(static_cast<std::uint16_t>(target)), hash);
        return dag.size() - 1;
    }

    inline auto DiffGetConst(DiffNodes& dag, DiffMemo& memo, DiffHashes& h, Operon::Scalar val) -> std::size_t
    {
        auto hash = DiffMix(static_cast<std::uint64_t>(Operon::NodeType::Constant),
            std::bit_cast<std::uint64_t>(static_cast<double>(val)));
        if (auto it = memo.find(hash); it != memo.end()) { return it->second; }
        auto idx = dag.size();
        DiffPush(dag, h, Node::Constant(val), hash);
        memo.insert_or_assign(hash, idx);
        return idx;
    }

    inline auto DiffMakeUnary(DiffNodes& dag, DiffMemo& memo, DiffHashes& h, Operon::BuiltinOp op, std::size_t a) -> std::size_t
    {
        auto opHash = DiffMix(static_cast<std::uint64_t>(op), h[a]);
        if (auto it = memo.find(opHash); it != memo.end()) { return it->second; }
        DiffAppendRef(dag, h, a);
        auto n = Node::Function(static_cast<Operon::Hash>(op), 1);
        auto idx = dag.size();
        DiffPush(dag, h, n, opHash);
        memo.insert_or_assign(opHash, idx);
        return idx;
    }

    inline auto DiffMakeBinary(DiffNodes& dag, DiffMemo& memo, DiffHashes& h, Operon::BuiltinOp op, std::size_t a, std::size_t b) -> std::size_t
    {
        auto opHash = DiffMix(DiffMix(static_cast<std::uint64_t>(op), h[a]), h[b]);
        if (auto it = memo.find(opHash); it != memo.end()) { return it->second; }
        DiffAppendRef(dag, h, b);
        DiffAppendRef(dag, h, a);
        auto n = Node::Function(static_cast<Operon::Hash>(op), 2);
        auto idx = dag.size();
        DiffPush(dag, h, n, opHash);
        memo.insert_or_assign(opHash, idx);
        return idx;
    }

    // Copies bodyNodes[k]'s structure into the live dag once, memoized in
    // `bodyToLive` (postfix order — children already copied by the time a
    // parent is reached). A param leaf isn't copied at all: it maps
    // directly to the live outer child index already sitting in `dag`
    // (that live subtree literally *is* the value of that parameter — no
    // new node needed, matching MakeComposedCallable's own "substitute, don't
    // recompute" approach).
    inline void DiffCopyBody(DiffNodes& dag, DiffMemo& memo, DiffHashes& h,
        Operon::Vector<Node> const& bodyNodes,
        std::array<std::int64_t, kMaxComposedFunctionArity> const& liveChildIdx,
        std::vector<std::size_t>& bodyToLive)
    {
        for (std::size_t k = 0; k < bodyNodes.size(); ++k) {
            auto const& n = bodyNodes[k];
            if (n.IsVariable()) {
                auto const pIdx = static_cast<std::size_t>(n.HashValue - Operon::BuiltinOpCount);
                bodyToLive[k] = static_cast<std::size_t>(liveChildIdx[pIdx]);
            } else if (n.Type == Operon::NodeType::Constant) {
                bodyToLive[k] = DiffGetConst(dag, memo, h, n.Value);
            } else {
                auto const op = static_cast<Operon::BuiltinOp>(n.HashValue);
                // Resolve children via Tree::Indices (already-established
                // near/far convention — see BindArgIndices) and translate
                // through bodyToLive.
                Operon::Vector<std::size_t> children;
                for (auto c : Tree::Indices(bodyNodes, k)) { children.push_back(bodyToLive[c]); }
                if (n.Arity == 1) {
                    bodyToLive[k] = DiffMakeUnary(dag, memo, h, op, children[0]);
                } else {
                    // near=children[0], far=children[1] — matches DiffMakeBinary's
                    // own (a=near,b=far) convention directly, no reordering.
                    bodyToLive[k] = DiffMakeBinary(dag, memo, h, op, children[0], children[1]);
                }
            }
        }
    }

    // Mirrors tree_diff.cpp's Deriv() structure exactly (Add/Mul/Sub/Div/Pow
    // hardcoded, unary ops via the existing GetUnarySymbolicDeriv registry),
    // but differentiates w.r.t. a *parameter* rather than a variable's
    // weight: d(param_k)/d(param_k) = 1 (a plain constant), not the
    // "weight-of-a-variable" rule Deriv() uses for ordinary Optimize=true
    // Variable leaves — see the design doc's Fix-2 for why conflating these
    // two would compute the wrong derivative.
    inline auto DiffParam(DiffNodes& dag, DiffMemo& memo, DiffHashes& h,
        Operon::Vector<Node> const& bodyNodes, std::vector<std::size_t> const& bodyToLive,
        std::size_t k, std::size_t targetParam) -> std::size_t
    {
        auto const& n = bodyNodes[k];

        if (n.IsVariable()) {
            auto const pIdx = static_cast<std::size_t>(n.HashValue - Operon::BuiltinOpCount);
            return pIdx == targetParam ? DiffGetConst(dag, memo, h, Operon::Scalar{1}) : DiffZero;
        }
        if (n.Type == Operon::NodeType::Constant) { return DiffZero; }

        Operon::Vector<std::size_t> children;
        for (auto c : Tree::Indices(bodyNodes, k)) { children.push_back(c); }
        auto const arity = children.size();

        if (n.IsAddition()) {
            Operon::Vector<std::size_t> terms;
            for (auto c : children) {
                auto dc = DiffParam(dag, memo, h, bodyNodes, bodyToLive, c, targetParam);
                if (dc != DiffZero) { terms.push_back(dc); }
            }
            if (terms.empty()) { return DiffZero; }
            auto result = terms[0];
            for (std::size_t m = 1; m < terms.size(); ++m) { result = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Add, result, terms[m]); }
            return result;
        }
        if (n.IsMultiplication()) {
            Operon::Vector<std::size_t> terms;
            for (std::size_t m = 0; m < arity; ++m) {
                auto dm = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[m], targetParam);
                if (dm == DiffZero) { continue; }
                std::size_t prod = DiffZero;
                for (std::size_t l = 0; l < arity; ++l) {
                    if (l == m) { continue; }
                    prod = (prod == DiffZero) ? bodyToLive[children[l]] : DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, prod, bodyToLive[children[l]]);
                }
                terms.push_back(prod == DiffZero ? dm : DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, dm, prod));
            }
            if (terms.empty()) { return DiffZero; }
            auto result = terms[0];
            for (std::size_t m = 1; m < terms.size(); ++m) { result = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Add, result, terms[m]); }
            return result;
        }
        if (n.IsSubtraction()) {
            auto dj = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[0], targetParam);
            if (arity == 1) {
                if (dj == DiffZero) { return DiffZero; }
                auto neg1 = DiffGetConst(dag, memo, h, Operon::Scalar{-1});
                return DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, neg1, dj);
            }
            // arity == 2 (v1 infix grammar never produces arity > 2 here)
            auto dk = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[1], targetParam);
            if (dj == DiffZero && dk == DiffZero) { return DiffZero; }
            if (dk == DiffZero) { return dj; }
            if (dj == DiffZero) {
                auto neg1 = DiffGetConst(dag, memo, h, Operon::Scalar{-1});
                return DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, neg1, dk);
            }
            return DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Sub, dj, dk);
        }
        if (n.IsDivision()) {
            if (arity == 1) {
                auto dj = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[0], targetParam);
                if (dj == DiffZero) { return DiffZero; }
                auto j    = bodyToLive[children[0]];
                auto j2   = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, j, j);
                auto neg1 = DiffGetConst(dag, memo, h, Operon::Scalar{-1});
                auto negDj = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, neg1, dj);
                return DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Div, negDj, j2);
            }
            // a/b: d = (da*b - a*db) / b^2 — arity == 2 only (v1 grammar)
            auto j  = bodyToLive[children[0]];
            auto k2 = bodyToLive[children[1]];
            auto dj = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[0], targetParam);
            auto dk = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[1], targetParam);
            if (dj == DiffZero && dk == DiffZero) { return DiffZero; }
            if (dk == DiffZero) { return DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Div, dj, k2); }
            auto denom = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, k2, k2);
            std::size_t num = DiffZero;
            if (dj != DiffZero) { num = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, dj, k2); }
            {
                auto term = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, j, dk);
                if (num == DiffZero) {
                    auto neg1 = DiffGetConst(dag, memo, h, Operon::Scalar{-1});
                    num = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, neg1, term);
                } else {
                    num = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Sub, num, term);
                }
            }
            return DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Div, num, denom);
        }
        if (n.IsPow()) {
            auto j  = bodyToLive[children[0]];
            auto k2 = bodyToLive[children[1]];
            auto dj = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[0], targetParam);
            auto dk = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[1], targetParam);
            if (dj == DiffZero && dk == DiffZero) { return DiffZero; }
            Operon::Vector<std::size_t> terms;
            if (dj != DiffZero) {
                auto ki   = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, k2, bodyToLive[k]);
                auto term = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Div, ki, j);
                terms.push_back(DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, dj, term));
            }
            if (dk != DiffZero) {
                auto logJ = DiffMakeUnary(dag, memo, h, Operon::BuiltinOp::Log, j);
                auto t    = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, bodyToLive[k], logJ);
                terms.push_back(DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, dk, t));
            }
            if (terms.empty()) { return DiffZero; }
            auto result = terms[0];
            for (std::size_t m = 1; m < terms.size(); ++m) { result = DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Add, result, terms[m]); }
            return result;
        }
        // unary built-in with registered symbolic-diff coverage (guaranteed
        // by ValidateSymbolicDiffCoverage at registration time — Aq/Powabs/
        // Fmin/Fmax and unregistered unary ops never reach this point).
        auto dj = DiffParam(dag, memo, h, bodyNodes, bodyToLive, children[0], targetParam);
        if (dj == DiffZero) { return DiffZero; }
        auto const* rule = Operon::GetUnarySymbolicDeriv(n.HashValue);
        auto fp = (*rule)(dag, memo, h, bodyToLive[k], bodyToLive[children[0]]);
        if (fp == DiffZero) { return DiffZero; }
        return DiffMakeBinary(dag, memo, h, Operon::BuiltinOp::Mul, fp, dj);
    }
} // namespace detail

// Builds a symbolic-derivative rule for an *arity-1* composed function,
// pluggable directly into the existing Operon::RegisterUnarySymbolicDeriv
// registry — no core tree_diff.cpp changes needed for this arity (Deriv()
// already consults that registry for every unary node). Differentiation is
// re-run fresh on every invocation (once per occurrence in whatever outer
// tree is being differentiated, at dag-construction time — not a per-row
// hot path) rather than precomputed-once-and-grafted: the body is small, so
// the simpler "just recompute" approach costs nothing measurable and avoids
// the added complexity of template-dag index-substitution.
//
inline auto MakeComposedUnarySymbolicDerivRule(Tree const& body) -> UnarySymbolicDerivRule
{
    auto const& bodyNodes = body.Nodes();
    return [bodyNodes](detail::DiffNodes& dag, detail::DiffMemo& memo, detail::DiffHashes& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
        std::array<std::int64_t, kMaxComposedFunctionArity> liveChildIdx{};
        liveChildIdx[0] = static_cast<std::int64_t>(j);
        std::vector<std::size_t> bodyToLive(bodyNodes.size());
        detail::DiffCopyBody(dag, memo, h, bodyNodes, liveChildIdx, bodyToLive);
        return detail::DiffParam(dag, memo, h, bodyNodes, bodyToLive, bodyNodes.size() - 1, 0);
    };
}

// Binary counterpart, pluggable into Operon::RegisterBinarySymbolicDeriv
// (a new registry, wired into tree_diff.cpp's Deriv() arity==2 fallthrough
// right after the Aq/Powabs/Fmin/Fmax exclusion block — Deriv() itself
// otherwise unmodified). Same "recompute fresh on every invocation, no
// template-dag grafting" simplification as the unary rule.
//
// j = near child (i-1), k = far child — Deriv()'s own convention, same as
// Pow's hardcoded case. This is the *reverse* of textual/formal-parameter
// order: param[0] (textually first) binds to the far child, param[1] binds
// to the near child — same reversal BindArgIndices applies for the numeric
// Callable derivation, confirmed empirically there (Tree::Indices
// enumerates nearest-first). Getting this backwards here would silently
// swap ∂f/∂param0 and ∂f/∂param1.
inline auto MakeComposedBinarySymbolicDerivRule(Tree const& body) -> BinarySymbolicDerivRule
{
    auto const& bodyNodes = body.Nodes();
    return [bodyNodes](detail::DiffNodes& dag, detail::DiffMemo& memo, detail::DiffHashes& h,
               std::size_t /*i*/, std::size_t j, std::size_t k) -> std::pair<std::size_t, std::size_t> {
        std::array<std::int64_t, kMaxComposedFunctionArity> liveChildIdx{};
        liveChildIdx[0] = static_cast<std::int64_t>(k); // param[0] = far child
        liveChildIdx[1] = static_cast<std::int64_t>(j); // param[1] = near child
        std::vector<std::size_t> bodyToLive(bodyNodes.size());
        detail::DiffCopyBody(dag, memo, h, bodyNodes, liveChildIdx, bodyToLive);
        auto const root = bodyNodes.size() - 1;
        auto fpk = detail::DiffParam(dag, memo, h, bodyNodes, bodyToLive, root, 0); // ∂body/∂param[0] = ∂f/∂k
        auto fpj = detail::DiffParam(dag, memo, h, bodyNodes, bodyToLive, root, 1); // ∂body/∂param[1] = ∂f/∂j
        return {fpj, fpk};
    };
}

template<typename DTable, typename T>
auto MakeComposedCallable(DTable const& dt, Tree const& body, std::size_t arity) -> typename DTable::template Callable<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    using Callable = typename DTable::template Callable<T>;

    auto ops = detail::ResolveOps<Callable>(dt, body, [](DTable const& t, Operon::Hash h) { return t.template GetFunction<T>(h); });
    auto const& bodyNodes = body.Nodes();

    return [bodyNodes, ops = std::move(ops), arity](
               Operon::Vector<Node> const& outerNodes,
               Backend::View<T, S> outerData,
               std::size_t i,
               Operon::Range rg) {
        auto const nNodes = static_cast<std::int64_t>(bodyNodes.size());
        auto const childIdx = detail::BindArgIndices(outerNodes, i, arity);

        Backend::Buffer<T, S> buf(S, nNodes);
        Backend::View<T, S> view{buf};

        for (std::int64_t k = 0; k < nNodes; ++k) {
            auto const& n = bodyNodes[static_cast<std::size_t>(k)];
            auto* dst = Backend::Ptr<T, S>(view, k);

            if (n.IsRef()) {
                auto const* src = Backend::Ptr<T, S>(view, static_cast<std::int64_t>(n.RefTo));
                std::copy_n(src, S, dst);
                continue;
            }
            if (n.Type == Operon::NodeType::Constant) {
                std::fill_n(dst, S, static_cast<T>(n.Value));
                continue;
            }
            if (n.IsVariable()) {
                // param leaf: HashValue = ParamHash(paramIdx); read the
                // corresponding already-computed argument value straight
                // from the outer buffer (weight is always 1 on a
                // freshly-parsed param leaf — see ParseFunctionBody — but
                // multiply anyway rather than assume, in case that ever
                // changes).
                auto const paramIdx = static_cast<std::size_t>(n.HashValue - Operon::BuiltinOpCount);
                auto const* src = Backend::Ptr<T, S>(outerData, childIdx[paramIdx]);
                auto const w = static_cast<T>(n.Value);
                for (std::size_t s = 0; s < S; ++s) { dst[s] = w * src[s]; }
                continue;
            }
            // Function node: every name in a v1 body is a built-in, already
            // resolved into `ops` above.
            std::invoke(ops.at(n.HashValue), bodyNodes, view, static_cast<std::size_t>(k), rg);
        }

        auto* dst = Backend::Ptr<T, S>(outerData, static_cast<std::int64_t>(i));
        auto const* src = Backend::Ptr<T, S>(view, nNodes - 1);
        auto const w = static_cast<T>(outerNodes[i].Value);
        for (std::size_t s = 0; s < S; ++s) { dst[s] = w * src[s]; }
    };
}

template<typename DTable, typename T>
auto MakeComposedCallableDiff(DTable const& dt, Tree const& body, std::size_t arity) -> typename DTable::template CallableDiff<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    using Callable = typename DTable::template Callable<T>;
    using CallableDiff = typename DTable::template CallableDiff<T>;

    auto fns = detail::ResolveOps<Callable>(dt, body, [](DTable const& t, Operon::Hash h) { return t.template GetFunction<T>(h); });
    auto dfs = detail::ResolveOps<CallableDiff>(dt, body, [](DTable const& t, Operon::Hash h) { return t.template GetDerivative<T>(h); });
    auto const& bodyNodes = body.Nodes();

    return [bodyNodes, fns = std::move(fns), dfs = std::move(dfs), arity](
               Operon::Vector<Node> const& outerNodes,
               Backend::View<T const, S> outerPrimal,
               Backend::View<T, S> outerTrace,
               int i,
               int j) {
        // Redeclared rather than captured: DTable/T are the enclosing
        // function template's own type parameters, visible here with no
        // capture needed (only values need capturing, not types) — this
        // sidesteps a real Clang/GCC disagreement over whether this
        // specific constexpr capture is "needed" (one requires it, the
        // other errors on its presence as unused; the clang build under
        // test even crashed outright on one capture-syntax variant tried
        // while chasing this) by not capturing it at all.
        constexpr auto S = DTable::template BatchSize<T>;
        auto const nNodes = static_cast<std::int64_t>(bodyNodes.size());
        auto const childIdx = detail::BindArgIndices(outerNodes, static_cast<std::size_t>(i), arity);

        std::size_t paramIdx = 0;
        for (std::size_t k = 0; k < arity; ++k) {
            if (childIdx[k] == j) { paramIdx = k; break; }
        }

        // Forward pass: re-derive the body's own primal values, seeded from
        // the outer buffer's already-computed argument value at childIdx[*]
        // (the *outer* primal, not a fresh recomputation — same "reuse the
        // caller's already-computed form" principle as the affine/interval
        // correlation fix in the design, applied here to keep this
        // self-contained rather than depending on the outer Callable's own
        // private buffer, which isn't visible here).
        Backend::Buffer<T, S> primalBuf(S, nNodes);
        Backend::View<T, S> primalView{primalBuf};
        for (std::int64_t k = 0; k < nNodes; ++k) {
            auto const& n = bodyNodes[static_cast<std::size_t>(k)];
            auto* dst = Backend::Ptr<T, S>(primalView, k);
            if (n.IsRef()) {
                auto const* src = Backend::Ptr<T, S>(primalView, static_cast<std::int64_t>(n.RefTo));
                std::copy_n(src, S, dst);
            } else if (n.Type == Operon::NodeType::Constant) {
                std::fill_n(dst, S, static_cast<T>(n.Value));
            } else if (n.IsVariable()) {
                auto const pIdx = static_cast<std::size_t>(n.HashValue - Operon::BuiltinOpCount);
                auto const* src = Backend::Ptr<T const, S>(outerPrimal, childIdx[pIdx]);
                auto const w = static_cast<T>(n.Value);
                for (std::size_t s = 0; s < S; ++s) { dst[s] = w * src[s]; }
            } else {
                std::invoke(fns.at(n.HashValue), bodyNodes, primalView, static_cast<std::size_t>(k), Operon::Range{0, S});
            }
        }

        // Reverse pass: seed the body root's trace to 1, walk backward,
        // consulting each constituent op's own already-registered
        // CallableDiff for the *unweighted* local derivative (matching the
        // confirmed-correct Log/Sin convention, never the buggy
        // Mul/Exp-style shortcut through a node's own weighted primal — see
        // the double-weighted-derivative bug writeup), then applying that
        // node's own weight once while propagating, exactly mirroring
        // Interpreter::ReverseTraceGeneric.
        Backend::Buffer<T, S> traceBuf(S, nNodes);
        Backend::View<T, S> traceView{traceBuf};
        std::fill_n(traceBuf.data(), static_cast<std::size_t>(S * nNodes), T{0});
        std::fill_n(Backend::Ptr<T, S>(traceView, nNodes - 1), S, T{1});

        for (std::int64_t k = nNodes - 1; k >= 0; --k) {
            auto const& n = bodyNodes[static_cast<std::size_t>(k)];
            if (n.IsRef()) {
                auto* dst = Backend::Ptr<T, S>(traceView, static_cast<std::int64_t>(n.RefTo));
                auto const* src = Backend::Ptr<T, S>(traceView, k);
                for (std::size_t s = 0; s < S; ++s) { dst[s] += src[s]; }
                continue;
            }
            if (n.IsLeaf()) { continue; }

            auto const w = static_cast<T>(n.Value);
            for (auto c : Tree::Indices(bodyNodes, static_cast<std::size_t>(k))) {
                std::invoke(dfs.at(n.HashValue), bodyNodes, primalView, traceView, static_cast<int>(k), static_cast<int>(c));
            }
            for (auto c : Tree::Indices(bodyNodes, static_cast<std::size_t>(k))) {
                auto* dst = Backend::Ptr<T, S>(traceView, static_cast<std::int64_t>(c));
                auto const* upstream = Backend::Ptr<T, S>(traceView, k);
                for (std::size_t s = 0; s < S; ++s) { dst[s] *= upstream[s] * w; }
            }
        }

        // Sum every occurrence of this param in the body (repeated textual
        // uses are separate leaf nodes, not deduplicated via Ref — see
        // ParseFunctionBody, reduce=false).
        std::array<T, S> sum{};
        std::fill(sum.begin(), sum.end(), T{0});
        for (std::int64_t k = 0; k < nNodes; ++k) {
            auto const& n = bodyNodes[static_cast<std::size_t>(k)];
            if (!n.IsVariable()) { continue; }
            if (static_cast<std::size_t>(n.HashValue - Operon::BuiltinOpCount) != paramIdx) { continue; }
            auto const* src = Backend::Ptr<T, S>(traceView, k);
            for (std::size_t s = 0; s < S; ++s) { sum[s] += src[s]; }
        }

        auto* dst = Backend::Ptr<T, S>(outerTrace, j);
        std::copy(sum.begin(), sum.end(), dst);
    };
}

// Builds an IntervalUnaryFn for an *arity-1* composed function, pluggable
// directly into RegisterUnaryInterval — walks the body's own node array
// once per call (not a nested IntervalEvaluator pass), substituting the
// single param leaf with the literal caller-supplied argument interval and
// dispatching internal nodes through the existing IntervalUnaryRules/
// IntervalBinaryRules registries plus the hardcoded Add/Mul/Sub/Div/Fmin/Fmax
// n-ary folds — mirrors IntervalEvaluator::Evaluate()'s own per-node
// dispatch exactly (see interval_evaluator.hpp), just body-scoped.
inline auto MakeComposedIntervalUnaryFn(Tree const& body) -> IntervalUnaryFn
{
    auto const& bodyNodes = body.Nodes();
    return [bodyNodes](pappus::interval<Operon::Scalar> const& arg) -> pappus::interval<Operon::Scalar> {
        using Scalar = Operon::Scalar;
        using Interval = pappus::interval<Scalar>;
        RegisterIntervalBuiltins();

        auto const n = bodyNodes.size();
        std::vector<Interval> primal(n);

        auto const addFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(bodyNodes, i)) { acc = pappus::ops::add<Scalar>(acc, primal[j]); }
            return acc;
        };
        auto const mulFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{1}};
            for (auto j : Tree::Indices(bodyNodes, i)) { acc = pappus::ops::mul<Scalar>(acc, primal[j]); }
            return acc;
        };
        auto const subFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (first) { acc = primal[j]; first = false; } else { acc = pappus::ops::sub<Scalar>(acc, primal[j]); }
            }
            return acc;
        };
        auto const divFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{1}};
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (first) { acc = primal[j]; first = false; } else { acc = pappus::ops::div<Scalar>(acc, primal[j]); }
            }
            return acc;
        };
        auto const minFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (first) { acc = primal[j]; first = false; } else { acc = pappus::ops::min<Scalar>(acc, primal[j]); }
            }
            return acc;
        };
        auto const maxFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (first) { acc = primal[j]; first = false; } else { acc = pappus::ops::max<Scalar>(acc, primal[j]); }
            }
            return acc;
        };

        for (std::size_t i = 0; i < n; ++i) {
            auto const& node = bodyNodes[i];
            // Body-internal constants are always Optimize=false (forced by
            // ParseFunctionBody) — no coefficient span to consult, always
            // node.Value directly.
            auto const v = static_cast<Scalar>(node.Value);

            if (node.Type == Operon::NodeType::Constant) {
                primal[i] = pappus::ops::constant<Scalar>(v);
            } else if (node.IsVariable()) {
                // Single param leaf (arity-1 scope): the literal caller-
                // supplied interval, not a fresh domain lookup — same object
                // reuse principle as the affine-correlation fix (Fix 1).
                primal[i] = arg * v;
            } else if (node.IsRef()) {
                primal[i] = primal[node.RefTo];
            } else {
                switch (node.HashValue) {
                case Operon::Hash(Operon::BuiltinOp::Add): primal[i] = addFold(i) * v; break;
                case Operon::Hash(Operon::BuiltinOp::Mul): primal[i] = mulFold(i) * v; break;
                case Operon::Hash(Operon::BuiltinOp::Sub):
                    primal[i] = (node.Arity == 1 ? pappus::ops::neg<Scalar>(primal[i - 1]) : subFold(i)) * v;
                    break;
                case Operon::Hash(Operon::BuiltinOp::Div):
                    primal[i] = (node.Arity == 1 ? pappus::ops::inv<Scalar>(primal[i - 1]) : divFold(i)) * v;
                    break;
                case Operon::Hash(Operon::BuiltinOp::Fmin): primal[i] = minFold(i) * v; break;
                case Operon::Hash(Operon::BuiltinOp::Fmax): primal[i] = maxFold(i) * v; break;
                default:
                    // Mirrors IntervalEvaluator::Evaluate()'s own miss
                    // behavior exactly (interval_evaluator.hpp: "still
                    // throws on a miss, unchanged from today") — today this
                    // is unreachable whenever the caller has already run
                    // ValidateSymbolicDiffCoverage (strictly narrower than
                    // interval coverage), but that ordering isn't enforced
                    // by the type system, so guard it explicitly rather
                    // than assume the caller got the sequencing right.
                    if (node.Arity == 1) {
                        if (auto const* unary = IntervalUnaryRules().TryGet(node.HashValue)) {
                            primal[i] = (*unary)(primal[i - 1]) * v;
                            break;
                        }
                    } else if (node.Arity == 2) {
                        auto const j = i - 1;
                        auto const k = j - (bodyNodes[j].Length + 1);
                        if (auto const* binary = IntervalBinaryRules().TryGet(node.HashValue)) {
                            primal[i] = (*binary)(primal[j], primal[k]) * v;
                            break;
                        }
                    }
                    throw std::runtime_error(fmt::format(
                        "composed-function interval evaluation: node kind `{}` not yet mapped", node.Name()));
                }
            }
        }
        return primal.back();
    };
}

// Binary counterpart, pluggable into RegisterBinaryInterval. Mirrors
// MakeComposedIntervalUnaryFn exactly except for param-leaf binding: `argJ`
// (near outer operand) binds to param[1], `argK` (far outer operand) binds
// to param[0] — same reversal BindArgIndices/MakeComposedBinarySymbolicDerivRule
// already established (Tree::Indices enumerates nearest-first, the reverse
// of textual/formal-parameter order).
inline auto MakeComposedIntervalBinaryFn(Tree const& body) -> IntervalBinaryFn
{
    auto const& bodyNodes = body.Nodes();
    return [bodyNodes](pappus::interval<Operon::Scalar> const& argJ, pappus::interval<Operon::Scalar> const& argK)
               -> pappus::interval<Operon::Scalar> {
        using Scalar = Operon::Scalar;
        using Interval = pappus::interval<Scalar>;
        RegisterIntervalBuiltins();
        std::array<Interval, kMaxComposedFunctionArity> const args{argK, argJ}; // args[paramIdx]

        auto const n = bodyNodes.size();
        std::vector<Interval> primal(n);

        auto const addFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(bodyNodes, i)) { acc = pappus::ops::add<Scalar>(acc, primal[j]); }
            return acc;
        };
        auto const mulFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{1}};
            for (auto j : Tree::Indices(bodyNodes, i)) { acc = pappus::ops::mul<Scalar>(acc, primal[j]); }
            return acc;
        };
        auto const subFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (first) { acc = primal[j]; first = false; } else { acc = pappus::ops::sub<Scalar>(acc, primal[j]); }
            }
            return acc;
        };
        auto const divFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{1}};
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (first) { acc = primal[j]; first = false; } else { acc = pappus::ops::div<Scalar>(acc, primal[j]); }
            }
            return acc;
        };
        auto const minFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (first) { acc = primal[j]; first = false; } else { acc = pappus::ops::min<Scalar>(acc, primal[j]); }
            }
            return acc;
        };
        auto const maxFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (first) { acc = primal[j]; first = false; } else { acc = pappus::ops::max<Scalar>(acc, primal[j]); }
            }
            return acc;
        };

        for (std::size_t i = 0; i < n; ++i) {
            auto const& node = bodyNodes[i];
            auto const v = static_cast<Scalar>(node.Value);

            if (node.Type == Operon::NodeType::Constant) {
                primal[i] = pappus::ops::constant<Scalar>(v);
            } else if (node.IsVariable()) {
                auto const pIdx = static_cast<std::size_t>(node.HashValue - Operon::BuiltinOpCount);
                primal[i] = args[pIdx] * v;
            } else if (node.IsRef()) {
                primal[i] = primal[node.RefTo];
            } else {
                switch (node.HashValue) {
                case Operon::Hash(Operon::BuiltinOp::Add): primal[i] = addFold(i) * v; break;
                case Operon::Hash(Operon::BuiltinOp::Mul): primal[i] = mulFold(i) * v; break;
                case Operon::Hash(Operon::BuiltinOp::Sub):
                    primal[i] = (node.Arity == 1 ? pappus::ops::neg<Scalar>(primal[i - 1]) : subFold(i)) * v;
                    break;
                case Operon::Hash(Operon::BuiltinOp::Div):
                    primal[i] = (node.Arity == 1 ? pappus::ops::inv<Scalar>(primal[i - 1]) : divFold(i)) * v;
                    break;
                case Operon::Hash(Operon::BuiltinOp::Fmin): primal[i] = minFold(i) * v; break;
                case Operon::Hash(Operon::BuiltinOp::Fmax): primal[i] = maxFold(i) * v; break;
                default:
                    if (node.Arity == 1) {
                        if (auto const* unary = IntervalUnaryRules().TryGet(node.HashValue)) {
                            primal[i] = (*unary)(primal[i - 1]) * v;
                            break;
                        }
                    } else if (node.Arity == 2) {
                        auto const j = i - 1;
                        auto const k = j - (bodyNodes[j].Length + 1);
                        if (auto const* binary = IntervalBinaryRules().TryGet(node.HashValue)) {
                            primal[i] = (*binary)(primal[j], primal[k]) * v;
                            break;
                        }
                    }
                    throw std::runtime_error(fmt::format(
                        "composed-function interval evaluation: node kind `{}` not yet mapped", node.Name()));
                }
            }
        }
        return primal.back();
    };
}

// Builds an AffineUnaryFn for an *arity-1* composed function, pluggable
// directly into RegisterUnaryAffine. Mirrors MakeComposedIntervalUnaryFn
// exactly, except every op threads the *caller-supplied* affine context
// (`ctx`, the same one the outer AffineEvaluator instance owns) rather than
// constructing a fresh one — this is exactly the affine-correlation fix
// (Fix 1) from the design: reusing the same context means the same param
// occurrence's affine noise symbols are shared correctly, so e.g. `x - f(x)`
// for `f(x)=x` encloses to exactly `0`, not `[-w,+w]`.
inline auto MakeComposedAffineUnaryFn(Tree const& body) -> AffineUnaryFn
{
    auto const& bodyNodes = body.Nodes();
    return [bodyNodes](pappus::ops::affine_context<Operon::Scalar> const& ctx, pappus::affine_form<Operon::Scalar> const& arg) -> pappus::affine_form<Operon::Scalar> {
        using Scalar = Operon::Scalar;
        using Affine = pappus::affine_form<Scalar>;
        RegisterAffineBuiltins();

        // pappus::ops::constant(context, value) needs a *non-const*
        // pappus::ops::affine_context<T>& (it allocates no new symbol for a
        // plain scalar, but is still declared non-const) — AffineUnaryFn's
        // fixed signature only ever hands us a const one. affine_form's own
        // constructor taking the *inner* pappus::affine_context (ctx.state)
        // by const reference is the equivalent leaf-construction path that
        // actually accepts a const context — used directly here instead.
        auto const makeConst = [&ctx](Scalar v) { return Affine{ctx.state, v}; };

        auto const n = bodyNodes.size();
        std::vector<Affine> primal;
        primal.reserve(n);

        auto const addFold = [&](std::size_t i) {
            auto acc = makeConst(Scalar{0});
            for (auto j : Tree::Indices(bodyNodes, i)) { acc = pappus::ops::add<Scalar>(ctx, acc, primal[j]); }
            return acc;
        };
        auto const mulFold = [&](std::size_t i) {
            auto acc = makeConst(Scalar{1});
            for (auto j : Tree::Indices(bodyNodes, i)) { acc = pappus::ops::mul<Scalar>(ctx, acc, primal[j]); }
            return acc;
        };
        auto const subFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (!acc) { acc = primal[j]; } else { acc = pappus::ops::sub<Scalar>(ctx, *acc, primal[j]); }
            }
            return std::move(*acc);
        };
        auto const divFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (!acc) { acc = primal[j]; } else { acc = pappus::ops::div<Scalar>(ctx, *acc, primal[j]); }
            }
            return std::move(*acc);
        };
        auto const minFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (!acc) { acc = primal[j]; } else { acc = pappus::ops::min<Scalar>(ctx, *acc, primal[j]); }
            }
            return std::move(*acc);
        };
        auto const maxFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (!acc) { acc = primal[j]; } else { acc = pappus::ops::max<Scalar>(ctx, *acc, primal[j]); }
            }
            return std::move(*acc);
        };

        for (std::size_t i = 0; i < n; ++i) {
            auto const& node = bodyNodes[i];
            auto const v = static_cast<Scalar>(node.Value);

            if (node.Type == Operon::NodeType::Constant) {
                primal.push_back(makeConst(v));
            } else if (node.IsVariable()) {
                primal.push_back(arg * v);
            } else if (node.IsRef()) {
                primal.push_back(primal[node.RefTo]);
            } else {
                switch (node.HashValue) {
                case Operon::Hash(Operon::BuiltinOp::Add): primal.push_back(addFold(i) * v); break;
                case Operon::Hash(Operon::BuiltinOp::Mul): primal.push_back(mulFold(i) * v); break;
                case Operon::Hash(Operon::BuiltinOp::Sub):
                    primal.push_back((node.Arity == 1 ? pappus::ops::neg<Scalar>(primal[i - 1]) : subFold(i)) * v);
                    break;
                case Operon::Hash(Operon::BuiltinOp::Div):
                    primal.push_back((node.Arity == 1 ? pappus::ops::inv<Scalar>(ctx, primal[i - 1]) : divFold(i)) * v);
                    break;
                case Operon::Hash(Operon::BuiltinOp::Fmin): primal.push_back(minFold(i) * v); break;
                case Operon::Hash(Operon::BuiltinOp::Fmax): primal.push_back(maxFold(i) * v); break;
                default:
                    // See the matching comment in MakeComposedIntervalUnaryFn.
                    if (node.Arity == 1) {
                        if (auto const* unary = AffineUnaryRules().TryGet(node.HashValue)) {
                            primal.push_back((*unary)(ctx, primal[i - 1]) * v);
                            break;
                        }
                    } else if (node.Arity == 2) {
                        auto const j = i - 1;
                        auto const k = j - (bodyNodes[j].Length + 1);
                        if (auto const* binary = AffineBinaryRules().TryGet(node.HashValue)) {
                            primal.push_back((*binary)(ctx, primal[j], primal[k]) * v);
                            break;
                        }
                    }
                    throw std::runtime_error(fmt::format(
                        "composed-function affine evaluation: node kind `{}` not yet mapped", node.Name()));
                }
            }
        }
        return primal.back();
    };
}

// Binary counterpart, pluggable into RegisterBinaryAffine. Mirrors
// MakeComposedAffineUnaryFn exactly except for param-leaf binding — same
// argJ(near)->param[1], argK(far)->param[0] reversal as
// MakeComposedIntervalBinaryFn.
inline auto MakeComposedAffineBinaryFn(Tree const& body) -> AffineBinaryFn
{
    auto const& bodyNodes = body.Nodes();
    return [bodyNodes](pappus::ops::affine_context<Operon::Scalar> const& ctx,
               pappus::affine_form<Operon::Scalar> const& argJ,
               pappus::affine_form<Operon::Scalar> const& argK) -> pappus::affine_form<Operon::Scalar> {
        using Scalar = Operon::Scalar;
        using Affine = pappus::affine_form<Scalar>;
        RegisterAffineBuiltins();

        auto const makeConst = [&ctx](Scalar v) { return Affine{ctx.state, v}; };
        std::array<Affine const*, kMaxComposedFunctionArity> const args{&argK, &argJ}; // args[paramIdx]

        auto const n = bodyNodes.size();
        std::vector<Affine> primal;
        primal.reserve(n);

        auto const addFold = [&](std::size_t i) {
            auto acc = makeConst(Scalar{0});
            for (auto j : Tree::Indices(bodyNodes, i)) { acc = pappus::ops::add<Scalar>(ctx, acc, primal[j]); }
            return acc;
        };
        auto const mulFold = [&](std::size_t i) {
            auto acc = makeConst(Scalar{1});
            for (auto j : Tree::Indices(bodyNodes, i)) { acc = pappus::ops::mul<Scalar>(ctx, acc, primal[j]); }
            return acc;
        };
        auto const subFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (!acc) { acc = primal[j]; } else { acc = pappus::ops::sub<Scalar>(ctx, *acc, primal[j]); }
            }
            return std::move(*acc);
        };
        auto const divFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (!acc) { acc = primal[j]; } else { acc = pappus::ops::div<Scalar>(ctx, *acc, primal[j]); }
            }
            return std::move(*acc);
        };
        auto const minFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (!acc) { acc = primal[j]; } else { acc = pappus::ops::min<Scalar>(ctx, *acc, primal[j]); }
            }
            return std::move(*acc);
        };
        auto const maxFold = [&](std::size_t i) {
            std::optional<Affine> acc;
            for (auto j : Tree::Indices(bodyNodes, i)) {
                if (!acc) { acc = primal[j]; } else { acc = pappus::ops::max<Scalar>(ctx, *acc, primal[j]); }
            }
            return std::move(*acc);
        };

        for (std::size_t i = 0; i < n; ++i) {
            auto const& node = bodyNodes[i];
            auto const v = static_cast<Scalar>(node.Value);

            if (node.Type == Operon::NodeType::Constant) {
                primal.push_back(makeConst(v));
            } else if (node.IsVariable()) {
                auto const pIdx = static_cast<std::size_t>(node.HashValue - Operon::BuiltinOpCount);
                primal.push_back(*args[pIdx] * v);
            } else if (node.IsRef()) {
                primal.push_back(primal[node.RefTo]);
            } else {
                switch (node.HashValue) {
                case Operon::Hash(Operon::BuiltinOp::Add): primal.push_back(addFold(i) * v); break;
                case Operon::Hash(Operon::BuiltinOp::Mul): primal.push_back(mulFold(i) * v); break;
                case Operon::Hash(Operon::BuiltinOp::Sub):
                    primal.push_back((node.Arity == 1 ? pappus::ops::neg<Scalar>(primal[i - 1]) : subFold(i)) * v);
                    break;
                case Operon::Hash(Operon::BuiltinOp::Div):
                    primal.push_back((node.Arity == 1 ? pappus::ops::inv<Scalar>(ctx, primal[i - 1]) : divFold(i)) * v);
                    break;
                case Operon::Hash(Operon::BuiltinOp::Fmin): primal.push_back(minFold(i) * v); break;
                case Operon::Hash(Operon::BuiltinOp::Fmax): primal.push_back(maxFold(i) * v); break;
                default:
                    if (node.Arity == 1) {
                        if (auto const* unary = AffineUnaryRules().TryGet(node.HashValue)) {
                            primal.push_back((*unary)(ctx, primal[i - 1]) * v);
                            break;
                        }
                    } else if (node.Arity == 2) {
                        auto const j = i - 1;
                        auto const k = j - (bodyNodes[j].Length + 1);
                        if (auto const* binary = AffineBinaryRules().TryGet(node.HashValue)) {
                            primal.push_back((*binary)(ctx, primal[j], primal[k]) * v);
                            break;
                        }
                    }
                    throw std::runtime_error(fmt::format(
                        "composed-function affine evaluation: node kind `{}` not yet mapped", node.Name()));
                }
            }
        }
        return primal.back();
    };
}

// Ties ParseFunctionBody + every backend derivation above (numeric eval,
// CPU gradient, symbolic diff, interval, affine) into one call — the
// orchestrating entry point the individual builder functions were
// deliberately built to feed into, deferred until now because there was
// nothing to orchestrate until arity-1 *and* arity-2 support existed for
// every backend. JIT codegen (asmjit machine code) is not one of the
// backends registered here — not built for composed functions at all
// (a tree containing one simply never gets JIT-compiled, verified to
// degrade gracefully rather than crash — see the composed_function.cpp
// test suite's `[jit]`-tagged case).
//
// Arity is `params.size()` (<= kMaxComposedFunctionArity), not a value the
// caller supplies independently — `info.Arity` is validated to match
// rather than trusted as a separate source of truth, since (unlike a
// hand-written scalar-lambda registration, where arity isn't otherwise
// derivable) it's always redundant with `params.size()` here.
template<typename DTable, typename T>
void RegisterComposedFunction(
    DTable& dt, PrimitiveSet& pset,
    FunctionInfo const& info,
    std::span<std::string const> params,
    std::string_view bodyInfix)
{
    if (params.size() > kMaxComposedFunctionArity) {
        throw std::invalid_argument(fmt::format(
            "RegisterComposedFunction: {} parameters exceeds the v1 cap of {}",
            params.size(), kMaxComposedFunctionArity));
    }
    if (info.Arity != params.size()) {
        throw std::invalid_argument(fmt::format(
            "RegisterComposedFunction: FunctionInfo::Arity ({}) does not match params.size() ({})",
            info.Arity, params.size()));
    }

    std::vector<std::string> const paramVec(params.begin(), params.end());
    auto body = InfixParser::ParseFunctionBody(bodyInfix, paramVec);
    ValidateSymbolicDiffCoverage(body);

    auto const hash = Operon::Hasher{}(info.Name);
    detail::ValidateUserHash(hash, info.Name);

    auto const arity = params.size();
    dt.template RegisterFunction<T>(hash,
        MakeComposedCallable<DTable, T>(dt, body, arity),
        MakeComposedCallableDiff<DTable, T>(dt, body, arity));

    // arity == 0 needs none of these: a zero-parameter composed function
    // is a constant expression over built-ins, contributing no free
    // coefficient of its own — Deriv() already treats any childless
    // Function node this way with no special-casing needed, and there is
    // no argument to bound for interval/affine either.
    if (arity == 1) {
        RegisterUnarySymbolicDeriv(hash, MakeComposedUnarySymbolicDerivRule(body));
        RegisterUnaryInterval(hash, MakeComposedIntervalUnaryFn(body));
        RegisterUnaryAffine(hash, MakeComposedAffineUnaryFn(body));
    } else if (arity == 2) {
        RegisterBinarySymbolicDeriv(hash, MakeComposedBinarySymbolicDerivRule(body));
        RegisterBinaryInterval(hash, MakeComposedIntervalBinaryFn(body));
        RegisterBinaryAffine(hash, MakeComposedAffineBinaryFn(body));
    }

    pset.AddFunction(hash, static_cast<uint16_t>(arity), info.Frequency);
    Node::RegisterName(hash, info.Name, info.Desc);
}

} // namespace Operon

#endif
