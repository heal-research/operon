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
#include <stdexcept>
#include <vector>

#include "dispatch.hpp"
#include "node.hpp"
#include "tree.hpp"
#include "tree_diff.hpp"

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

    inline auto DiffMix(std::uint64_t a, std::uint64_t b) -> std::uint64_t
    {
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
// v1 scope: arity-2 composed-function symbolic-diff support (a new
// BinarySymbolicDerivRule registry + wiring into Deriv()'s arity==2
// fallthrough) is NOT implemented yet — deliberately deferred, see
// operon-planning/designs/composed-functions.md. A binary composed function
// works fully for numeric eval and the CPU CallableDiff path; only the
// JIT/BuildJacobianDag path is affected, and ValidateSymbolicDiffCoverage
// does not currently reject binary composed functions on this basis (that
// gap is separate from "body references a non-diff built-in").
inline auto MakeComposedUnarySymbolicDerivRule(Tree const& body) -> UnarySymbolicDerivRule
{
    auto const& bodyNodes = body.Nodes();
    return [bodyNodes](detail::DiffNodes& dag, detail::DiffMemo& memo, detail::DiffHashes& h, std::size_t i, std::size_t j) -> std::size_t {
        std::array<std::int64_t, kMaxComposedFunctionArity> liveChildIdx{};
        liveChildIdx[0] = static_cast<std::int64_t>(j);
        std::vector<std::size_t> bodyToLive(bodyNodes.size());
        detail::DiffCopyBody(dag, memo, h, bodyNodes, liveChildIdx, bodyToLive);
        return detail::DiffParam(dag, memo, h, bodyNodes, bodyToLive, bodyNodes.size() - 1, 0);
    };
}

template<typename DTable, typename T>
auto MakeComposedCallable(DTable const& dt, Tree const& body, std::size_t arity) -> typename DTable::template Callable<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    using Callable = typename DTable::template Callable<T>;

    auto ops = detail::ResolveOps<Callable>(dt, body, [](DTable const& t, Operon::Hash h) { return t.template GetFunction<T>(h); });
    auto const& bodyNodes = body.Nodes();

    return [bodyNodes, ops = std::move(ops), arity, S](
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

    return [bodyNodes, fns = std::move(fns), dfs = std::move(dfs), arity, S](
               Operon::Vector<Node> const& outerNodes,
               Backend::View<T const, S> outerPrimal,
               Backend::View<T, S> outerTrace,
               int i,
               int j) {
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

} // namespace Operon

#endif
