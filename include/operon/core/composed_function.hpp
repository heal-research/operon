// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_COMPOSED_FUNCTION_HPP
#define OPERON_CORE_COMPOSED_FUNCTION_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <fmt/format.h>
#include <stdexcept>

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
