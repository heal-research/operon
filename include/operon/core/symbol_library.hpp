// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_SYMBOL_LIBRARY_HPP
#define OPERON_SYMBOL_LIBRARY_HPP

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "dispatch.hpp"
#include "node.hpp"
#include "pset.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/dual.hpp"

// symbol_library.hpp: scalar-lambda adapters for registering user-defined
// symbols into a DispatchTable.
//
// The adapters follow the same conventions as built-in Func<>/Diff<>
// specialisations:
//   - Primal callables read nodes[i].Value as the node weight and multiply.
//   - Derivative callables write ∂f/∂child into trace[child_index], evaluated
//     at the child's primal value, without the weight (the interpreter handles
//     weight accumulation during the reverse/forward trace).
//
// If no derivative is provided, one is generated automatically via
// ceres::Jet<T,1> (forward-mode AD with a single tangent component).
// The lambda must accept a generic (auto) argument for this to work —
// if it is typed to T specifically, provide an explicit derivative instead.

namespace Operon {

// Wrap a scalar unary lambda f(x) -> y into a batched Callable<T>.
// Reads the argument from primal column i-1 and writes weight * f(arg) to
// column i, where weight = nodes[i].Value.
template<typename DTable, typename T, typename F>
auto MakeUnaryCallable(F primal) -> typename DTable::template Callable<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    return [fn = std::move(primal)](
        Operon::Vector<Node> const& nodes,
        Backend::View<T, S>         data,
        size_t                      i,
        Operon::Range               /*rg*/)
    {
        auto const  w   = static_cast<T>(nodes[i].Value);
        auto*       dst = Backend::Ptr(data, i);
        auto const* src = Backend::Ptr(data, i - 1);
        for (auto k = 0UL; k < S; ++k) { dst[k] = w * fn(src[k]); }
    };
}

// Wrap a scalar binary lambda f(a, b) -> y into a batched Callable<T>.
// First child is at index j = i-1, second at k = j - nodes[j].Length - 1.
// Writes weight * f(a, b) to column i, where weight = nodes[i].Value.
template<typename DTable, typename T, typename F>
auto MakeBinaryCallable(F primal) -> typename DTable::template Callable<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    return [fn = std::move(primal)](
        Operon::Vector<Node> const& nodes,
        Backend::View<T, S>         data,
        size_t                      i,
        Operon::Range               /*rg*/)
    {
        auto const  j   = i - 1;
        auto const  k   = j - nodes[j].Length - 1;
        auto const  w   = static_cast<T>(nodes[i].Value);
        auto*       dst = Backend::Ptr(data, i);
        auto const* lhs = Backend::Ptr(data, j);
        auto const* rhs = Backend::Ptr(data, k);
        for (auto s = 0UL; s < S; ++s) { dst[s] = w * fn(lhs[s], rhs[s]); }
    };
}

// Wrap a scalar unary derivative lambda df(x) -> ∂f/∂x into a batched
// CallableDiff<T>.  Reads the child's primal value from primal[j] and writes
// the partial derivative into trace[j].
template<typename DTable, typename T, typename DF>
auto MakeUnaryDiff(DF deriv) -> typename DTable::template CallableDiff<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    return [fn = std::move(deriv)](
        Operon::Vector<Node> const& /*nodes*/,
        Backend::View<T const, S>   primal,
        Backend::View<T>            trace,
        int                         /*i*/,
        int                         j)
    {
        auto*       res = Backend::Ptr(trace, j);
        auto const* pj  = Backend::Ptr(primal, j);
        for (auto s = 0UL; s < S; ++s) { res[s] = fn(pj[s]); }
    };
}

// Wrap two scalar partial-derivative lambdas (dfA, dfB) for a binary
// function into a batched CallableDiff<T>.
//   dfA(a, b) = ∂f/∂a   (called when j is the first child, i-1)
//   dfB(a, b) = ∂f/∂b   (called when j is the second child)
// Reads both children's primal values and writes the appropriate partial
// derivative into trace[j].
template<typename DTable, typename T, typename DFa, typename DFb>
auto MakeBinaryDiff(DFa derivA, DFb derivB) -> typename DTable::template CallableDiff<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    return [fna = std::move(derivA), fnb = std::move(derivB)](
        Operon::Vector<Node> const& nodes,
        Backend::View<T const, S>   primal,
        Backend::View<T>            trace,
        int                         i,
        int                         j)
    {
        auto const  first  = i - 1;
        auto const  second = first - static_cast<int>(nodes[first].Length) - 1;
        auto*       res    = Backend::Ptr(trace, j);
        auto const* pa     = Backend::Ptr(primal, first);
        auto const* pb     = Backend::Ptr(primal, second);
        if (j == first) {
            for (auto s = 0UL; s < S; ++s) { res[s] = fna(pa[s], pb[s]); }
        } else {
            for (auto s = 0UL; s < S; ++s) { res[s] = fnb(pa[s], pb[s]); }
        }
    };
}

// Generate a batched CallableDiff<T> for a unary function using forward-mode
// AD (ceres::Jet<T,1>).  The primal lambda must accept a generic argument
// (i.e., use auto or be templated) so it can be instantiated with Jet<T,1>.
template<typename DTable, typename T, typename F>
auto MakeUnaryAutoDiff(F primal) -> typename DTable::template CallableDiff<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    return [fn = std::move(primal)](
        Operon::Vector<Node> const& /*nodes*/,
        Backend::View<T const, S>   primal,
        Backend::View<T>            trace,
        int                         /*i*/,
        int                         j)
    {
        using Jet = ceres::Jet<T, 1>;
        auto*       res = Backend::Ptr(trace, j);
        auto const* pj  = Backend::Ptr(primal, j);
        for (auto s = 0UL; s < S; ++s) {
            res[s] = fn(Jet{pj[s], 0}).v[0]; // seed x with unit tangent
        }
    };
}

// Generate a batched CallableDiff<T> for a binary function using forward-mode
// AD (ceres::Jet<T,1>).  Seeds one argument at a time; called once per child.
template<typename DTable, typename T, typename F>
auto MakeBinaryAutoDiff(F primal) -> typename DTable::template CallableDiff<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    return [fn = std::move(primal)](
        Operon::Vector<Node> const& nodes,
        Backend::View<T const, S>   primal,
        Backend::View<T>            trace,
        int                         i,
        int                         j)
    {
        using Jet = ceres::Jet<T, 1>;
        auto const  first  = i - 1;
        auto const  second = first - static_cast<int>(nodes[first].Length) - 1;
        auto*       res    = Backend::Ptr(trace, j);
        auto const* pa     = Backend::Ptr(primal, first);
        auto const* pb     = Backend::Ptr(primal, second);
        if (j == first) {
            for (auto s = 0UL; s < S; ++s) {
                res[s] = fn(Jet{pa[s], 0}, Jet{pb[s]}).v[0]; // ∂f/∂a
            }
        } else {
            for (auto s = 0UL; s < S; ++s) {
                res[s] = fn(Jet{pa[s]}, Jet{pb[s], 0}).v[0]; // ∂f/∂b
            }
        }
    };
}

// Collect the column indices of node i's children, in left-to-right tree
// order (Nodes() stores children right-to-left of their parent).
inline auto ChildIndices(Operon::Vector<Node> const& nodes, size_t i) -> std::vector<size_t>
{
    std::vector<size_t> children;
    children.reserve(nodes[i].Arity);
    auto j = i - 1;
    for (auto a = 0; a < nodes[i].Arity; ++a) {
        children.push_back(j);
        j -= nodes[j].Length + 1;
    }
    std::ranges::reverse(children);
    return children;
}

// Wrap a scalar binary reduce lambda f(acc, x) -> acc' into a batched
// Callable<T> for n-ary (arity >= 2) functions. Children are folded strictly
// left-to-right: acc = f(f(c0, c1), c2, ...). Arity is read per-instance
// from nodes[i].Arity, mirroring how built-in Add/Mul accumulate. This does
// NOT match every built-in n-ary op's semantics: Sub/Div compute
// c0 - (c1+c2+...) / c0 / (c1*c2*...) (head vs. reduced tail), not a strict
// left fold, so this adapter is only correct for associative reductions
// like Add/Mul; a Sub/Div-shaped n-ary function needs a different adapter.
template<typename DTable, typename T, typename F>
auto MakeNaryCallable(F primal) -> typename DTable::template Callable<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    return [fn = std::move(primal)](
        Operon::Vector<Node> const& nodes,
        Backend::View<T, S>         data,
        size_t                      i,
        Operon::Range               /*rg*/)
    {
        auto const  w        = static_cast<T>(nodes[i].Value);
        auto*       dst      = Backend::Ptr(data, i);
        auto const  children = ChildIndices(nodes, i);
        for (auto s = 0UL; s < S; ++s) {
            auto acc = Backend::Ptr(data, children[0])[s];
            for (size_t c = 1; c < children.size(); ++c) {
                acc = fn(acc, Backend::Ptr(data, children[c])[s]);
            }
            dst[s] = w * acc;
        }
    };
}

// Generate a batched CallableDiff<T> for an n-ary reduce function using
// forward-mode AD (ceres::Jet<T,1>): re-runs the same fold with the target
// child (index j) seeded with unit tangent and every other child at zero
// tangent, so a single primal lambda covers every child's partial without a
// separate derivative rule per argument position.
template<typename DTable, typename T, typename F>
auto MakeNaryAutoDiff(F primal) -> typename DTable::template CallableDiff<T>
{
    constexpr auto S = DTable::template BatchSize<T>;
    return [fn = std::move(primal)](
        Operon::Vector<Node> const& nodes,
        Backend::View<T const, S>   primal,
        Backend::View<T>            trace,
        int                         i,
        int                         j)
    {
        using Jet = ceres::Jet<T, 1>;
        auto const  children = ChildIndices(nodes, static_cast<size_t>(i));
        auto*       res       = Backend::Ptr(trace, j);
        for (auto s = 0UL; s < S; ++s) {
            auto const seed = [&](size_t child) {
                auto const v = Backend::Ptr(primal, children[child])[s];
                // Jet{v, 0} seeds tangent direction 0 (the only one Jet<T,1>
                // has) to 1; Jet{v} alone leaves the tangent at 0 (matches
                // MakeUnaryAutoDiff's convention above).
                return static_cast<int>(children[child]) == j ? Jet{v, 0} : Jet{v};
            };
            auto acc = seed(0);
            for (size_t c = 1; c < children.size(); ++c) {
                acc = fn(acc, seed(c));
            }
            res[s] = acc.v[0];
        }
    };
}

// Register an n-ary (arity >= 2) function from a scalar binary reduce lambda
// f(acc, x) -> acc'. Only Jet<T,1> auto-diff is supported (no explicit
// per-argument derivative overload, unlike RegisterUnary/RegisterBinary —
// the fold structure means a single primal lambda already determines every
// child's partial via MakeNaryAutoDiff).
template<typename DTable, typename T, typename F>
void RegisterNary(DTable& dt, Operon::Hash hash, F primal)
{
    dt.template RegisterFunction<T>(hash,
        MakeNaryCallable<DTable, T>(primal),
        MakeNaryAutoDiff<DTable, T>(std::move(primal)));
}

// Register a unary function from scalar lambdas.
// primal:  f(x)  -> y         (required; must use auto/generic argument for
//                               auto-diff to work)
// deriv:   df(x) -> ∂f/∂x    (optional; if omitted, Jet<T,1> auto-diff is used)
template<typename DTable, typename T, typename F, typename DF = Dispatch::Noop>
void RegisterUnary(DTable& dt, Operon::Hash hash, F primal, DF deriv = {})
{
    if constexpr (std::is_same_v<std::remove_cvref_t<DF>, Dispatch::Noop>) {
        // auto-diff: primal is copied into both the callable and the diff adapter
        dt.template RegisterFunction<T>(hash,
            MakeUnaryCallable<DTable, T>(primal),
            MakeUnaryAutoDiff<DTable, T>(std::move(primal)));
    } else {
        dt.template RegisterFunction<T>(hash,
            MakeUnaryCallable<DTable, T>(std::move(primal)),
            MakeUnaryDiff<DTable, T>(std::move(deriv)));
    }
}

// Register a binary function from scalar lambdas.
// primal:  f(a, b)    -> y         (required; must use auto/generic arguments)
// derivA:  df_da(a,b) -> ∂f/∂a    (optional; if omitted, Jet<T,1> is used)
// derivB:  df_db(a,b) -> ∂f/∂b    (optional; if omitted, Jet<T,1> is used)
template<typename DTable, typename T,
         typename F,
         typename DFa = Dispatch::Noop,
         typename DFb = Dispatch::Noop>
void RegisterBinary(DTable& dt, Operon::Hash hash, F primal,
                    DFa derivA = {}, DFb derivB = {})
{
    constexpr auto aIsNoop = std::is_same_v<std::remove_cvref_t<DFa>, Dispatch::Noop>;
    constexpr auto bIsNoop = std::is_same_v<std::remove_cvref_t<DFb>, Dispatch::Noop>;
    static_assert(aIsNoop == bIsNoop,
        "RegisterBinary: provide both partial derivatives or neither; "
        "supplying only one is not supported.");
    if constexpr (aIsNoop) {
        dt.template RegisterFunction<T>(hash,
            MakeBinaryCallable<DTable, T>(primal),
            MakeBinaryAutoDiff<DTable, T>(std::move(primal)));
    } else {
        dt.template RegisterFunction<T>(hash,
            MakeBinaryCallable<DTable, T>(std::move(primal)),
            MakeBinaryDiff<DTable, T>(std::move(derivA), std::move(derivB)));
    }
}

// Bundles the metadata needed to register a user-defined function symbol.
// Hash is derived from Name (via Operon::Hasher) rather than supplied by the
// caller, so it can't accidentally collide with the small-integer hash range
// reserved for built-ins (see BuiltinOp's HashValue == static_cast<Hash>(op)).
struct FunctionInfo {
    std::string Name;
    std::string Desc;
    uint16_t    Arity;
    size_t      Frequency{1};
};

namespace detail {
    // [0, BuiltinOpCount) is reserved for built-in ops (a Function node's
    // HashValue == static_cast<Hash>(some BuiltinOp)); a name-derived hash
    // landing there would silently overwrite a built-in's dispatch entry /
    // name+desc. A real 64-bit XXHash of a non-empty name landing in that
    // ~30-value range is not a plausible accident, but the check is nearly
    // free and turns the impossible case into a clear error instead of
    // silent corruption.
    inline void ValidateUserHash(Operon::Hash hash, std::string_view name)
    {
        if (name.empty()) {
            throw std::invalid_argument("FunctionInfo::Name must not be empty (it seeds the function's hash)");
        }
        if (hash < BuiltinOpCount) {
            throw std::invalid_argument("FunctionInfo: name-derived hash falls in the range reserved for built-in ops");
        }
        if (hash < BuiltinOpCount + kMaxComposedFunctionArity) {
            throw std::invalid_argument("FunctionInfo: name-derived hash falls in the range reserved for composed-function parameters");
        }
    }
} // namespace detail

// Register a unary function in the dispatch table, PrimitiveSet, and name
// registry in a single call.  An optional explicit derivative may be supplied;
// if omitted, Jet<T,1> auto-diff is used.
template<typename DTable, typename T, typename F, typename DF = Dispatch::Noop>
void RegisterUnaryFunction(DTable& dt, PrimitiveSet& pset,
                           FunctionInfo const& info, F primal, DF deriv = {})
{
    auto const hash = Operon::Hasher{}(info.Name);
    detail::ValidateUserHash(hash, info.Name);
    RegisterUnary<DTable, T>(dt, hash, std::move(primal), std::move(deriv));
    pset.AddFunction(hash, info.Arity, info.Frequency);
    Node::RegisterName(hash, info.Name, info.Desc);
}

// Register a binary function in the dispatch table, PrimitiveSet, and name
// registry in a single call.  Optional explicit partial derivatives may be
// supplied; if omitted, Jet<T,1> auto-diff is used.
template<typename DTable, typename T, typename F,
         typename DFa = Dispatch::Noop, typename DFb = Dispatch::Noop>
void RegisterBinaryFunction(DTable& dt, PrimitiveSet& pset,
                            FunctionInfo const& info, F primal,
                            DFa derivA = {}, DFb derivB = {})
{
    auto const hash = Operon::Hasher{}(info.Name);
    detail::ValidateUserHash(hash, info.Name);
    RegisterBinary<DTable, T>(dt, hash, std::move(primal),
                              std::move(derivA), std::move(derivB));
    pset.AddFunction(hash, info.Arity, info.Frequency);
    Node::RegisterName(hash, info.Name, info.Desc);
}

// Register an n-ary function in the dispatch table, PrimitiveSet, and name
// registry in a single call. info.Arity is the minimum (and default) child
// count; maxArity widens the PrimitiveSet entry so tree generation can
// sample any arity in [info.Arity, maxArity] for this function, the same
// range PrimitiveSet::SampleRandomSymbol already draws from for built-in
// n-ary ops.
template<typename DTable, typename T, typename F>
void RegisterNaryFunction(DTable& dt, PrimitiveSet& pset,
                          FunctionInfo const& info, uint16_t maxArity, F primal)
{
    if (info.Arity < 2) {
        throw std::invalid_argument("RegisterNaryFunction: info.Arity must be >= 2 (n-ary means 2 or more children)");
    }
    if (maxArity < info.Arity) {
        throw std::invalid_argument("RegisterNaryFunction: maxArity must be >= info.Arity");
    }
    auto const hash = Operon::Hasher{}(info.Name);
    detail::ValidateUserHash(hash, info.Name);
    RegisterNary<DTable, T>(dt, hash, std::move(primal));
    pset.AddFunction(hash, info.Arity, info.Frequency);
    if (maxArity > info.Arity) { pset.SetMaximumArity(hash, maxArity); }
    Node::RegisterName(hash, info.Name, info.Desc);
}

} // namespace Operon

#endif
