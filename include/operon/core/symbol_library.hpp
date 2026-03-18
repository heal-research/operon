// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_SYMBOL_LIBRARY_HPP
#define OPERON_SYMBOL_LIBRARY_HPP

#include "dispatch.hpp"
#include "node.hpp"
#include "pset.hpp"
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
    if constexpr (std::is_same_v<std::remove_cvref_t<DFa>, Dispatch::Noop>) {
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
struct FunctionInfo {
    Operon::Hash Hash;
    std::string  Name;
    std::string  Desc;
    uint16_t     Arity;
    size_t       Frequency{1};
};

// Register a unary function in the dispatch table, PrimitiveSet, and name
// registry in a single call.  An optional explicit derivative may be supplied;
// if omitted, Jet<T,1> auto-diff is used.
template<typename DTable, typename T, typename F, typename DF = Dispatch::Noop>
void RegisterUnaryFunction(DTable& dt, PrimitiveSet& pset,
                           FunctionInfo const& info, F primal, DF deriv = {})
{
    RegisterUnary<DTable, T>(dt, info.Hash, std::move(primal), std::move(deriv));
    pset.AddFunction(info.Hash, info.Arity, info.Frequency);
    if (!info.Name.empty()) { Node::RegisterName(info.Hash, info.Name, info.Desc); }
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
    RegisterBinary<DTable, T>(dt, info.Hash, std::move(primal),
                              std::move(derivA), std::move(derivB));
    pset.AddFunction(info.Hash, info.Arity, info.Frequency);
    if (!info.Name.empty()) { Node::RegisterName(info.Hash, info.Name, info.Desc); }
}

} // namespace Operon

#endif
