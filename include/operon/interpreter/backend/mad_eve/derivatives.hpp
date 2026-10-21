// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_BACKEND_MAD_EVE_DERIVATIVES_HPP
#define OPERON_BACKEND_MAD_EVE_DERIVATIVES_HPP

#include "functions.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/backend/detail/scalar_derivatives.hpp"

namespace Operon::Backend {
// Computes the derivative directly from the child's value, using the
// same math the forward pass uses for this operation (an approximation
// for float, exact for double — matching functions.hpp). Div/Aq/Pow/
// Powabs (scalar_derivatives.hpp) can't do this: their derivative also
// needs a sibling child's value, so they divide this node's own
// weighted result by its weight instead, which is undefined when the
// weight is 0. This function never divides by the weight, so it has no
// such problem.
template <typename T, std::size_t S>
auto Exp(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace,
    std::integral auto /*i*/, std::integral auto j)
{
    auto* res = Ptr(trace, j);
    auto const* pj = Ptr(primal, j);
    if constexpr (std::is_same_v<T, float>) {
        std::transform(pj, pj + S, res, [](auto x) { return Mad::exp_impl<MadPrecision::Exp>(x); });
    } else {
        std::transform(pj, pj + S, res, [](auto x) { return std::exp(x); });
    }
}

// Computes tan(x) from the child's value pj. primal[i] holds w*tan(x),
// not tan(x) alone, so it can't be used directly here.
template <typename T, std::size_t S>
auto Tan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace,
    std::integral auto /*i*/, std::integral auto j)
{
    auto* res = Ptr(trace, j);
    auto const* pj = Ptr(primal, j);
    if constexpr (std::is_same_v<T, float>) {
        std::transform(pj, pj + S, res, [](auto x) {
            auto const t = Mad::tan_impl<MadPrecision::Tan>(x);
            return T { 1 } + t * t;
        });
    } else {
        std::transform(pj, pj + S, res, [](auto x) {
            auto const t = std::tan(x);
            return T { 1 } + t * t;
        });
    }
}

// Like Tan above: computes tanh(x) from the child's value pj.
template <typename T, std::size_t S>
auto Tanh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace,
    std::integral auto /*i*/, std::integral auto j)
{
    auto* res = Ptr(trace, j);
    auto const* pj = Ptr(primal, j);
    if constexpr (std::is_same_v<T, float>) {
        std::transform(pj, pj + S, res, [](auto x) {
            auto const t = Mad::tanh_impl<MadPrecision::Tanh>(x);
            return T { 1 } - t * t;
        });
    } else {
        std::transform(pj, pj + S, res, [](auto x) {
            auto const t = std::tanh(x);
            return T { 1 } - t * t;
        });
    }
}

template <typename T, std::size_t S>
auto Sqrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace,
    std::integral auto /*i*/, std::integral auto j)
{
    auto* res = Ptr(trace, j);
    auto const* pj = Ptr(primal, j);
    if constexpr (std::is_same_v<T, float>) {
        std::transform(
            pj, pj + S, res, [](auto x) { return T { 1 } / (T { 2 } * Mad::sqrt_impl<MadPrecision::Sqrt>(x)); });
    } else {
        std::transform(pj, pj + S, res, [](auto x) { return T { 1 } / (T { 2 } * std::sqrt(x)); });
    }
}

template <typename T, std::size_t S>
auto Sqrtabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace,
    std::integral auto /*i*/, std::integral auto j)
{
    auto* res = Ptr(trace, j);
    auto const* pj = Ptr(primal, j);
    if constexpr (std::is_same_v<T, float>) {
        std::transform(pj, pj + S, res,
            [](auto x) { return detail::Sgn(x) / (T { 2 } * Mad::sqrt_impl<MadPrecision::Sqrt>(std::abs(x))); });
    } else {
        std::transform(pj, pj + S, res, [](auto x) { return detail::Sgn(x) / (T { 2 } * std::sqrt(std::abs(x))); });
    }
}
} // namespace Operon::Backend

#endif
