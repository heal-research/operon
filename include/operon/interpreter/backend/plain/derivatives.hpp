// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_BACKEND_PLAIN_DERIVATIVES_HPP
#define OPERON_BACKEND_PLAIN_DERIVATIVES_HPP

#include "operon/core/node.hpp"
#include "functions.hpp"

namespace Operon::Backend {
namespace detail {
    template<typename T>
    inline auto IsNaN(T value) { return std::isnan(value); }

    template<typename Compare>
    struct FComp {
        auto operator()(auto x, auto y) const {
            using T = std::common_type_t<decltype(x), decltype(y)>;
            if ((IsNaN(x) && IsNaN(y)) || (x == y)) {
                return std::numeric_limits<T>::quiet_NaN();
            }
            if (IsNaN(x)) { return T{0}; }
            if (IsNaN(y)) { return T{1}; }
            return static_cast<T>(Compare{}(T{x}, T{y}));
        }
    };

    template<typename T>
    inline auto Sgn(T x) {
        return (T{0} < x) - (x < T{0});
    }
} // namespace detail

    template<typename T, std::size_t S>
    auto Add(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        std::ranges::fill_n(Ptr(trace, j), S, T{1});
    }

    // Every node's forward pass multiplies by its own weight (Backend::Mul
    // etc: `res = weight * f(args)`), so primal[i] already has `w` baked in.
    // Reading primal[i] as a shortcut for the *unweighted* value (as the
    // rest of this function otherwise would) double-counts `w`, since
    // ReverseTraceGeneric separately multiplies the returned local
    // derivative by `w` once more during the backward sweep — dividing it
    // back out here keeps the shortcut while still returning the correct
    // unweighted local derivative. Confirmed via reproduction (see
    // foolnotion/operon-planning's double-weighted-derivative bug writeup)
    // against an independent JAX-computed ground truth.
    template<typename T, std::size_t S>
    auto Mul(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const w = static_cast<T>(nodes[i].Value);
        auto *res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        std::transform(pi, pi+S, pj, res, [w](auto x, auto y) { return x / (w * y); });
    }

    template<typename T, std::size_t S>
    auto Sub(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto v = (nodes[i].Arity == 1 || j < i-1) ? T{-1} : T{+1};
        std::ranges::fill_n(Ptr(trace, j), S, v);
    }

    template<typename T, std::size_t S>
    auto Div(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const& n = nodes[i];
        auto *res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);

        if (n.Arity == 1) {
            std::transform(pj, pj+S, res, [](auto x) { return -T{1} / (x * x); });
        } else {
            auto const w = static_cast<T>(n.Value);
            auto v = j == i-1 ? T{1} : T{-1};
            std::transform(pi, pi+S, pj, res, [v, w](auto x, auto y) { return v * x / (w * y); });
        }
    }

    template<typename T, std::size_t S>
    auto Aq(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const w = static_cast<T>(nodes[i].Value);
        auto *res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);

        if (j == i-1) {
            std::transform(pi, pi+S, pj, res, [w](auto x, auto y) { return x / (w * y); });
        } else {
            auto const* pk = Ptr(primal, i-1);
            auto const w3 = w * w * w;
            for (auto s = 0UL; s < S; ++s) {
                auto const a = pi[s];
                auto const b = pj[s];
                auto const c = pk[s];
                res[s] = -b * a * a * a / (w3 * c * c);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Pow(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const w = static_cast<T>(nodes[i].Value);
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);

        if (j == i-1) {
            auto const k = j - (nodes[j].Length + 1);
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; ++s) {
                res[s] = pi[s] * pk[s] / (w * pj[s]);
            }
        } else {
            auto const k = i-1;
            auto const* pk = Ptr(primal, k);
            std::transform(pi, pi+S, pk, res, [w](auto x, auto y) { return x * std::log(y) / w; });
        }
    }

    template<typename T, std::size_t S>
    auto Powabs(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const w = static_cast<T>(nodes[i].Value);
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);

        if (j == i-1) {
            auto const k = j - (nodes[j].Length + 1);
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; ++s) {
                res[s] = pi[s] * pk[s] * detail::Sgn(pj[s]) / (w * std::abs(pj[s]));
            }
        } else {
            auto const k = i-1;
            auto const* pk = Ptr(primal, k);
            std::transform(pi, pi+S, pk, res, [w](auto x, auto y) { return x * std::log(std::abs(y)) / w; });
        }
    }

    template<typename T, std::size_t S>
    auto Min(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto k = j == i - 1 ? (j - nodes[j].Length - 1) : i - 1;
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        auto const* pk = Ptr(primal, k);
        std::transform(pj, pj+S, pk, res, detail::FComp<std::less<>>{});
    }

    template<typename T, std::size_t S>
    auto Max(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto k = j == i - 1 ? (j - nodes[j].Length - 1) : i - 1;
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        auto const* pk = Ptr(primal, k);
        std::transform(pj, pj+S, pk, res, detail::FComp<std::greater<>>{});
    }

    template<typename T, std::size_t S>
    auto Square(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { return T{2} * x; });
    }

    template<typename T, std::size_t S>
    auto Abs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, detail::Sgn<T>);
    }

    template<typename T, std::size_t S>
    auto Ceil(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        // Derivative is zero a.e.; provides no gradient information (cf. Ceres jet.h).
        std::fill_n(Ptr(trace, j), S, T{0});
    }

    template<typename T, std::size_t S>
    auto Floor(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        // Derivative is zero a.e.; provides no gradient information (cf. Ceres jet.h).
        std::fill_n(Ptr(trace, j), S, T{0});
    }

    template<typename T, std::size_t S>
    auto Exp(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const w = static_cast<T>(nodes[i].Value);
        auto const* pi = Ptr(primal, i);
        auto* res = Ptr(trace, j);
        std::transform(pi, pi+S, res, [w](auto x) { return x / w; });
    }

    template<typename T, std::size_t S>
    auto Log(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x){ return T{1} / x; });
    }

    template<typename T, std::size_t S>
    auto Log1p(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x){ return T{1} / (T{1} + x); });
    }

    template<typename T, std::size_t S>
    auto Logabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { return detail::Sgn(x) / std::abs(x); });
    }

    template<typename T, std::size_t S>
    auto Sin(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x){ return std::cos(x); });
    }

    template<typename T, std::size_t S>
    auto Cos(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x){ return -std::sin(x); });
    }

    // Recomputes tan(x) from the child's own primal `pj` (the argument),
    // rather than reading this node's own weighted primal[i] (=w*tan(x)) as
    // if it were tan(x) directly — matching the already-correct eve/eigen
    // backends' implementation of this op.
    template<typename T, std::size_t S>
    auto Tan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { auto const t = std::tan(x); return T{1} + t * t; });
    }

    template<typename T, std::size_t S>
    auto Sinh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { return std::cosh(x); });
    }

    template<typename T, std::size_t S>
    auto Cosh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { return std::sinh(x); });
    }

    // Same fix as Tan above: recompute tanh(x) from the child's own primal
    // `pj`, matching the already-correct eve/eigen backends.
    template<typename T, std::size_t S>
    auto Tanh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { auto const t = std::tanh(x); return T{1} - t * t; });
    }

    template<typename T, std::size_t S>
    auto Asin(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { return T{1} / std::sqrt(T{1} - x * x); });
    }

    template<typename T, std::size_t S>
    auto Acos(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { return -T{1} / std::sqrt(T{1} - x * x); });
    }

    template<typename T, std::size_t S>
    auto Atan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x) { return T{1} / (T{1} + x * x); });
    }

    template<typename T, std::size_t S>
    auto Sqrt(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const w = static_cast<T>(nodes[i].Value);
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        std::transform(pi, pi+S, res, [w](auto x){ return w / (T{2} * x); });
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const w = static_cast<T>(nodes[i].Value);
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        std::transform(pi, pi+S, pj, res, [w](auto x, auto y){ return w * detail::Sgn(y) / (T{2} * x); });
    }

    template<typename T, std::size_t S>
    auto Cbrt(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const w = static_cast<T>(nodes[i].Value);
        auto const w2 = w * w;
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        std::transform(pi, pi+S, res, [w2](auto x){ return w2 / (T{3} * x*x); });
    }
}  // namespace Operon::Backend

#endif