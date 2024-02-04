// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

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

    template<typename T, std::size_t S>
    auto Mul(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto *res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        std::transform(pi, pi+S, pj, res, std::divides{});
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
            auto v = j == i-1 ? T{1} : T{-1};
            std::transform(pi, pi+S, pj, res, [v](auto x, auto y) { return v * x / y; });
        }
    }

    template<typename T, std::size_t S>
    auto Aq(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto *res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);

        if (j == i-1) {
            std::transform(pi, pi+S, pj, res, std::divides{});
        } else {
            auto const* pk = Ptr(primal, i-1);
            for (auto s = 0UL; s < S; ++s) {
                auto const a = pi[s];
                auto const b = pj[s];
                auto const c = pk[s];
                res[s] = -b * a * a * a / (c * c);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Pow(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);

        if (j == i-1) {
            auto const k = j - (nodes[j].Length + 1);
            auto const* pk = Ptr(primal, k);
            for (auto s = 0UL; s < S; ++s) {
                res[s] = pi[s] * pk[s] / pj[s];
            }
        } else {
            auto const k = i-1;
            auto const* pk = Ptr(primal, k);
            std::transform(pi, pi+S, pk, res, [](auto x, auto y) { return x * std::log(y); });
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
    auto Ceil(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x){ return std::ceil(x); });
    }

    template<typename T, std::size_t S>
    auto Floor(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pj = Ptr(primal, j);
        std::transform(pj, pj+S, res, [](auto x){ return std::floor(x); });
    }

    template<typename T, std::size_t S>
    auto Exp(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        std::ranges::copy_n(Ptr(primal, i), S, Ptr(trace, j));
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

    template<typename T, std::size_t S>
    auto Tan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        std::transform(pi, pi+S, res, [](auto x) { return T{1} + x * x; });
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

    template<typename T, std::size_t S>
    auto Tanh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        std::transform(pi, pi+S, res, [](auto x) { return T{1} - x * x; });
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
    auto Sqrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        std::transform(pi, pi+S, res, [](auto x){ return T{1} / (T{2} * x); });
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        auto const* pj = Ptr(primal, j);
        std::transform(pi, pi+S, pj, res, [](auto x, auto y){ return detail::Sgn(y) / (T{2} * x); });
    }

    template<typename T, std::size_t S>
    auto Cbrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        // Col(trace, j) = (T{3} * Col(primal, i).square()).inverse();
        auto* res = Ptr(trace, j);
        auto const* pi = Ptr(primal, i);
        std::transform(pi, pi+S, res, [](auto x){ return T{1} / (T{3} * x*x); });
    }
}  // namespace Operon::Backend

#endif