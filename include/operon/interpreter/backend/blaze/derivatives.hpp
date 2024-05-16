// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_BLAZE_DERIVATIVES_HPP
#define OPERON_BACKEND_BLAZE_DERIVATIVES_HPP

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
} // namespace detail

    template<typename T, std::size_t S>
    auto Add(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{1};
    }

    template<typename T, std::size_t S>
    auto Mul(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = Col(primal, i) / Col(primal, j);
    }

    template<typename T, std::size_t S>
    auto Sub(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto v = (nodes[i].Arity == 1 || j < i-1) ? T{-1} : T{+1};
        Col(trace, j) = v;
    }

    template<typename T, std::size_t S>
    auto Div(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const& n = nodes[i];
        if (n.Arity == 1) {
            Col(trace, j) = -T{1} / (Col(primal, j) * Col(primal, j));
        } else {
            Col(trace, j) = (j == i-1 ? T{1} : T{-1}) * Col(primal, i) / Col(primal, j);
        }
    }

    template<typename T, std::size_t S>
    auto Aq(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        if (j == i-1) {
            Col(trace, j) = Col(primal, i) / Col(primal, j);
        } else {
            Col(trace, j) = -Col(primal, j) * blaze::pow(Col(primal, i), T{3}) / (Col(primal, i-1) * Col(primal, i-1));
        }
    }

    template<typename T, std::size_t S>
    auto Pow(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        if (j == i-1) {
            auto const k = j - (nodes[j].Length + 1);
            Col(trace, j) = Col(primal, i) * Col(primal, k) / Col(primal, j);
        } else {
            auto const k = i-1;
            Col(trace, j) = Col(primal, i) * blaze::log(Col(primal, k));
        }
    }

    template<typename T, std::size_t S>
    auto Min(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto k = j == i - 1 ? (j - nodes[j].Length - 1) : i - 1;
        auto const* a = Ptr(primal, j);
        std::transform(a, a + S, Ptr(primal, k), Ptr(trace, j), detail::FComp<std::less<>>{});
    }

    template<typename T, std::size_t S>
    auto Max(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto k = j == i - 1 ? (j - nodes[j].Length - 1) : i - 1;
        auto const* a = Ptr(primal, j);
        std::transform(a, a + S, Ptr(primal, k), Ptr(trace, j), detail::FComp<std::greater<>>{});
    }

    template<typename T, std::size_t S>
    auto Square(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{2} * Col(primal, j);
    }

    template<typename T, std::size_t S>
    auto Abs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = blaze::sign(Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Ceil(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = blaze::ceil(Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Floor(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = blaze::floor(Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Exp(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        // fmt::print("dexp 1: {} => {}\n", std::span{Ptr(trace, j), S}, std::span{Ptr(primal, i), S});
        // Col(trace, j) = Col(primal, i);
        std::copy_n(Ptr(primal, i), S, Ptr(trace, j));
        // fmt::print("dexp 2: {} => {}\n", std::span{Ptr(trace, j), S}, std::span{Ptr(primal, i), S});
    }

    template<typename T, std::size_t S>
    auto Log(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{1} / (Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Log1p(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{1} / (T{1} + Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Logabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = blaze::sign(Col(primal, j)) / blaze::abs(Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Sin(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = blaze::cos(Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Cos(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = -blaze::sin(Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Tan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        blaze::DynamicVector<T> t = blaze::tan(Col(primal, j));
        Col(trace, j) = T{1} + t*t;
    }

    template<typename T, std::size_t S>
    auto Sinh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = blaze::cosh(Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Cosh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = blaze::sinh(Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Tanh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{1} - blaze::pow(blaze::tanh(Col(primal, j)), T{2});
    }

    template<typename T, std::size_t S>
    auto Asin(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{1} / (blaze::sqrt(T{1} - Col(primal, j) * Col(primal, j)));
    }

    template<typename T, std::size_t S>
    auto Acos(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = -T{1} / (blaze::sqrt(T{1} - Col(primal, j) * Col(primal, j)));
    }

    template<typename T, std::size_t S>
    auto Atan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{1} / (T{1} + Col(primal, j) * Col(primal, j));
    }

    template<typename T, std::size_t S>
    auto Sqrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = T{1} / (T{2} * Col(primal, i));
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = blaze::sign(Col(primal, j)) / (T{2} * Col(primal, i));
    }

    template<typename T, std::size_t S>
    auto Cbrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = T{1} / (T{3} * Col(primal, i) * Col(primal, i));
    }
}  // namespace Operon::Backend

#endif
