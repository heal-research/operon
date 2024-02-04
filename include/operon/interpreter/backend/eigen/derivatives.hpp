// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_EIGEN_DERIVATIVES_HPP
#define OPERON_BACKEND_EIGEN_DERIVATIVES_HPP

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
        Col(trace, j).setConstant(T{1});
    }

    template<typename T, std::size_t S>
    auto Mul(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = Col(primal, i) / Col(primal, j);
    }

    template<typename T, std::size_t S>
    auto Sub(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> /*primal*/, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto v = (nodes[i].Arity == 1 || j < i-1) ? T{-1} : T{+1};
        Col(trace, j).setConstant(v);
    }

    template<typename T, std::size_t S>
    auto Div(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto const& n = nodes[i];
        if (n.Arity == 1) {
            Col(trace, j) = -Col(primal, j).square().inverse();
        } else {
            Col(trace, j) = (j == i-1 ? T{1} : T{-1}) * Col(primal, i) / Col(primal, j);
        }
    }

    template<typename T, std::size_t S>
    auto Aq(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        if (j == i-1) {
            Col(trace, j) = Col(primal, i) / Col(primal, j);
        } else {
            Col(trace, j) = -Col(primal, j) * Col(primal, i).pow(T{3}) / Col(primal, i-1).square();
        }
    }

    template<typename T, std::size_t S>
    auto Pow(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        if (j == i-1) {
            auto const k = j - (nodes[j].Length + 1);
            Col(trace, j) = Col(primal, i) * Col(primal, k) / Col(primal, j);
        } else {
            auto const k = i-1;
            Col(trace, j) = Col(primal, i) * Col(primal, k).log();
        }
    }

    template<typename T, std::size_t S>
    auto Min(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto k = j == i - 1 ? (j - nodes[j].Length - 1) : i - 1;
        Col(trace, j) = Col(primal, j).binaryExpr(Col(primal, k), detail::FComp<std::less<>>{});
    }

    template<typename T, std::size_t S>
    auto Max(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        auto k = j == i - 1 ? (j - nodes[j].Length - 1) : i - 1;
        Col(trace, j) = Col(primal, j).binaryExpr(Col(primal, k), detail::FComp<std::greater<>>{});
    }

    template<typename T, std::size_t S>
    auto Square(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{2} * Col(primal, j);
    }

    template<typename T, std::size_t S>
    auto Abs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = Col(primal, j).sign();
    }

    template<typename T, std::size_t S>
    auto Ceil(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = Col(primal, j).ceil();
    }

    template<typename T, std::size_t S>
    auto Floor(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = Col(primal, j).floor();
    }

    template<typename T, std::size_t S>
    auto Exp(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = Col(primal, i);
    }

    template<typename T, std::size_t S>
    auto Log(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = Col(primal, j).inverse();
    }

    template<typename T, std::size_t S>
    auto Log1p(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = (T{1} + Col(primal, j)).inverse();
    }

    template<typename T, std::size_t S>
    auto Logabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = Col(primal, j).sign() / Col(primal, j).abs();
    }

    template<typename T, std::size_t S>
    auto Sin(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = Col(primal, j).cos();
    }

    template<typename T, std::size_t S>
    auto Cos(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = -Col(primal, j).sin();
    }

    template<typename T, std::size_t S>
    auto Tan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{1} + Col(primal, j).tan().square();
    }

    template<typename T, std::size_t S>
    auto Sinh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = Col(primal, j).cosh();
    }

    template<typename T, std::size_t S>
    auto Cosh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = Col(primal, j).sinh();
    }

    template<typename T, std::size_t S>
    auto Tanh(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = T{1} - Col(primal, j).tanh().square();
    }

    template<typename T, std::size_t S>
    auto Asin(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = (T{1} - Col(primal, j).square()).sqrt().inverse();
    }

    template<typename T, std::size_t S>
    auto Acos(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = -((T{1} - Col(primal, j).square()).sqrt().inverse());
    }

    template<typename T, std::size_t S>
    auto Atan(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto /*i*/, std::integral auto j) {
        Col(trace, j) = (T{1} + Col(primal, j).square()).inverse();
    }

    template<typename T, std::size_t S>
    auto Sqrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = (T{2} * Col(primal, i)).inverse();
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = Col(primal, j).sign() / (T{2} * Col(primal, i));
    }

    template<typename T, std::size_t S>
    auto Cbrt(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
        Col(trace, j) = (T{3} * Col(primal, i).square()).inverse();
    }
}  // namespace Operon::Backend

#endif