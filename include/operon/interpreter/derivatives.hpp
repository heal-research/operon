// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_DERIVATIVES_HPP
#define OPERON_INTERPRETER_DERIVATIVES_HPP

#include "functions.hpp"
#include "dual.hpp"
#include <functional>

namespace Operon {
namespace detail {
    template<typename T>
    inline auto IsNaN(T value) { return std::isnan(value); }

    template<>
    inline auto IsNaN<Operon::Dual>(Operon::Dual value) { return ceres::isnan(value); } // NOLINT

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

    template<Operon::NodeType N = Operon::NodeType::Add>
    struct Diff {
        auto operator()(auto const& /*nodes*/, auto const& /*primal*/, auto& trace, int /*parent*/, int j) {
            trace.col(j).setConstant(scalar_t<decltype(trace)>{1});
        }
    };

    template<Operon::NodeType N = Operon::NodeType::Add>
    struct Diff2 {
        auto operator()(auto const& /*nodes*/, auto const& /*primal*/, auto& trace, int /*parent*/, int j) {
            trace.col(j).setConstant(scalar_t<decltype(trace)>{0});
        }
    };

    template<>
    struct Diff<Operon::NodeType::Mul> {
        auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, int i, int j) {
            trace.col(j) = primal.col(i) / primal.col(j);
        }
    };

    template<>
    struct Diff2<Operon::NodeType::Mul> {
        auto operator()(auto const& /*nodes*/, auto const& /*primal*/, auto& trace, int /*parent*/, int j) {
            trace.col(j).setConstant(scalar_t<decltype(trace)>{1});
        }
    };

    template<>
    struct Diff<Operon::NodeType::Sub> {
        auto operator()(auto const& nodes, auto const& /*primal*/, auto& trace, int i, int j) {
            auto v = (nodes[i].Arity == 1 || j < i-1) ? -1 : +1;
            trace.col(j).setConstant(scalar_t<decltype(trace)>(v));
        }
    };

    template<>
    struct Diff2<Operon::NodeType::Sub> {
        auto operator()(auto const& /*nodes*/, auto const& /*primal*/, auto& trace, int /*parent*/, int j) {
            trace.col(j).setConstant(scalar_t<decltype(trace)>{0});
        }
    };

    template<>
    struct Diff<Operon::NodeType::Div> {
        auto operator()(auto const& nodes, auto const& primal, auto& trace, int i, int j) {
            auto const& n = nodes[i];

            if (n.Arity == 1) {
                trace.col(j) = -primal.col(j).square().inverse();
            } else {
                auto v = scalar_t<decltype(trace)>{1.0};
                trace.col(j) = (j == i-1 ? +v : -v) * primal.col(i) / primal.col(j); 
            }
        }
    };

    template<>
    struct Diff2<Operon::NodeType::Div> {
        auto operator()(auto const& nodes, auto const& primal, auto& trace, int i, int j) {
            auto const& n = nodes[i];

            if (n.Arity == 1) {
                trace.col(j) = 2 * primal.col(j).pow(scalar_t<decltype(trace)>{3.0}).inverse();
            } else {
                auto v = scalar_t<decltype(trace)>{1.0};
                trace.col(j) = (j == i-1 ? +v : -v) * primal.col(i) / primal.col(j); 
            }
        }
    };

    template <>
    struct Diff<Operon::NodeType::Aq> {
        auto operator()(auto const& nodes, auto const& primal, auto& trace, int i, int j) {
            auto const& n = nodes[i];

            if (j == i-1) {
                // first arg
                trace.col(j) = primal.col(i) / primal.col(j);
            } else {
                // second arg
                trace.col(j) = -primal.col(j) * primal.col(i).pow(scalar_t<decltype(trace)>(3)) / primal.col(i-1).square();
            }
        }
    };

    template <>
    struct Diff<Operon::NodeType::Pow> {
        auto operator()(auto const& nodes, auto const& primal, auto& trace, auto i, auto j)
        {
            if (j == i-1) {
                // first arg
                auto const k = j - (nodes[j].Length + 1);
                trace.col(j) = primal.col(i) * primal.col(k) / primal.col(j);
            } else {
                // second arg
                auto const k = i - 1;
                trace.col(j) = primal.col(i) * primal.col(k).log();
            }
        }
    };

    template <>
    struct Diff<Operon::NodeType::Fmin> {
        auto operator()(auto const& nodes, auto const& primal, auto& trace, auto i, auto j)
        {
            auto k = j == i-1 ? (j - nodes[j].Length - 1) : i-1;
            trace.col(j) = primal.col(j).binaryExpr(primal.col(k), detail::FComp<std::less<>>{});
        }
    };

    template <>
    struct Diff<Operon::NodeType::Fmax> {
        auto operator()(auto const& nodes, auto const& primal, auto& trace, auto i, auto j)
        {
            auto k = j == i-1 ? (j - nodes[j].Length - 1) : i-1;
            trace.col(j) = primal.col(j).binaryExpr(primal.col(k), detail::FComp<std::greater<>>{});
        }
    };

    template <>
    struct Diff<Operon::NodeType::Square> {
        auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = scalar_t<decltype(trace)>{2} * primal.col(j);
        }
    };

    template <>
    struct Diff<Operon::NodeType::Abs> {
        auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = primal.col(j).sign();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Exp> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto i, auto j)
        {
            trace.col(j) = primal.col(i);
        }
    };

    template <>
    struct Diff<Operon::NodeType::Log> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = primal.col(j).inverse();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Logabs> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = primal.col(j).sign() / primal.col(j).abs();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Log1p> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = (primal.col(j) + scalar_t<decltype(trace)>{1}).inverse();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Sin> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = primal.col(j).cos();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Sinh> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = primal.col(j).cosh();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Cos> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = -primal.col(j).sin();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Cosh> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = primal.col(j).sinh();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Tan> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = scalar_t<decltype(trace)>{1} + primal.col(j).tan().square();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Tanh> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = scalar_t<decltype(trace)>{1} - primal.col(j).tanh().square();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Asin> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = (scalar_t<decltype(trace)>{1} - primal.col(j).square()).sqrt().inverse();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Acos> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = -(scalar_t<decltype(trace)>{1} - primal.col(j).square()).sqrt().inverse();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Atan> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto /*i*/, auto j)
        {
            trace.col(j) = (scalar_t<decltype(trace)>{1} + primal.col(j).square()).inverse();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Sqrt> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto i, auto j)
        {
            trace.col(j) = (scalar_t<decltype(trace)>{2} * primal.col(i)).inverse();
        }
    };

    template <>
    struct Diff<Operon::NodeType::Sqrtabs> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto i, auto j)
        {
            trace.col(j) = sign(primal.col(j)) / (scalar_t<decltype(trace)>{2} * primal.col(i));
        }
    };

    template <>
    struct Diff<Operon::NodeType::Cbrt> {
        inline auto operator()(auto const& /*nodes*/, auto const& primal, auto& trace, auto i, auto j)
        {
            trace.col(j) = (scalar_t<decltype(trace)>{3} * (primal.col(i)).square()).inverse();
        }
    };
} // namespace Operon

#endif
