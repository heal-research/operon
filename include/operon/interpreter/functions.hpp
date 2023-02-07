// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_FUNCTIONS_HPP
#define OPERON_INTERPRETER_FUNCTIONS_HPP

#include "operon/core/node.hpp"
#include "operon/ceres/jet.h" // for ceres::cbrt

namespace Operon
{
    template<Operon::NodeType = NodeType::Add, bool Continued = false>
    struct Func {
        auto operator()(auto... args) { return (args + ...); }
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Mul, Continued> {
        auto operator()(auto... args) { return (args * ...); }
    };

    template<bool B>
    struct Func<Operon::NodeType::Sub, B> {
        auto operator()(auto first, auto... rest) {
            if constexpr (sizeof...(rest) == 0) {
                return -first;
            } else if constexpr (B) {
                return -first - (rest + ...);
            } else {
                return first - (rest + ...);
            }
        }
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Div, Continued> {
        auto operator()(auto first, auto... rest) {
            if constexpr (sizeof...(rest) == 0) {
                return first.inverse();
            } else if constexpr (Continued) {
                return (first * (rest * ...));
            } else {
                return first / (rest * ...);
            }
        }
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Fmin, Continued> {
        auto operator()(auto first, auto... rest) {
            if constexpr (sizeof...(rest) == 0) { return first; }
            else { return (first.min(rest), ...); }
        };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Fmax, Continued> {
        auto operator()(auto first, auto... rest) {
            if constexpr (sizeof...(rest) == 0) { return first; }
            else { return (first.max(rest), ...); }
        };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Aq, Continued> {
        auto operator()(auto a, auto b) {
            auto x = typename decltype(a)::Scalar{1};
            return a / (x + b.square()).sqrt();
        };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Pow, Continued> {
        auto operator()(auto a, auto b) {
            return a.pow(b);
        };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Abs, Continued> {
        auto operator()(auto a) { return a.abs(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Log, Continued> {
        auto operator()(auto a) { return a.log(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Logabs, Continued> {
        auto operator()(auto a) { return a.abs().log(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Log1p, Continued> {
        auto operator()(auto a) { return a.log1p(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Exp, Continued> {
        auto operator()(auto a) { return a.exp(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Ceil, Continued> {
        auto operator()(auto a) { return a.ceil(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Floor, Continued> {
        auto operator()(auto a) { return a.floor(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sin, Continued> {
        auto operator()(auto a) { return a.sin(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Cos, Continued> {
        auto operator()(auto a) { return a.cos(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Tan, Continued> {
        auto operator()(auto a) { return a.tan(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Asin, Continued> {
        auto operator()(auto a) { return a.asin(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Acos, Continued> {
        auto operator()(auto a) { return a.acos(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Atan, Continued> {
        auto operator()(auto a) { return a.atan(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sinh, Continued> {
        auto operator()(auto a) { return a.sinh(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Cosh, Continued> {
        auto operator()(auto a) { return a.cosh(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Tanh, Continued> {
        auto operator()(auto a) { return a.tanh(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sqrt, Continued> {
        auto operator()(auto a) { return a.sqrt(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sqrtabs, Continued> {
        auto operator()(auto a) { return a.abs().sqrt(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Cbrt, Continued> {
        auto operator()(auto a) {
            using T = typename decltype(a)::Scalar;
            return a.unaryExpr([](T const& v) { return ceres::cbrt(v); });
        };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Square, Continued> {
        auto operator()(auto a) { return a.square(); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Dynamic, Continued> {
        auto operator()(auto /*unused*/) { /* nothing */ };
    };
} // namespace Operon

#endif
