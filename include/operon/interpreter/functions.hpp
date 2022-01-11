// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_INTERPRETER_FUNCTIONS_HPP
#define OPERON_INTERPRETER_FUNCTIONS_HPP

#include "operon/core/node.hpp"
#include <ceres/jet.h>

namespace Operon
{
    // addition up to 5 arguments
    template<Operon::NodeType N = NodeType::Add>
    struct Function 
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a; }

        template<typename T, typename... Tn>
        inline void operator()(T r, T a1, Tn... an) { r = a1 + (an + ...); }
    };

    template<>
    struct Function<NodeType::Sub>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = -a; }

        template<typename T, typename... Tn>
        inline void operator()(T r, T a1, Tn... an) { r = a1 - (an + ...); }
    };

    template<>
    struct Function<NodeType::Mul>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a; }

        template<typename T, typename... Tn>
        inline void operator()(T r, T a1, Tn... an) { r = a1 * (an * ...); }
    };

    template<>
    struct Function<NodeType::Div>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.inverse(); }

        template<typename T, typename... Tn>
        inline void operator()(T r, T a1, Tn... an) { r = a1 / (an * ...); }
    };

    // continuations for n-ary functions (add, sub, mul, div)
    template<NodeType N = NodeType::Add>
    struct ContinuedFunction
    {
        template<typename T>
        inline void operator()(T r, T a) { r += a; }

        template<typename T, typename... Ts>
        inline void operator()(T r, T a1, Ts... an) { r += a1 + (an + ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Sub>
    {
        template<typename T>
        inline void operator()(T r, T a) { r -= a; }

        template<typename T, typename... Ts>
        inline void operator()(T r, T a1, Ts... an) { r -= a1 + (an + ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Mul>
    {
        template<typename T>
        inline void operator()(T r, T a) { r *= a; }

        template<typename T, typename... Ts>
        inline void operator()(T r, T a1, Ts... an) { r *= a1 * (an * ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Div>
    {
        template<typename T>
        inline void operator()(T r, T t) { r /= t; }

        template<typename T, typename... Ts>
        inline void operator()(T r, T a1, Ts... an) { r /= a1 * (an * ...); }
    };

    // binary and unary functions
    template<>
    struct Function<NodeType::Aq>
    {
        template<typename T>
        inline void operator()(T r, T a1, T a2) { r = a1 / (typename T::Scalar{1.0} + a2.square()).sqrt(); }
    };

    template<>
    struct Function<NodeType::Pow>
    {
        template<typename T>
        inline void operator()(T r, T a1, T a2) { r = a1.pow(a2); }
    };

    template<>
    struct Function<NodeType::Log>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.log(); }
    };

    template<>
    struct Function<NodeType::Exp>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.exp(); }
    };

    template<>
    struct Function<NodeType::Sin>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.sin(); }
    };

    template<>
    struct Function<NodeType::Cos>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.cos(); }
    };

    template<>
    struct Function<NodeType::Tan>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.tan(); }
    };

    template<>
    struct Function<NodeType::Tanh>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.tanh(); }
    };

    template<>
    struct Function<NodeType::Sqrt>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.sqrt(); }
    };

    template<>
    struct Function<NodeType::Cbrt>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.unaryExpr([](typename T::Scalar const& v) { return ceres::cbrt(v); }); }
    };

    template<>
    struct Function<NodeType::Square>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.square(); }
    };

    template<>
    struct Function<NodeType::Dynamic>
    {
        template<typename T>
        inline void operator()(T /*unused*/, T /*unused*/) { /* do nothing */ }
    };
} // namespace Operon

#endif
