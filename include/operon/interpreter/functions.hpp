// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_INTERPRETER_FUNCTIONS_HPP
#define OPERON_INTERPRETER_FUNCTIONS_HPP

#include "operon/core/node.hpp"
#include "operon/ceres/jet.h" // for ceres::cbrt 

namespace Operon
{
    template<Operon::NodeType N = NodeType::Add>
    struct Function 
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t; }

        template<typename R, typename T, typename... Tn>
        inline void operator()(R r, T t1, Tn... tn) { r = t1 + (tn + ...); }
    };

    template<>
    struct Function<NodeType::Sub>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = -t; }

        template<typename R, typename T, typename... Tn>
        inline void operator()(R r, T t1, Tn... tn) { r = t1 - (tn + ...); }
    };

    template<>
    struct Function<NodeType::Mul>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t; }

        template<typename R, typename T, typename... Tn>
        inline void operator()(R r, T t1, Tn... tn) { r = t1 * (tn * ...); }
    };

    template<>
    struct Function<NodeType::Div>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.inverse(); }

        template<typename R, typename T, typename... Tn>
        inline void operator()(R r, T t1, Tn... tn) { r = t1 / (tn * ...); }
    };

    template<>
    struct Function<NodeType::Fmin>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t; }
        
        template<typename R, typename T, typename... Tn>
        inline void operator()(R r, T t1, Tn... tn) { r = (t1.min(tn), ...); }
    };

    template<>
    struct Function<NodeType::Fmax>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t; }
        
        template<typename R, typename T, typename... Tn>
        inline void operator()(R r, T t1, Tn... tn) { r = (t1.max(tn), ...); }
    };

    // continuations for n-ary functions (add, sub, mul, div, fmin, fmax)
    template<NodeType N = NodeType::Add>
    struct ContinuedFunction
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r += t; }

        template<typename R, typename T, typename... Ts>
        inline void operator()(R r, T t1, Ts... tn) { r += t1 + (tn + ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Sub>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r -= t; }

        template<typename R, typename T, typename... Ts>
        inline void operator()(R r, T t1, Ts... tn) { r -= t1 + (tn + ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Mul>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r *= t; }

        template<typename R, typename T, typename... Ts>
        inline void operator()(R r, T t1, Ts... tn) { r *= t1 * (tn * ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Div>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r /= t; }

        template<typename R, typename T, typename... Ts>
        inline void operator()(R r, T t1, Ts... tn) { r /= t1 * (tn * ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Fmin>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = r.min(t); }

        template<typename R, typename T, typename... Ts>
        inline void operator()(R r, T t1, Ts... tn) { r = r.min((t1.min(tn), ...)); }
    };

    template<>
    struct ContinuedFunction<NodeType::Fmax>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = r.max(t); }

        template<typename R, typename T, typename... Ts>
        inline void operator()(R r, T t1, Ts... tn) { r = r.max((t1.max(tn), ...)); }
    };

    // bin... and unary functions
    template<>
    struct Function<NodeType::Aq>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t1, T t2) { r = t1 / (typename T::Scalar{1.0} + t2.square()).sqrt(); }
    };

    template<>
    struct Function<NodeType::Pow>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t1, T t2) { r = t1.pow(t2); }
    };

    template<>
    struct Function<NodeType::Abs>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.abs(); }
    };

    template<>
    struct Function<NodeType::Log>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.log(); }
    };

    template<>
    struct Function<NodeType::Logabs>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.abs().log(); }
    };

    template<>
    struct Function<NodeType::Log1p>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.log1p(); }
    };

    template<>
    struct Function<NodeType::Ceil>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.ceil(); }
    };

    template<>
    struct Function<NodeType::Floor>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.floor(); }
    };

    template<>
    struct Function<NodeType::Exp>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.exp(); }
    };

    template<>
    struct Function<NodeType::Sin>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.sin(); }
    };

    template<>
    struct Function<NodeType::Cos>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.cos(); }
    };

    template<>
    struct Function<NodeType::Tan>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.tan(); }
    };

    template<>
    struct Function<NodeType::Asin>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.asin(); }
    };

    template<>
    struct Function<NodeType::Acos>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.acos(); }
    };

    template<>
    struct Function<NodeType::Atan>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.atan(); }
    };

    template<>
    struct Function<NodeType::Sinh>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.sinh(); }
    };

    template<>
    struct Function<NodeType::Cosh>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.cosh(); }
    };

    template<>
    struct Function<NodeType::Tanh>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.tanh(); }
    };

    template<>
    struct Function<NodeType::Sqrt>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.sqrt(); }
    };

    template<>
    struct Function<NodeType::Sqrtabs>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.abs().sqrt(); }
    };

    template<>
    struct Function<NodeType::Cbrt>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.unaryExpr([](typename T::Scalar const& v) { return ceres::cbrt(v); }); }
    };

    template<>
    struct Function<NodeType::Square>
    {
        template<typename R, typename T>
        inline void operator()(R r, T t) { r = t.square(); }
    };

    template<>
    struct Function<NodeType::Dynamic>
    {
        template<typename R, typename T>
        inline void operator()(R /*unused*/, T /*unused*/) { /* do nothing */ }
    };
} // namespace Operon

#endif
