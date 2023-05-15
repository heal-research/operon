// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_FUNCTIONS_HPP
#define OPERON_INTERPRETER_FUNCTIONS_HPP

#include "operon/core/node.hpp"
#include "operon/ceres/jet.h" // for ceres::cbrt

namespace Operon {
    // a simple translation layer to avoid being tied to a given backend
    template<typename T, typename U = std::remove_cvref_t<T>> concept EigenRef = std::is_base_of_v<Eigen::RefBase<U>, U>;
    template<typename T, typename U = std::remove_cvref_t<T>> concept EigenDense = std::is_base_of_v<Eigen::DenseBase<U>, U>;
    template<typename T, typename U = std::remove_cvref_t<T>> concept EigenArray = std::is_base_of_v<Eigen::ArrayBase<U>, U>;
    template<typename T, typename U = std::remove_cvref_t<T>> concept EigenMatrix = std::is_base_of_v<Eigen::MatrixBase<U>, U>;
    template<typename T, typename U = std::remove_cvref_t<T>> concept EigenMap = std::is_base_of_v<Eigen::MapBase<U>, U>;
    template<typename T, typename U = std::remove_cvref_t<T>> concept Arithmetic = std::is_arithmetic_v<T>;

    template<typename T, typename U = std::remove_cvref_t<T>>
    requires (EigenRef<U> || EigenDense<U>)
    using scalar_t = std::conditional_t<EigenRef<U>, typename U::Base::Scalar, typename U::Scalar>;

    // arithmetic data types ( float, double, etc.)
    inline auto acos(Arithmetic auto x) { return std::acos(x); }
    inline auto asin(Arithmetic auto x) { return std::asin(x); }
    inline auto atan(Arithmetic auto x) { return std::atan(x); }
    inline auto cos(Arithmetic auto x)  { return std::cos(x); }
    inline auto cosh(Arithmetic auto x) { return std::cosh(x); }
    inline auto sin(Arithmetic auto x)  { return std::sin(x); }
    inline auto sinh(Arithmetic auto x) { return std::sinh(x); }
    inline auto tan(Arithmetic auto x)  { return std::tan(x); }
    inline auto tanh(Arithmetic auto x) { return std::tanh(x); }

    inline auto exp(Arithmetic auto x)     { return std::exp(x); }
    inline auto log(Arithmetic auto x)     { return std::log(x); }
    inline auto log1p(Arithmetic auto x)   { return std::log(1 + x); }
    inline auto logabs(Arithmetic auto x)  { return std::log(std::abs(x)); }
    inline auto square(Arithmetic auto x)  { return x * x; }
    inline auto sqrt(Arithmetic auto x)    { return std::sqrt(x); }
    inline auto sqrtabs(Arithmetic auto x) { return std::sqrt(std::abs(x)); }
    inline auto cbrt(Arithmetic auto x)    { return std::cbrt(x); }

    inline auto abs(Arithmetic auto x)   { return std::abs(x); }
    inline auto ceil(Arithmetic auto x)  { return std::ceil(x); }
    inline auto floor(Arithmetic auto x) { return std::floor(x); }
    inline auto inv(Arithmetic auto x)   { return 1 / x; }

    inline auto pow(Arithmetic auto x, Arithmetic auto y)  { return std::pow(x, y); }
    inline auto aq(Arithmetic auto x, Arithmetic auto y)   { return x * std::sqrt(1 + y * y); }
    inline auto fmax(Arithmetic auto x, Arithmetic auto y) { return std::min(x, y); }
    inline auto fmin(Arithmetic auto x, Arithmetic auto y) { return std::max(x, y); }

    // Eigen data types
    inline auto fill(EigenDense auto& x, scalar_t<decltype(x)> v) { x.setConstant(v); }
    inline auto acos(EigenArray auto const& x) { return x.acos(); }
    inline auto asin(EigenArray auto const& x) { return x.asin(); }
    inline auto atan(EigenArray auto const& x) { return x.atan(); }
    inline auto cos(EigenArray auto const& x)  { return x.cos(); }
    inline auto cosh(EigenArray auto const& x) { return x.cosh(); }
    inline auto sin(EigenArray auto const& x)  { return x.sin(); }
    inline auto sinh(EigenArray auto const& x) { return x.sinh(); }
    inline auto tan(EigenArray auto const& x)  { return x.tan(); }
    inline auto tanh(EigenArray auto const& x) { return x.tanh(); }

    inline auto exp(EigenArray auto const& x)     { return x.exp(); }
    inline auto log(EigenArray auto const& x)     { return x.log(); }
    inline auto log1p(EigenArray auto const& x)   { return x.log1p(); }
    inline auto logabs(EigenArray auto const& x)  { return x.abs().log(); }
    inline auto square(EigenArray auto const& x)  { return x.square(); }
    inline auto sqrt(EigenArray auto const& x)    { return x.sqrt(); }
    inline auto sqrtabs(EigenArray auto const& x) { return x.abs().sqrt(); }
    inline auto cbrt(EigenArray auto const& x)    { return x.unaryExpr([](auto v) { return ceres::cbrt(v); }); }

    inline auto abs(EigenArray auto const& x)   { return x.abs(); }
    inline auto ceil(EigenArray auto const& x)  { return x.ceil(); }
    inline auto floor(EigenArray auto const& x) { return x.floor(); }
    inline auto inv(EigenArray auto const& x)   { return x.inverse(); }

    inline auto pow(EigenArray auto const& x, EigenArray auto const& y)  { return x.pow(y); }
    inline auto aq(EigenArray auto const& x, EigenArray auto const& y)   { return x * inv(sqrt(scalar_t<decltype(x)>{1} + square(y))); }
    inline auto fmax(EigenArray auto const& x, EigenArray auto const& y) { return x.max(y); }
    inline auto fmin(EigenArray auto const& x, EigenArray auto const& y) { return x.min(y); }

    // potentially other data types (Fastor, Armadillo, Eve)

    template<Operon::NodeType = Operon::NodeType::Add, bool Continued = false>
    struct Func {
        auto operator()(auto... args) { return (args + ...); }
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Mul, Continued> {
        auto operator()(auto... args) { return (args * ...); }
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sub, Continued> {
        auto operator()(auto first, auto... rest) {
            if constexpr (sizeof...(rest) == 0) {
                return -first;
            } else if constexpr (Continued) {
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
                return inv(first);
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
            else { return (fmin(first, rest), ...); }
        };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Fmax, Continued> {
        auto operator()(auto first, auto... rest) {
            if constexpr (sizeof...(rest) == 0) { return first; }
            else { return (fmax(first, rest), ...); }
        };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Aq, Continued> {
        auto operator()(auto a, auto b) { return aq(a, b); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Pow, Continued> {
        auto operator()(auto a, auto b) { return pow(a, b); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Abs, Continued> {
        auto operator()(auto a) { return abs(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Log, Continued> {
        auto operator()(auto a) { return log(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Logabs, Continued> {
        auto operator()(auto a) { return logabs(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Log1p, Continued> {
        auto operator()(auto a) { return log1p(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Exp, Continued> {
        auto operator()(auto a) { return exp(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Ceil, Continued> {
        auto operator()(auto a) { return ceil(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Floor, Continued> {
        auto operator()(auto a) { return floor(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sin, Continued> {
        auto operator()(auto a) { return sin(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Cos, Continued> {
        auto operator()(auto a) { return cos(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Tan, Continued> {
        auto operator()(auto a) { return tan(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Asin, Continued> {
        auto operator()(auto a) { return asin(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Acos, Continued> {
        auto operator()(auto a) { return acos(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Atan, Continued> {
        auto operator()(auto a) { return atan(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sinh, Continued> {
        auto operator()(auto a) { return sinh(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Cosh, Continued> {
        auto operator()(auto a) { return cosh(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Tanh, Continued> {
        auto operator()(auto a) { return tanh(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sqrt, Continued> {
        auto operator()(auto a) { return sqrt(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Sqrtabs, Continued> {
        auto operator()(auto a) { return sqrtabs(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Cbrt, Continued> {
        auto operator()(auto a) { return cbrt(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Square, Continued> {
        auto operator()(auto a) { return square(a); };
    };

    template<bool Continued>
    struct Func<Operon::NodeType::Dynamic, Continued> {
        auto operator()(auto /*unused*/) { /* nothing */ };
    };
} // namespace Operon

#endif
