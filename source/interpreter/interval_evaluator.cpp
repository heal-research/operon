// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/interpreter/interval_evaluator.hpp"

namespace Operon {

template<typename T>
auto IntervalUnaryRules() -> IntervalUnaryRegistry<T>&
{
    static IntervalUnaryRegistry<T> registry;
    return registry;
}

template<typename T>
auto IntervalBinaryRules() -> IntervalBinaryRegistry<T>&
{
    static IntervalBinaryRegistry<T> registry;
    return registry;
}

template<typename T>
void RegisterIntervalBuiltins()
{
    using Interval = pappus::interval<T>;
    static auto const registered = [] {
        auto& unary  = IntervalUnaryRules<T>();
        auto& binary = IntervalBinaryRules<T>();

        unary.Register(Operon::Hash(BuiltinOp::Square),  [](Interval const& v) { return pappus::ops::square<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrt),    [](Interval const& v) { return pappus::ops::sqrt<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Exp),     [](Interval const& v) { return pappus::ops::exp<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Log),     [](Interval const& v) { return pappus::ops::log<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sin),     [](Interval const& v) { return pappus::ops::sin<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cos),     [](Interval const& v) { return pappus::ops::cos<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Tan),     [](Interval const& v) { return pappus::ops::tan<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Asin),    [](Interval const& v) { return pappus::ops::asin<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Acos),    [](Interval const& v) { return pappus::ops::acos<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Atan),    [](Interval const& v) { return pappus::ops::atan<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sinh),    [](Interval const& v) { return pappus::ops::sinh<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cosh),    [](Interval const& v) { return pappus::ops::cosh<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Tanh),    [](Interval const& v) { return pappus::ops::tanh<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Abs),     [](Interval const& v) { return pappus::ops::abs<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrtabs), [](Interval const& v) { return pappus::ops::sqrtabs<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Logabs),  [](Interval const& v) { return pappus::ops::logabs<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cbrt),    [](Interval const& v) { return pappus::ops::cbrt<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Log1p),   [](Interval const& v) { return pappus::ops::log1p<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Floor),   [](Interval const& v) { return pappus::ops::floor<T>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Ceil),    [](Interval const& v) { return pappus::ops::ceil<T>(v); });

        binary.Register(Operon::Hash(BuiltinOp::Pow), [](Interval const& a, Interval const& b) {
            // degenerate exponent: dispatch through pow(interval, Scalar), which
            // detects an integer exponent and avoids restricting the base to >= 0
            if (b.inf() == b.sup()) { return pappus::ops::pow<T>(a, b.inf()); }
            return pappus::ops::pow<T>(a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Aq), [](Interval const& a, Interval const& b) {
            return pappus::ops::aq<T>(a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Powabs), [](Interval const& a, Interval const& b) {
            return pappus::ops::pow<T>(pappus::ops::abs<T>(a), b);
        });

        return true;
    }();
    static_cast<void>(registered);
}

template<typename T>
void RegisterUnaryInterval(Operon::Hash hash, IntervalUnaryFn<T> fn)
{
    RegisterIntervalBuiltins<T>();
    IntervalUnaryRules<T>().Register(hash, std::move(fn));
}

template<typename T>
void RegisterBinaryInterval(Operon::Hash hash, IntervalBinaryFn<T> fn)
{
    RegisterIntervalBuiltins<T>();
    IntervalBinaryRules<T>().Register(hash, std::move(fn));
}

template<typename T>
auto HasUnaryInterval(Operon::Hash hash) -> bool
{
    RegisterIntervalBuiltins<T>();
    return IntervalUnaryRules<T>().Contains(hash);
}

template<typename T>
auto HasBinaryInterval(Operon::Hash hash) -> bool
{
    RegisterIntervalBuiltins<T>();
    return IntervalBinaryRules<T>().Contains(hash);
}

template auto OPERON_EXPORT IntervalUnaryRules<Operon::Scalar>() -> IntervalUnaryRegistry<Operon::Scalar>&;
template auto OPERON_EXPORT IntervalBinaryRules<Operon::Scalar>() -> IntervalBinaryRegistry<Operon::Scalar>&;
template void OPERON_EXPORT RegisterIntervalBuiltins<Operon::Scalar>();
template void OPERON_EXPORT RegisterUnaryInterval<Operon::Scalar>(Operon::Hash, IntervalUnaryFn<Operon::Scalar>);
template void OPERON_EXPORT RegisterBinaryInterval<Operon::Scalar>(Operon::Hash, IntervalBinaryFn<Operon::Scalar>);
template auto OPERON_EXPORT HasUnaryInterval<Operon::Scalar>(Operon::Hash) -> bool;
template auto OPERON_EXPORT HasBinaryInterval<Operon::Scalar>(Operon::Hash) -> bool;

template auto OPERON_EXPORT IntervalUnaryRules<eve::wide<Operon::Scalar>>() -> IntervalUnaryRegistry<eve::wide<Operon::Scalar>>&;
template auto OPERON_EXPORT IntervalBinaryRules<eve::wide<Operon::Scalar>>() -> IntervalBinaryRegistry<eve::wide<Operon::Scalar>>&;
template void OPERON_EXPORT RegisterIntervalBuiltins<eve::wide<Operon::Scalar>>();

} // namespace Operon
