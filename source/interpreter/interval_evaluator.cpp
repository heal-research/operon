// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/interpreter/interval_evaluator.hpp"

namespace Operon {

auto IntervalUnaryRules() -> IntervalUnaryRegistry&
{
    static IntervalUnaryRegistry registry;
    return registry;
}

auto IntervalBinaryRules() -> IntervalBinaryRegistry&
{
    static IntervalBinaryRegistry registry;
    return registry;
}

void RegisterIntervalBuiltins()
{
    using Scalar = Operon::Scalar;
    using Interval = pappus::interval<Scalar>;
    static auto const registered = [] {
        auto& unary  = IntervalUnaryRules();
        auto& binary = IntervalBinaryRules();

        unary.Register(Operon::Hash(BuiltinOp::Square),  [](Interval const& v) { return pappus::ops::square<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrt),    [](Interval const& v) { return pappus::ops::sqrt<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Exp),     [](Interval const& v) { return pappus::ops::exp<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Log),     [](Interval const& v) { return pappus::ops::log<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sin),     [](Interval const& v) { return pappus::ops::sin<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cos),     [](Interval const& v) { return pappus::ops::cos<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Tan),     [](Interval const& v) { return pappus::ops::tan<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Asin),    [](Interval const& v) { return pappus::ops::asin<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Acos),    [](Interval const& v) { return pappus::ops::acos<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Atan),    [](Interval const& v) { return pappus::ops::atan<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sinh),    [](Interval const& v) { return pappus::ops::sinh<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cosh),    [](Interval const& v) { return pappus::ops::cosh<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Tanh),    [](Interval const& v) { return pappus::ops::tanh<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Abs),     [](Interval const& v) { return pappus::ops::abs<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrtabs), [](Interval const& v) { return pappus::ops::sqrtabs<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Logabs),  [](Interval const& v) { return pappus::ops::logabs<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Cbrt),    [](Interval const& v) { return pappus::ops::cbrt<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Log1p),   [](Interval const& v) { return pappus::ops::log1p<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Floor),   [](Interval const& v) { return pappus::ops::floor<Scalar>(v); });
        unary.Register(Operon::Hash(BuiltinOp::Ceil),    [](Interval const& v) { return pappus::ops::ceil<Scalar>(v); });

        binary.Register(Operon::Hash(BuiltinOp::Pow), [](Interval const& a, Interval const& b) {
            return pappus::ops::pow<Scalar>(a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Aq), [](Interval const& a, Interval const& b) {
            return pappus::ops::aq<Scalar>(a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Powabs), [](Interval const& a, Interval const& b) {
            return pappus::ops::pow<Scalar>(pappus::ops::abs<Scalar>(a), b);
        });

        return true;
    }();
    static_cast<void>(registered);
}

void RegisterUnaryInterval(Operon::Hash hash, IntervalUnaryFn fn)
{
    RegisterIntervalBuiltins();
    IntervalUnaryRules().Register(hash, std::move(fn));
}

void RegisterBinaryInterval(Operon::Hash hash, IntervalBinaryFn fn)
{
    RegisterIntervalBuiltins();
    IntervalBinaryRules().Register(hash, std::move(fn));
}

auto HasUnaryInterval(Operon::Hash hash) -> bool
{
    RegisterIntervalBuiltins();
    return IntervalUnaryRules().Contains(hash);
}

auto HasBinaryInterval(Operon::Hash hash) -> bool
{
    RegisterIntervalBuiltins();
    return IntervalBinaryRules().Contains(hash);
}

} // namespace Operon
