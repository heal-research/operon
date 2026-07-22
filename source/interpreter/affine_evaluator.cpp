// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/interpreter/affine_evaluator.hpp"

namespace Operon {

auto AffineUnaryRules() -> AffineUnaryRegistry&
{
    static AffineUnaryRegistry registry;
    return registry;
}

auto AffineBinaryRules() -> AffineBinaryRegistry&
{
    static AffineBinaryRegistry registry;
    return registry;
}

void RegisterAffineBuiltins()
{
    using Scalar = Operon::Scalar;
    using Affine = pappus::affine_form<Scalar>;
    using Context = pappus::ops::affine_context<Scalar>;
    static auto const registered = [] {
        auto& unary  = AffineUnaryRules();
        auto& binary = AffineBinaryRules();

        unary.Register(Operon::Hash(BuiltinOp::Square), [](Context const& ctx, Affine const& v) { return pappus::ops::square<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrt), [](Context const& ctx, Affine const& v) { return pappus::ops::sqrt<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Exp), [](Context const& ctx, Affine const& v) { return pappus::ops::exp<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Log), [](Context const& ctx, Affine const& v) { return pappus::ops::log<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Sin), [](Context const& ctx, Affine const& v) { return pappus::ops::sin<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Cos), [](Context const& ctx, Affine const& v) { return pappus::ops::cos<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Tan), [](Context const& ctx, Affine const& v) { return pappus::ops::tan<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Asin), [](Context const& ctx, Affine const& v) { return pappus::ops::asin<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Acos), [](Context const& ctx, Affine const& v) { return pappus::ops::acos<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Atan), [](Context const& ctx, Affine const& v) { return pappus::ops::atan<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Sinh), [](Context const& ctx, Affine const& v) { return pappus::ops::sinh<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Cosh), [](Context const& ctx, Affine const& v) { return pappus::ops::cosh<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Tanh), [](Context const& ctx, Affine const& v) { return pappus::ops::tanh<Scalar>(ctx, v); });
        // May throw if the domain crosses zero (requires Chebyshev V-shape).
        unary.Register(Operon::Hash(BuiltinOp::Abs), [](Context const& ctx, Affine const& v) { return pappus::ops::abs<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrtabs), [](Context const& ctx, Affine const& v) { return pappus::ops::sqrtabs<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Logabs), [](Context const& ctx, Affine const& v) { return pappus::ops::logabs<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Cbrt), [](Context const& ctx, Affine const& v) { return pappus::ops::cbrt<Scalar>(ctx, v); });
        // May throw if the domain includes values <= -1.
        unary.Register(Operon::Hash(BuiltinOp::Log1p), [](Context const& ctx, Affine const& v) { return pappus::ops::log1p<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Floor), [](Context const& ctx, Affine const& v) { return pappus::ops::floor<Scalar>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Ceil), [](Context const& ctx, Affine const& v) { return pappus::ops::ceil<Scalar>(ctx, v); });

        binary.Register(Operon::Hash(BuiltinOp::Pow), [](Context const& ctx, Affine const& a, Affine const& b) {
            return pappus::ops::pow<Scalar>(ctx, a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Aq), [](Context const& ctx, Affine const& a, Affine const& b) {
            return pappus::ops::aq<Scalar>(ctx, a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Powabs), [](Context const& ctx, Affine const& a, Affine const& b) {
            auto absBase = pappus::ops::abs<Scalar>(ctx, a);
            return pappus::ops::pow<Scalar>(ctx, absBase, b);
        });

        return true;
    }();
    static_cast<void>(registered);
}

void RegisterUnaryAffine(Operon::Hash hash, AffineUnaryFn fn)
{
    RegisterAffineBuiltins();
    AffineUnaryRules().Register(hash, std::move(fn));
}

void RegisterBinaryAffine(Operon::Hash hash, AffineBinaryFn fn)
{
    RegisterAffineBuiltins();
    AffineBinaryRules().Register(hash, std::move(fn));
}

auto HasUnaryAffine(Operon::Hash hash) -> bool
{
    RegisterAffineBuiltins();
    return AffineUnaryRules().Contains(hash);
}

auto HasBinaryAffine(Operon::Hash hash) -> bool
{
    RegisterAffineBuiltins();
    return AffineBinaryRules().Contains(hash);
}

} // namespace Operon
