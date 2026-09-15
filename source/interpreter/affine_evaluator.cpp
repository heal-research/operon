// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/interpreter/affine_evaluator.hpp"

namespace Operon {

template<typename T>
auto AffineUnaryRules() -> AffineUnaryRegistry<T>&
{
    static AffineUnaryRegistry<T> registry;
    return registry;
}

template<typename T>
auto AffineBinaryRules() -> AffineBinaryRegistry<T>&
{
    static AffineBinaryRegistry<T> registry;
    return registry;
}

template<typename T>
void RegisterAffineBuiltins()
{
    using Affine = pappus::affine_form<T>;
    using Context = pappus::ops::affine_context<T>;
    static auto const registered = [] {
        auto& unary  = AffineUnaryRules<T>();
        auto& binary = AffineBinaryRules<T>();

        unary.Register(Operon::Hash(BuiltinOp::Square), [](Context const& ctx, Affine const& v) { return pappus::ops::square<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrt), [](Context const& ctx, Affine const& v) { return pappus::ops::sqrt<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Exp), [](Context const& ctx, Affine const& v) { return pappus::ops::exp<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Log), [](Context const& ctx, Affine const& v) { return pappus::ops::log<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Sin), [](Context const& ctx, Affine const& v) { return pappus::ops::sin<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Cos), [](Context const& ctx, Affine const& v) { return pappus::ops::cos<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Tan), [](Context const& ctx, Affine const& v) { return pappus::ops::tan<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Asin), [](Context const& ctx, Affine const& v) { return pappus::ops::asin<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Acos), [](Context const& ctx, Affine const& v) { return pappus::ops::acos<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Atan), [](Context const& ctx, Affine const& v) { return pappus::ops::atan<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Sinh), [](Context const& ctx, Affine const& v) { return pappus::ops::sinh<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Cosh), [](Context const& ctx, Affine const& v) { return pappus::ops::cosh<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Tanh), [](Context const& ctx, Affine const& v) { return pappus::ops::tanh<T>(ctx, v); });
        // Uses a Chebyshev/secant V-shape enclosure when the domain crosses
        // zero -- returns an ordinary (non-invalid) affine form, no domain error.
        unary.Register(Operon::Hash(BuiltinOp::Abs), [](Context const& ctx, Affine const& v) { return pappus::ops::abs<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Sqrtabs), [](Context const& ctx, Affine const& v) { return pappus::ops::sqrtabs<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Logabs), [](Context const& ctx, Affine const& v) { return pappus::ops::logabs<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Cbrt), [](Context const& ctx, Affine const& v) { return pappus::ops::cbrt<T>(ctx, v); });
        // Returns an invalid() (NaN-poisoned) form, not a throw, if the domain
        // includes values <= -1.
        unary.Register(Operon::Hash(BuiltinOp::Log1p), [](Context const& ctx, Affine const& v) { return pappus::ops::log1p<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Floor), [](Context const& ctx, Affine const& v) { return pappus::ops::floor<T>(ctx, v); });
        unary.Register(Operon::Hash(BuiltinOp::Ceil), [](Context const& ctx, Affine const& v) { return pappus::ops::ceil<T>(ctx, v); });

        binary.Register(Operon::Hash(BuiltinOp::Pow), [](Context const& ctx, Affine const& a, Affine const& b) {
            if (b.radius() != T{0}) {
                return Affine(ctx.state, pappus::ops::pow<T>(a.to_interval(), b.to_interval()));
            }
            // Fully constant subtree (degenerate base AND exponent): dispatch
            // through pow(ctx, base, T exponent), whose terms_.empty()
            // branch detects an integer exponent and allows a negative base
            // via plain std::pow -- unlike the general affine-affine overload
            // (pow(ctx, a, b) with b still an Affine), which unconditionally
            // rejects any negative base regardless of the exponent's value.
            // Mirrors IntervalEvaluator's Pow rule's identical special case.
            //
            // Deliberately NOT widened to a non-degenerate base (radius != 0)
            // with a negative-valued domain: that routes into pow(T
            // exponent)'s CHEBYSHEV/MINRANGE approximation for a negative
            // base, which a shape-bound-correctness B2 regression (deeper
            // bisection producing a WIDER union than shallower -- Jackson_2_11,
            // ((-0.834647) * y) ^ 2 with y's domain always negative through
            // that subtree, never crossing zero) showed is not reliably
            // monotone under this evaluator's use across box refinement.
            // Keep the affine-affine rejection + interval fallback for that
            // case until that's root-caused.
            if (a.radius() == T{0}) {
                return pappus::ops::pow<T>(ctx, a, b.center());
            }
            return pappus::ops::pow<T>(ctx, a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Aq), [](Context const& ctx, Affine const& a, Affine const& b) {
            return pappus::ops::aq<T>(ctx, a, b);
        });
        binary.Register(Operon::Hash(BuiltinOp::Powabs), [](Context const& ctx, Affine const& a, Affine const& b) {
            auto absBase = pappus::ops::abs<T>(ctx, a);
            if (b.radius() != T{0}) {
                return Affine(ctx.state, pappus::ops::pow<T>(absBase.to_interval(), b.to_interval()));
            }
            return pappus::ops::pow<T>(ctx, absBase, b);
        });

        return true;
    }();
    static_cast<void>(registered);
}

template<typename T>
void RegisterUnaryAffine(Operon::Hash hash, AffineUnaryFn<T> fn)
{
    RegisterAffineBuiltins<T>();
    AffineUnaryRules<T>().Register(hash, std::move(fn));
}

template<typename T>
void RegisterBinaryAffine(Operon::Hash hash, AffineBinaryFn<T> fn)
{
    RegisterAffineBuiltins<T>();
    AffineBinaryRules<T>().Register(hash, std::move(fn));
}

template<typename T>
auto HasUnaryAffine(Operon::Hash hash) -> bool
{
    RegisterAffineBuiltins<T>();
    return AffineUnaryRules<T>().Contains(hash);
}

template<typename T>
auto HasBinaryAffine(Operon::Hash hash) -> bool
{
    RegisterAffineBuiltins<T>();
    return AffineBinaryRules<T>().Contains(hash);
}

template auto OPERON_EXPORT AffineUnaryRules<Operon::Scalar>() -> AffineUnaryRegistry<Operon::Scalar>&;
template auto OPERON_EXPORT AffineBinaryRules<Operon::Scalar>() -> AffineBinaryRegistry<Operon::Scalar>&;
template void OPERON_EXPORT RegisterAffineBuiltins<Operon::Scalar>();
template void OPERON_EXPORT RegisterUnaryAffine<Operon::Scalar>(Operon::Hash, AffineUnaryFn<Operon::Scalar>);
template void OPERON_EXPORT RegisterBinaryAffine<Operon::Scalar>(Operon::Hash, AffineBinaryFn<Operon::Scalar>);
template auto OPERON_EXPORT HasUnaryAffine<Operon::Scalar>(Operon::Hash) -> bool;
template auto OPERON_EXPORT HasBinaryAffine<Operon::Scalar>(Operon::Hash) -> bool;

} // namespace Operon
