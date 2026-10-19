// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERVAL_SECOND_ORDER_JET_EVALUATOR_HPP
#define OPERON_INTERVAL_SECOND_ORDER_JET_EVALUATOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <gsl/pointers>
#include <tl/expected.hpp>

#include <pappus/pappus.hpp>

#include "operon/core/contracts.hpp"
#include "operon/core/jet.hpp"
#include "operon/core/node.hpp"
#include "operon/core/postorder_evaluator.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/backend/interval_jet_math.hpp"
#include "operon/interpreter/interval_jet_evaluator.hpp"

namespace Operon {

// Result of IntervalSecondOrderJetEvaluator: the tree's value over the box plus, per differentiated variable, an
// enclosure of the first derivative d f / d x_k and of the pure second derivative d^2 f / d x_k^2. First[k],
// Second[k] and Status[k] are only meaningful for k < Directions; the derivative enclosures are only enclosures of
// the true derivatives when Status[k] == Certain. Status[k] (IntervalJetStatus, shared with IntervalJetEvaluator)
// covers the direction as a whole: Uncertain when an operation without a differentiation rule lies on a path that
// depends on the variable, Failed when the value, the first or the second derivative came out empty/NaN. Entries at
// or beyond Directions read Failed -- an unevaluated direction must never be mistakable for a certified [0, 0].
template <typename T, std::size_t N = 4> struct IntervalSecondOrderJetBounds {
    static constexpr std::size_t MaxDirections = N;
    using Interval = pappus::interval<T>;

    Interval Value {};
    std::size_t Directions {};
    std::array<Interval, MaxDirections> First {};
    std::array<Interval, MaxDirections> Second {};
    std::array<IntervalJetStatus, MaxDirections> Status {};
};

namespace detail {
    // The direction counts of the second-order interval jet pass (1..4 differentiated variables).
    inline constexpr std::size_t IntervalSecondOrderJetMaxDirections = 4;

    // An order-two Jet over intervals plus the same per-direction bookkeeping as IntervalJetValue: bit k of Dep is
    // set iff the node's value depends on variable k, bit k of Unsup iff, in addition, an operation without a
    // differentiation rule lies on a path from variable k to this node. Tangent k (first and second derivative) is
    // exactly [0, 0] wherever Dep bit k is clear, so a domain error in the derivatives of one variable cannot leak
    // into another.
    template <typename T, int K> struct IntervalSecondOrderJetValue {
        using Interval = pappus::interval<T>;
        Operon::Jet<Interval, K, JetOrder::SecondDiagonal> J {};
        IntervalJetMask<K> Dep {};
        IntervalJetMask<K> Unsup {};
    };

    template <typename T, typename Seq> struct IntervalSecondOrderJetScratchOf;
    template <typename T, std::size_t... I> struct IntervalSecondOrderJetScratchOf<T, std::index_sequence<I...>> {
        using type = std::tuple<std::vector<IntervalSecondOrderJetValue<T, static_cast<int>(I + 1)>>...>;
    };
    // One reusable post-order scratch vector per supported direction count.
    template <typename T>
    using IntervalSecondOrderJetScratch = typename IntervalSecondOrderJetScratchOf<T,
        std::make_index_sequence<IntervalSecondOrderJetMaxDirections>>::type;

    // Policy for detail::EvaluatePostOrder. The differentiable built-ins are the ones the symbolic derivative rules
    // of tree_diff.cpp cover (exp log log1p logabs sin cos tan asin acos atan sinh cosh tanh sqrt cbrt square,
    // n-ary Add/Mul, Sub, Div, Pow); every other built-in keeps its interval primal (identical to the
    // IntervalEvaluator rules) and marks the directions it depends on uncertain. Operations that are not built-ins
    // are not supported: they surface as MissingNode.
    template <typename T, int K> struct IntervalSecondOrderJetPolicy {
        static_assert(K >= 1 && static_cast<std::size_t>(K) <= IntervalSecondOrderJetMaxDirections);
        using Mask = IntervalJetMask<K>;

        using Scalar = T;
        using Interval = pappus::interval<T>;
        using J = Operon::Jet<Interval, K, JetOrder::SecondDiagonal>;
        using Value = IntervalSecondOrderJetValue<T, K>;
        using UnaryFn = auto (*)(Value const&) -> Value;
        using BinaryFn = auto (*)(Value const&, Value const&) -> Value;
        struct Context {};

        static constexpr Mask Full = static_cast<Mask>((std::uint64_t { 1 } << K) - 1U);

        static void RegisterBuiltins() {}
        static auto UnaryRules() -> IntervalJetRuleTable<UnaryFn> const&
        {
            static constexpr auto table = BuildUnary();
            return table;
        }
        static auto BinaryRules() -> IntervalJetRuleTable<BinaryFn> const&
        {
            static constexpr auto table = BuildBinary();
            return table;
        }
        static auto EmptyTree() -> std::string { return "IntervalSecondOrderJetEvaluator: empty tree"; }
        static auto MissingNode(Node const& node) -> std::string
        {
            return fmt::format(
                "IntervalSecondOrderJetEvaluator: node kind `{}` is not supported by the jet path", node.Name());
        }

        static auto MakeConstant(Context const&, Scalar value) -> Value { return Constant(Interval { value }); }

        static auto Add(Context const&, Value const& x, Value const& y) -> Value
        {
            return { x.J + y.J, static_cast<Mask>(x.Dep | y.Dep), static_cast<Mask>(x.Unsup | y.Unsup) };
        }
        static auto Sub(Context const&, Value const& x, Value const& y) -> Value
        {
            return { x.J - y.J, static_cast<Mask>(x.Dep | y.Dep), static_cast<Mask>(x.Unsup | y.Unsup) };
        }
        static auto Neg(Context const&, Value const& x) -> Value { return { -x.J, x.Dep, x.Unsup }; }

        // (x y)' = x' y + x y', (x y)'' = x'' y + 2 x' y' + x y''; each term only for the directions its operand
        // depends on.
        static auto Mul(Context const&, Value const& x, Value const& y) -> Value
        {
            Value r;
            r.Dep = static_cast<Mask>(x.Dep | y.Dep);
            r.Unsup = static_cast<Mask>(x.Unsup | y.Unsup);
            r.J.a = x.J.a * y.J.a;
            ForEachDirection(r.Dep, [&](int k) {
                auto const i = static_cast<std::size_t>(k);
                auto const xd = Depends(x.Dep, k);
                auto const yd = Depends(y.Dep, k);
                auto v = Interval::zero();
                auto w = Interval::zero();
                if (xd) {
                    v = x.J.v[i] * y.J.a;
                    w = x.J.w[i] * y.J.a;
                }
                if (yd) {
                    v = v + x.J.a * y.J.v[i];
                    w = w + x.J.a * y.J.w[i];
                }
                if (xd && yd) {
                    auto const cross = x.J.v[i] * y.J.v[i];
                    w = w + (cross + cross);
                }
                r.J.v[i] = v;
                r.J.w[i] = w;
            });
            return r;
        }

        // The primal is x / y (same as IntervalEvaluator); with q = x / y: q' = (x' - q y') / y and
        // q'' = (x'' - 2 q' y' - q y'') / y, or x' / y and x'' / y when the denominator is constant in the direction.
        static auto Div(Context const&, Value const& x, Value const& y) -> Value
        {
            Value r;
            r.Dep = static_cast<Mask>(x.Dep | y.Dep);
            r.Unsup = static_cast<Mask>(x.Unsup | y.Unsup);
            r.J.a = x.J.a / y.J.a;
            if (r.Dep == 0) {
                return r;
            }
            auto const inv = y.Dep != 0 ? y.J.a.inv() : Interval::zero();
            ForEachDirection(r.Dep, [&](int k) {
                auto const i = static_cast<std::size_t>(k);
                if (!Depends(y.Dep, k)) {
                    r.J.v[i] = x.J.v[i] / y.J.a;
                    r.J.w[i] = x.J.w[i] / y.J.a;
                    return;
                }
                auto const v = (x.J.v[i] - r.J.a * y.J.v[i]) * inv;
                auto const cross = v * y.J.v[i];
                r.J.v[i] = v;
                r.J.w[i] = (x.J.w[i] - (cross + cross) - r.J.a * y.J.w[i]) * inv;
            });
            return r;
        }

        // 1 / x: f' = -(1/x)^2, f'' = 2 (1/x)^3
        static auto Inv(Context const&, Value const& x) -> Value
        {
            auto const inv = x.J.a.inv();
            return Unary(x, inv, -inv.square(), (inv + inv) * inv.square());
        }

        static auto Min(Context const&, Value const& x, Value const& y) -> Value
        {
            return Opaque(x, y, pappus::ops::min<T>(x.J.a, y.J.a));
        }
        static auto Max(Context const&, Value const& x, Value const& y) -> Value
        {
            return Opaque(x, y, pappus::ops::max<T>(x.J.a, y.J.a));
        }

        static auto CallUnary(Context const&, UnaryFn function, Value const& x) -> tl::expected<Value, std::string>
        {
            return function(x);
        }
        static auto CallBinary(Context const&, BinaryFn function, Value const& x, Value const& y)
            -> tl::expected<Value, std::string>
        {
            return function(x, y);
        }

        // Every Function node multiplies its op result by its own weight; a zero weight zeroes the derivatives
        // exactly like tree_diff's applyWeight (interval * 0 == [0, 0]).
        static auto Scale(Value x, Scalar scale) -> Value
        {
            if (scale == Scalar { 1 }) {
                return x;
            }
            x.J.a = x.J.a * scale;
            ForEachDirection(x.Dep, [&](int k) {
                auto const i = static_cast<std::size_t>(k);
                x.J.v[i] = x.J.v[i] * scale;
                x.J.w[i] = x.J.w[i] * scale;
            });
            return x;
        }

        static auto Constant(Interval value) -> Value
        {
            Value r;
            r.J.a = value;
            return r;
        }

        // Variable leaf w * X: primal [lo, hi] * w, first derivative w and second derivative 0 along `direction`
        // (-1: not a differentiated variable).
        static auto Variable(Interval domain, Scalar scale, int direction) -> Value
        {
            Value r;
            r.J.a = domain * scale;
            if (direction >= 0) {
                r.J.v[static_cast<std::size_t>(direction)] = Interval { scale };
                r.Dep = static_cast<Mask>(Mask { 1 } << direction);
            }
            return r;
        }

    private:
        static auto Depends(Mask mask, int k) -> bool { return ((mask >> k) & 1U) != 0; }

        template <typename F> static void ForEachDirection(Mask mask, F&& f)
        {
            for (int k = 0; k < K; ++k) {
                if (Depends(mask, k)) {
                    f(k);
                }
            }
        }

        // Zero the derivatives of the directions the node does not depend on.
        static void Restrict(Value& r)
        {
            if (r.Dep != Full) {
                for (int k = 0; k < K; ++k) {
                    if (!Depends(r.Dep, k)) {
                        r.J.v[static_cast<std::size_t>(k)] = Interval::zero();
                        r.J.w[static_cast<std::size_t>(k)] = Interval::zero();
                    }
                }
            }
        }

        // Smooth unary op: `f` is the order-two Jet function (primal and chain rule).
        template <typename F> static auto Smooth(Value const& x, F f) -> Value
        {
            Value r { f(x.J), x.Dep, x.Unsup };
            Restrict(r);
            return r;
        }

        // Unary op with the given primal, first derivative d1 and second derivative d2 at x.
        static auto Unary(Value const& x, Interval primal, Interval d1, Interval d2) -> Value
        {
            Value r;
            r.Dep = x.Dep;
            r.Unsup = x.Unsup;
            r.J.a = primal;
            ForEachDirection(r.Dep, [&](int k) {
                auto const i = static_cast<std::size_t>(k);
                r.J.v[i] = d1 * x.J.v[i];
                r.J.w[i] = d2 * x.J.v[i].square() + d1 * x.J.w[i];
            });
            return r;
        }

        // Operation without a differentiation rule: primal only; dependent directions become (-inf, inf) and
        // are flagged unsupported.
        static auto Opaque(Value const& x, Interval primal) -> Value
        {
            Value r;
            r.J.a = primal;
            r.Dep = x.Dep;
            r.Unsup = static_cast<Mask>(x.Unsup | x.Dep);
            ForEachDirection(r.Dep, [&](int k) {
                r.J.v[static_cast<std::size_t>(k)] = Interval::infinite();
                r.J.w[static_cast<std::size_t>(k)] = Interval::infinite();
            });
            return r;
        }
        static auto Opaque(Value const& x, Value const& y, Interval primal) -> Value
        {
            Value r;
            r.J.a = primal;
            r.Dep = static_cast<Mask>(x.Dep | y.Dep);
            r.Unsup = static_cast<Mask>(x.Unsup | y.Unsup | r.Dep);
            ForEachDirection(r.Dep, [&](int k) {
                r.J.v[static_cast<std::size_t>(k)] = Interval::infinite();
                r.J.w[static_cast<std::size_t>(k)] = Interval::infinite();
            });
            return r;
        }

        // log|x|: f' = 1 / x (as in the symbolic rule), f'' = -1 / x^2; the primal is the IntervalEvaluator's
        // log(abs(x)).
        static auto Logabs(Value const& x) -> Value
        {
            auto const d1 = Interval { T { 1 } } / x.J.a;
            return Unary(x, pappus::ops::logabs<T>(x.J.a), d1, -d1.square());
        }

        // x^y with r = x^y: r_x = y x^(y-1), r_xx = y (y-1) x^(y-2), r_y = r log x, r_yy = r log^2 x,
        // r_xy = x^(y-1) (1 + y log x). The integer-power interval operation is used for an integral point exponent
        // (see JetMath); the exponents 0, 1 and 2 are exact, where y x^(y-1) or y (y-1) x^(y-2) would otherwise
        // evaluate 0^0 over a zero base. Each direction only receives the partials of the operands that depend on it.
        static auto Pow(Value const& x, Value const& y) -> Value
        {
            using M = Operon::JetMath<Interval>;
            Value r;
            r.Dep = static_cast<Mask>(x.Dep | y.Dep);
            r.Unsup = static_cast<Mask>(x.Unsup | y.Unsup);
            r.J.a = M::pow(x.J.a, y.J.a);
            if (r.Dep == 0) {
                return r;
            }
            auto const one = Interval { T { 1 } };
            auto const exponentIs = [&](T c) { return y.J.a.inf() == c && y.J.a.sup() == c; };
            Interval dx {};
            Interval dxx {};
            Interval dy {};
            Interval dyy {};
            Interval dxy {};
            if (x.Dep != 0) {
                if (exponentIs(T { 1 })) {
                    dx = one;
                } else if (!y.J.a.is_zero()) {
                    dx = y.J.a * M::pow(x.J.a, y.J.a - one);
                    dxx = exponentIs(T { 2 }) ? Interval { T { 2 } }
                                              : y.J.a * (y.J.a - one) * M::pow(x.J.a, y.J.a - Interval { T { 2 } });
                }
            }
            if (y.Dep != 0) {
                dy = M::xlogy(r.J.a, x.J.a);
                dyy = M::xlogy(dy, x.J.a);
            }
            if ((x.Dep & y.Dep) != 0) {
                auto const p = M::pow(x.J.a, y.J.a - one);
                dxy = p + y.J.a * M::xlogy(p, x.J.a);
            }
            ForEachDirection(r.Dep, [&](int k) {
                auto const i = static_cast<std::size_t>(k);
                auto const xd = Depends(x.Dep, k);
                auto const yd = Depends(y.Dep, k);
                auto v = Interval::zero();
                auto w = Interval::zero();
                if (xd) {
                    v = dx * x.J.v[i];
                    w = dx * x.J.w[i] + dxx * x.J.v[i].square();
                }
                if (yd) {
                    v = v + dy * y.J.v[i];
                    w = w + dy * y.J.w[i] + dyy * y.J.v[i].square();
                }
                if (xd && yd) {
                    auto const cross = dxy * x.J.v[i] * y.J.v[i];
                    w = w + (cross + cross);
                }
                r.J.v[i] = v;
                r.J.w[i] = w;
            });
            return r;
        }

        static consteval auto BuildUnary() -> IntervalJetRuleTable<UnaryFn>
        {
            IntervalJetRuleTable<UnaryFn> t;
            auto const set = [&](BuiltinOp op, UnaryFn fn) { t.Rules[Operon::Hash(op)] = fn; };
            set(BuiltinOp::Square,
                [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::square(j); }); });
            set(BuiltinOp::Sqrt, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::sqrt(j); }); });
            set(BuiltinOp::Cbrt, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::cbrt(j); }); });
            set(BuiltinOp::Exp, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::exp(j); }); });
            set(BuiltinOp::Log, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::log(j); }); });
            set(BuiltinOp::Log1p,
                [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::log1p(j); }); });
            set(BuiltinOp::Logabs, [](Value const& x) { return Logabs(x); });
            set(BuiltinOp::Sin, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::sin(j); }); });
            set(BuiltinOp::Cos, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::cos(j); }); });
            set(BuiltinOp::Tan, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::tan(j); }); });
            set(BuiltinOp::Asin, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::asin(j); }); });
            set(BuiltinOp::Acos, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::acos(j); }); });
            set(BuiltinOp::Atan, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::atan(j); }); });
            set(BuiltinOp::Sinh, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::sinh(j); }); });
            set(BuiltinOp::Cosh, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::cosh(j); }); });
            set(BuiltinOp::Tanh, [](Value const& x) { return Smooth(x, [](J const& j) { return Operon::tanh(j); }); });
            // no differentiation rule (non-smooth): primal only, dependent directions uncertain
            set(BuiltinOp::Abs, [](Value const& x) { return Opaque(x, pappus::ops::abs<T>(x.J.a)); });
            set(BuiltinOp::Sqrtabs, [](Value const& x) { return Opaque(x, pappus::ops::sqrtabs<T>(x.J.a)); });
            set(BuiltinOp::Floor, [](Value const& x) { return Opaque(x, pappus::ops::floor<T>(x.J.a)); });
            set(BuiltinOp::Ceil, [](Value const& x) { return Opaque(x, pappus::ops::ceil<T>(x.J.a)); });
            return t;
        }

        static consteval auto BuildBinary() -> IntervalJetRuleTable<BinaryFn>
        {
            IntervalJetRuleTable<BinaryFn> t;
            auto const set = [&](BuiltinOp op, BinaryFn fn) { t.Rules[Operon::Hash(op)] = fn; };
            set(BuiltinOp::Pow, [](Value const& x, Value const& y) { return Pow(x, y); });
            set(BuiltinOp::Aq,
                [](Value const& x, Value const& y) { return Opaque(x, y, pappus::ops::aq<T>(x.J.a, y.J.a)); });
            set(BuiltinOp::Powabs, [](Value const& x, Value const& y) {
                return Opaque(x, y, pappus::ops::pow<T>(pappus::ops::abs<T>(x.J.a), y.J.a));
            });
            return t;
        }
    };
} // namespace detail

// Forward-mode second-order interval bounds: one post-order pass over a tree computes, per node, a
// Jet<pappus::interval<T>, K, 2> -- an enclosure of the value, of its first derivatives and of its pure
// second derivatives d^2 f / d x_k^2 with respect to K (1..4) designated variables over a single input box. It is
// the replacement for building the symbolic derivative DAG of the first derivative and differentiating that again
// (the twice-differentiated tree the shape certifier used to materialize): both are sound enclosures, neither
// dominates the other, and the jet needs neither the symbolic nodes nor a second evaluator pass.
//
// Retained semantics of IntervalJetEvaluator, per direction: operations the symbolic dag cannot differentiate (abs
// sqrtabs floor ceil aq powabs fmin fmax) keep their interval primal and mark the dependent directions
// IntervalJetStatus::Uncertain -- a second-order enclosure is never reported Certain across a non-smooth operation.
// User-registered operations are not supported at all: the evaluator returns an error and the caller must fall back to
// the symbolic path. The same holds for n-ary Div (3+ operands), which the dag does not differentiate either.
// Variables of the tree that are not among the differentiated `variables` are constants over their domain box.
template <typename T = Operon::Scalar> class IntervalSecondOrderJetEvaluator {
public:
    using Scalar = T;
    using Interval = pappus::interval<T>;
    using Domain = std::pair<Operon::Scalar, Operon::Scalar>;
    using DomainMap = Operon::Map<Operon::Hash, Domain>;
    using Bounds = IntervalSecondOrderJetBounds<T>;

    static constexpr std::size_t MaxDirections = Bounds::MaxDirections;

    // `variables`: distinct, nonzero variable hashes, 1..MaxDirections of them; direction k differentiates
    // variables[k]. (A zero hash is the "no variable" marker of constraints without one, never a dataset
    // variable.)
    IntervalSecondOrderJetEvaluator(
        gsl::not_null<Operon::Tree const*> tree, DomainMap const& domains, Operon::Span<Operon::Hash const> variables)
        : tree_(tree)
        , domains_(domains)
        , variables_(variables.begin(), variables.end())
    {
        EXPECT(std::ranges::none_of(variables_, [](Operon::Hash h) { return h == Operon::Hash {}; }));
        EXPECT(AllDistinct(variables_));
        RebuildSlots();
    }

    void SetTree(gsl::not_null<Operon::Tree const*> tree)
    {
        tree_ = tree;
        RebuildSlots();
    }

    // Evaluates the tree. Never throws: an unsupported node, a missing domain, an unsupported direction count
    // or an empty tree comes back as the error string; domain errors of the function itself are reported
    // through Bounds::Status instead.
    [[nodiscard]] auto operator()(Operon::Span<Operon::Scalar const> coeff) const -> tl::expected<Bounds, std::string>
    {
        switch (variables_.size()) {
        case 1:
            return Run<1>(coeff);
        case 2:
            return Run<2>(coeff);
        case 3:
            return Run<3>(coeff);
        case 4:
            return Run<4>(coeff);
        default:
            return tl::unexpected(
                fmt::format("IntervalSecondOrderJetEvaluator: {} differentiated variables (supported: 1..{})",
                    variables_.size(), MaxDirections));
        }
    }

private:
    [[nodiscard]] static auto AllDistinct(std::vector<Operon::Hash> const& hashes) -> bool
    {
        for (std::size_t i = 0; i < hashes.size(); ++i) {
            for (std::size_t j = i + 1; j < hashes.size(); ++j) {
                if (hashes[i] == hashes[j]) {
                    return false;
                }
            }
        }
        return true;
    }

    struct Slot {
        Domain Bounds {};
        int Direction { -1 };
        bool Present { false };
    };

    template <int K>
    [[nodiscard]] auto Run(Operon::Span<Operon::Scalar const> coeff) const -> tl::expected<Bounds, std::string>
    {
        using Policy = detail::IntervalSecondOrderJetPolicy<T, K>;
        auto const& nodes = tree_->Nodes();

        std::size_t coefficientIndex = 0;
        bool naryDivision = false;
        auto const weight = [&](Node const& node) {
            naryDivision = naryDivision || (node.IsDivision() && node.Arity > 2);
            if (node.Optimize) {
                EXPECT(coefficientIndex < coeff.size());
                return static_cast<Scalar>(coeff[coefficientIndex++]);
            }
            return static_cast<Scalar>(node.Value);
        };
        auto const bindLeaf = [&](Node const& node, std::size_t index,
                                  Scalar scale) -> tl::expected<typename Policy::Value, std::string> {
            if (node.IsConstant()) {
                return Policy::Constant(Interval { scale });
            }
            auto const& slot = slots_[index];
            if (!slot.Present) {
                return tl::unexpected(fmt::format(
                    "IntervalSecondOrderJetEvaluator: no domain bound for variable hash {}", node.HashValue));
            }
            return Policy::Variable(
                Interval { static_cast<Scalar>(slot.Bounds.first), static_cast<Scalar>(slot.Bounds.second) }, scale,
                slot.Direction);
        };

        auto& primal = std::get<static_cast<std::size_t>(K - 1)>(primal_);
        auto evaluated
            = detail::EvaluatePostOrder<Policy>(nodes, primal, typename Policy::Context {}, weight, bindLeaf);
        if (!evaluated) {
            return tl::unexpected(std::move(evaluated.error()));
        }
        if (naryDivision) {
            return tl::unexpected(
                std::string { "IntervalSecondOrderJetEvaluator: n-ary Div is not supported by the jet path" });
        }

        auto const& root = *evaluated;
        Bounds out;
        out.Value = root.J.a;
        out.Directions = static_cast<std::size_t>(K);
        auto const valueFailed = root.J.a.is_empty();
        // Directions and beyond stay Failed: only k < K is evaluated (see the Bounds contract above).
        out.Status.fill(IntervalJetStatus::Failed);
        for (std::size_t k = 0; k < static_cast<std::size_t>(K); ++k) {
            out.First[k] = root.J.v[k];
            out.Second[k] = root.J.w[k];
            if (((root.Unsup >> k) & 1U) != 0) {
                out.Status[k] = IntervalJetStatus::Uncertain;
            } else if (valueFailed || root.J.v[k].is_empty() || root.J.w[k].is_empty()) {
                out.Status[k] = IntervalJetStatus::Failed;
            } else {
                out.Status[k] = IntervalJetStatus::Certain;
            }
        }
        return out;
    }

    void RebuildSlots()
    {
        auto const& nodes = tree_->Nodes();
        slots_.clear();
        slots_.reserve(nodes.size());
        for (auto const& node : nodes) {
            Slot slot;
            if (node.Type == NodeType::Variable) {
                if (auto const it = domains_.find(node.HashValue); it != domains_.end()) {
                    slot.Bounds = it->second;
                    slot.Present = true;
                }
                if (auto const it = std::ranges::find(variables_, node.HashValue); it != variables_.end()) {
                    slot.Direction = static_cast<int>(it - variables_.begin());
                }
            }
            slots_.push_back(slot);
        }
    }

    gsl::not_null<Operon::Tree const*> tree_;
    DomainMap domains_;
    std::vector<Operon::Hash> variables_;
    std::vector<Slot> slots_;
    mutable detail::IntervalSecondOrderJetScratch<T> primal_; // reused across calls
};

} // namespace Operon

#endif
