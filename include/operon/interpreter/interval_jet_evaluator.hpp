// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERVAL_JET_EVALUATOR_HPP
#define OPERON_INTERVAL_JET_EVALUATOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
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

namespace Operon {

// Outcome of one derivative direction of an IntervalJetEvaluator run.
enum class IntervalJetStatus : std::uint8_t {
    Certain, // Derivative is a sound enclosure of the partial derivative over the whole box
    Uncertain, // an operation without a differentiation rule lies on a path that depends on the variable
    Failed, // domain error: the primal or the derivative came out empty/NaN
};

// Result of IntervalJetEvaluator: the tree's value over the box plus one derivative enclosure per variable.
// Derivative[k]/Status[k] are only meaningful for k < Directions; Derivative[k] is only an enclosure of the true
// partial derivative when Status[k] == Certain. N is the capacity (4 for the regular evaluator call, 32 for the
// wide benchmark entry point IntervalJetEvaluator::EvaluateWide).
template <typename T, std::size_t N = 4> struct IntervalJetBounds {
    static constexpr std::size_t MaxDirections = N;
    using Interval = pappus::interval<T>;

    Interval Value {};
    std::size_t Directions {};
    std::array<Interval, MaxDirections> Derivative {};
    std::array<IntervalJetStatus, MaxDirections> Status {};
};

template <typename T> using IntervalJetWideBounds = IntervalJetBounds<T, 32>;

namespace detail {
    // Dependency bitmask of a jet with K directions: one byte up to 8 directions (the regular 1..4 path), 32 bits
    // beyond.
    template <int K> using IntervalJetMask = std::conditional_t<(K <= 8), std::uint8_t, std::uint32_t>;

    // The direction counts the wide entry point can run (K variables differentiated at once).
    inline constexpr std::array<int, 10> IntervalJetWideDirections { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 };

    // A Jet over intervals plus the bookkeeping that makes the derivative directions independent: bit k of Dep
    // is set iff the node's value depends on variable k, bit k of Unsup iff, in addition, an operation without a
    // differentiation rule lies on a path from variable k to this node. Tangent k is exactly [0, 0] wherever
    // Dep bit k is clear, so a domain error in a partial derivative of one variable cannot leak into another.
    template <typename T, int K> struct IntervalJetValue {
        using Interval = pappus::interval<T>;
        Operon::Jet<Interval, K> J {};
        IntervalJetMask<K> Dep {};
        IntervalJetMask<K> Unsup {};
    };

    constexpr auto IntervalJetWideIndex(int k) -> std::size_t
    {
        std::size_t i = 0;
        while (i < IntervalJetWideDirections.size() && IntervalJetWideDirections[i] != k) {
            ++i;
        }
        return i;
    }

    template <typename T, typename Seq> struct IntervalJetScratchOf;
    template <typename T, std::size_t... I> struct IntervalJetScratchOf<T, std::index_sequence<I...>> {
        using type = std::tuple<std::vector<IntervalJetValue<T, IntervalJetWideDirections[I]>>...>;
    };
    // One reusable post-order scratch vector per supported direction count.
    template <typename T>
    using IntervalJetScratch =
        typename IntervalJetScratchOf<T, std::make_index_sequence<IntervalJetWideDirections.size()>>::type;

    template <typename Fn> struct IntervalJetRuleTable {
        std::array<Fn, BuiltinOpCount> Rules {};

        [[nodiscard]] auto TryGet(Operon::Hash hash) const -> Fn const*
        {
            return hash < Rules.size() && Rules[hash] != nullptr ? &Rules[hash] : nullptr;
        }
    };

    // Policy for detail::EvaluatePostOrder. The differentiable built-ins mirror the symbolic derivative rules of
    // tree_diff.cpp (exp log log1p logabs sin cos tan asin acos atan sinh cosh tanh sqrt cbrt square, n-ary
    // Add/Mul, Sub, Div, Pow); every other built-in keeps its interval primal (identical to the IntervalEvaluator
    // rules) and marks the directions it depends on uncertain. Operations that are not built-ins are not
    // supported: they surface as MissingNode.
    template <typename T, int K> struct IntervalJetPolicy {
        static_assert(K >= 1 && K <= 32);
        using Mask = IntervalJetMask<K>;

        using Scalar = T;
        using Interval = pappus::interval<T>;
        using J = Operon::Jet<Interval, K>;
        using Value = IntervalJetValue<T, K>;
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
        static auto EmptyTree() -> std::string { return "IntervalJetEvaluator: empty tree"; }
        static auto MissingNode(Node const& node) -> std::string
        {
            return fmt::format("IntervalJetEvaluator: node kind `{}` is not supported by the jet path", node.Name());
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
        static auto Mul(Context const&, Value const& x, Value const& y) -> Value
        {
            return { x.J * y.J, static_cast<Mask>(x.Dep | y.Dep), static_cast<Mask>(x.Unsup | y.Unsup) };
        }
        static auto Neg(Context const&, Value const& x) -> Value { return { -x.J, x.Dep, x.Unsup }; }

        // The primal is x / y (same as IntervalEvaluator); the tangent is (x' - q y') / y, or x' / y when the
        // denominator is constant (the DAG's `dj / k` special case).
        static auto Div(Context const&, Value const& x, Value const& y) -> Value
        {
            Value r;
            r.Dep = static_cast<Mask>(x.Dep | y.Dep);
            r.Unsup = static_cast<Mask>(x.Unsup | y.Unsup);
            r.J.a = x.J.a / y.J.a;
            if (r.Dep == 0) {
                return r;
            }
            if (y.Dep == 0) {
                ForEachDirection(r.Dep, [&](int k) { r.J.v[k] = x.J.v[k] / y.J.a; });
                return r;
            }
            auto const inv = y.J.a.inv();
            ForEachDirection(r.Dep, [&](int k) {
                r.J.v[k] = ((y.Dep >> k) & 1U) != 0 ? (x.J.v[k] - r.J.a * y.J.v[k]) * inv : x.J.v[k] / y.J.a;
            });
            return r;
        }

        // 1 / x: tangent -x' / x^2 = -(1/x)^2 x'
        static auto Inv(Context const&, Value const& x) -> Value
        {
            Value r;
            r.Dep = x.Dep;
            r.Unsup = x.Unsup;
            r.J.a = x.J.a.inv();
            if (r.Dep != 0) {
                auto const d = -r.J.a.square();
                ForEachDirection(r.Dep, [&](int k) { r.J.v[k] = d * x.J.v[k]; });
            }
            return r;
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

        // Every Function node multiplies its op result by its own weight; a zero weight zeroes the derivative
        // exactly like tree_diff's applyWeight (interval * 0 == [0, 0]).
        static auto Scale(Value x, Scalar scale) -> Value
        {
            if (scale == Scalar { 1 }) {
                return x;
            }
            x.J.a = x.J.a * scale;
            ForEachDirection(x.Dep, [&](int k) { x.J.v[k] = x.J.v[k] * scale; });
            return x;
        }

        static auto Constant(Interval value) -> Value
        {
            Value r;
            r.J.a = value;
            return r;
        }

        // Variable leaf w * X: primal [lo, hi] * w, tangent w along `direction` (-1: not a differentiated variable).
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
        template <typename F> static void ForEachDirection(Mask mask, F&& f)
        {
            for (int k = 0; k < K; ++k) {
                if (((mask >> k) & 1U) != 0) {
                    f(k);
                }
            }
        }

        // Zero the tangents of the directions the node does not depend on.
        static void Restrict(Value& r)
        {
            if (r.Dep != Full) {
                for (int k = 0; k < K; ++k) {
                    if (((r.Dep >> k) & 1U) == 0) {
                        r.J.v[static_cast<std::size_t>(k)] = Interval::zero();
                    }
                }
            }
        }

        // Smooth unary op: `f` is the Jet function (primal and chain rule); the primal never depends on K.
        template <typename F> static auto Smooth(Value const& x, F f) -> Value
        {
            Value r { f(x.J), x.Dep, x.Unsup };
            Restrict(r);
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
            ForEachDirection(r.Dep, [&](int k) { r.J.v[k] = Interval::infinite(); });
            return r;
        }
        static auto Opaque(Value const& x, Value const& y, Interval primal) -> Value
        {
            Value r;
            r.J.a = primal;
            r.Dep = static_cast<Mask>(x.Dep | y.Dep);
            r.Unsup = static_cast<Mask>(x.Unsup | y.Unsup | r.Dep);
            ForEachDirection(r.Dep, [&](int k) { r.J.v[k] = Interval::infinite(); });
            return r;
        }

        // d/dx log|x| = 1 / x (as in the symbolic rule); the primal is the IntervalEvaluator's log(abs(x)).
        static auto Logabs(Value const& x) -> Value
        {
            Value r;
            r.Dep = x.Dep;
            r.Unsup = x.Unsup;
            r.J.a = pappus::ops::logabs<T>(x.J.a);
            if (r.Dep != 0) {
                auto const d = Interval { T { 1 } } / x.J.a;
                ForEachDirection(r.Dep, [&](int k) { r.J.v[k] = d * x.J.v[k]; });
            }
            return r;
        }

        // x^y: d/dx = y x^(y-1) (integer-power interval operation for an integral point exponent, see JetMath),
        // d/dy = x^y log x. Each direction only receives the partials of the operands that depend on it.
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
            Interval dx {};
            Interval dy {};
            if (x.Dep != 0) {
                // exact for the exponents 0 and 1, where y x^(y-1) would otherwise evaluate 0^0 over a zero base
                if (y.J.a.inf() == T { 1 } && y.J.a.sup() == T { 1 }) {
                    dx = Interval { T { 1 } };
                } else if (!y.J.a.is_zero()) {
                    dx = y.J.a * M::pow(x.J.a, y.J.a - Interval { T { 1 } });
                }
            }
            if (y.Dep != 0) {
                dy = M::xlogy(r.J.a, x.J.a);
            }
            ForEachDirection(r.Dep, [&](int k) {
                auto t = Interval::zero();
                if (((x.Dep >> k) & 1U) != 0) {
                    t = dx * x.J.v[k];
                }
                if (((y.Dep >> k) & 1U) != 0) {
                    t = t + dy * y.J.v[k];
                }
                r.J.v[k] = t;
            });
            return r;
        }

        static constexpr auto BuildUnary() -> IntervalJetRuleTable<UnaryFn>
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

        static constexpr auto BuildBinary() -> IntervalJetRuleTable<BinaryFn>
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

// Forward-mode first-derivative interval bounds: one post-order pass over a tree computes, per node, a
// Jet<pappus::interval<T>, K> -- an enclosure of the value and of its partial derivatives with respect to K
// (1..4) designated variables over a single input box. It is the forward-mode counterpart of building the
// symbolic VariableGradientDag and evaluating it with IntervalEvaluator; the two enclosures are both sound but
// neither dominates (the jet knows pow(x, 2)' = 2x is finite at 0, the symbolic dag knows 1 + x^2 >= 1 through
// the nonnegativity of Square).
//
// Operations the symbolic dag cannot differentiate (abs sqrtabs floor ceil aq powabs fmin fmax) keep their
// interval primal and mark the dependent directions IntervalJetStatus::Uncertain. User-registered operations
// are not supported at all: the evaluator returns an error and the caller must fall back to the symbolic path.
// The same holds for n-ary Div (3+ operands), which the dag does not differentiate either.
//
// Variables of the tree that are not among the differentiated `variables` are constants over their domain box.
// operator() supports 1..4 differentiated variables; EvaluateWide (benchmark/scaling use) supports 1 2 3 4 6 8 12 16
// 24 32.
template <typename T = Operon::Scalar> class IntervalJetEvaluator {
public:
    using Scalar = T;
    using Interval = pappus::interval<T>;
    using Domain = std::pair<Operon::Scalar, Operon::Scalar>;
    using DomainMap = Operon::Map<Operon::Hash, Domain>;
    using Bounds = IntervalJetBounds<T>;

    static constexpr std::size_t MaxDirections = Bounds::MaxDirections;

    // `variables`: distinct variable hashes, 1..MaxDirections of them; direction k differentiates variables[k].
    IntervalJetEvaluator(
        gsl::not_null<Operon::Tree const*> tree, DomainMap const& domains, Operon::Span<Operon::Hash const> variables)
        : tree_(tree)
        , domains_(domains)
        , variables_(variables.begin(), variables.end())
    {
        EXPECT(std::ranges::adjacent_find(variables_) == variables_.end() || variables_.size() < 2);
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
            return tl::unexpected(fmt::format("IntervalJetEvaluator: {} differentiated variables (supported: 1..{})",
                variables_.size(), MaxDirections));
        }
    }

    // Benchmark entry point: the same pass for 1..32 differentiated variables. The direction count must be one
    // of detail::IntervalJetWideDirections (1 2 3 4 6 8 12 16 24 32); every other count is an error. Derivative[k] and
    // Status[k] are meaningful for k < Directions as in operator(). Only instantiated by callers of this method.
    [[nodiscard]] auto EvaluateWide(Operon::Span<Operon::Scalar const> coeff) const
        -> tl::expected<IntervalJetWideBounds<T>, std::string>
    {
        return RunWide<0>(coeff);
    }

private:
    struct Slot {
        Domain Bounds {};
        int Direction { -1 };
        bool Present { false };
    };

    template <std::size_t I>
    [[nodiscard]] auto RunWide(Operon::Span<Operon::Scalar const> coeff) const
        -> tl::expected<IntervalJetWideBounds<T>, std::string>
    {
        if constexpr (I == detail::IntervalJetWideDirections.size()) {
            return tl::unexpected(
                fmt::format("IntervalJetEvaluator: {} differentiated variables is not a supported wide direction count",
                    variables_.size()));
        } else {
            if (static_cast<std::size_t>(detail::IntervalJetWideDirections[I]) == variables_.size()) {
                return Run<detail::IntervalJetWideDirections[I], IntervalJetWideBounds<T>>(coeff);
            }
            return RunWide<I + 1>(coeff);
        }
    }

    template <int K, typename B = Bounds>
    [[nodiscard]] auto Run(Operon::Span<Operon::Scalar const> coeff) const -> tl::expected<B, std::string>
    {
        static_assert(detail::IntervalJetWideIndex(K) < detail::IntervalJetWideDirections.size());
        static_assert(static_cast<std::size_t>(K) <= B::MaxDirections);
        using Policy = detail::IntervalJetPolicy<T, K>;
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
                return tl::unexpected(
                    fmt::format("IntervalJetEvaluator: no domain bound for variable hash {}", node.HashValue));
            }
            return Policy::Variable(
                Interval { static_cast<Scalar>(slot.Bounds.first), static_cast<Scalar>(slot.Bounds.second) }, scale,
                slot.Direction);
        };

        auto& primal = std::get<detail::IntervalJetWideIndex(K)>(primal_);
        auto evaluated
            = detail::EvaluatePostOrder<Policy>(nodes, primal, typename Policy::Context {}, weight, bindLeaf);
        if (!evaluated) {
            return tl::unexpected(std::move(evaluated.error()));
        }
        if (naryDivision) {
            return tl::unexpected(std::string { "IntervalJetEvaluator: n-ary Div is not supported by the jet path" });
        }

        auto const& root = *evaluated;
        B out;
        out.Value = root.J.a;
        out.Directions = static_cast<std::size_t>(K);
        auto const valueFailed = root.J.a.is_empty();
        for (std::size_t k = 0; k < static_cast<std::size_t>(K); ++k) {
            out.Derivative[k] = root.J.v[k];
            if (((root.Unsup >> k) & 1U) != 0) {
                out.Status[k] = IntervalJetStatus::Uncertain;
            } else if (valueFailed || root.J.v[k].is_empty()) {
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
    mutable detail::IntervalJetScratch<T> primal_; // reused across calls
};

} // namespace Operon

#endif
