// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Shared helpers of the interval-jet tests: a double-precision reference evaluator over a tree (forward-mode
// Jet<double, 4>), random point sampling inside a box, and containment bookkeeping.

#ifndef OPERON_TEST_INTERVAL_JET_SUPPORT_HPP
#define OPERON_TEST_INTERVAL_JET_SUPPORT_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <pappus/pappus.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/jet.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interval_jet_evaluator.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test::JetSupport {

using S = Operon::Scalar;
using Interval = pappus::interval<S>;
using RefJet = Operon::Jet<double, 4>;
using DomainMap = Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>>;

inline constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// Dataset with variables X1..Xn (two dummy rows): only the names matter to the parser.
inline auto MakeDataset(std::size_t nvars) -> Operon::Dataset
{
    std::vector<std::string> names;
    for (std::size_t i = 1; i <= nvars; ++i) {
        names.push_back(fmt::format("X{}", i));
    }
    std::vector<std::vector<S>> vals(nvars, std::vector<S>(2, S { 1 }));
    return Operon::Dataset { names, vals };
}

inline auto VarHash(Operon::Dataset const& ds, std::size_t i) -> Operon::Hash
{
    return ds.GetVariable(fmt::format("X{}", i))->Hash;
}

// Double-precision forward-mode reference for the whole tree at one point of the box. Direction k differentiates
// vars[k]. Weights/coefficients are consumed exactly as IntervalEvaluator does (Optimize nodes in order).
//
// `regular` is cleared when the point is not a regular point of the function, i.e. one where the interval
// semantics and the double reference legitimately disagree about the function: a non-finite (or beyond float
// range) value or tangent anywhere in the tree (singular endpoints, NaN-swallowing fmin/fmax, overflow), or a
// pow whose base is <= 0 while its exponent depends on a variable (the interval pow clips such a base to
// [0, inf) and treats 0^y as 0, whereas the reference evaluates std::pow on the point).
inline auto Reference(Operon::Tree const& tree, Operon::Span<Operon::Scalar const> coeff,
    std::span<Operon::Hash const> vars, Operon::Map<Operon::Hash, double> const& point, bool& regular,
    std::vector<RefJet>* nodeValues = nullptr) -> RefJet
{
    auto const& nodes = tree.Nodes();
    std::vector<RefJet> val;
    std::vector<bool> dependsOnVariable;
    val.reserve(nodes.size());
    dependsOnVariable.reserve(nodes.size());
    regular = true;
    constexpr double FloatRange = 3.0e38;
    auto const finite = [&](RefJet const& j) {
        auto const ok = [](double x) { return std::isfinite(x) && std::abs(x) < FloatRange; };
        return ok(j.a) && ok(j.v[0]) && ok(j.v[1]) && ok(j.v[2]) && ok(j.v[3]);
    };
    std::size_t ci = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto const& n = nodes[i];
        if (n.IsRef()) {
            val.push_back(val[n.RefTo]);
            dependsOnVariable.push_back(dependsOnVariable[n.RefTo]);
            continue;
        }
        double const w = n.Optimize ? static_cast<double>(coeff[ci++]) : static_cast<double>(n.Value);
        if (n.IsConstant()) {
            val.emplace_back(w);
            dependsOnVariable.push_back(false);
            continue;
        }
        if (n.IsVariable()) {
            std::array<double, 4> t {};
            for (std::size_t k = 0; k < vars.size(); ++k) {
                if (vars[k] == n.HashValue) {
                    t[k] = w;
                }
            }
            auto const it = point.find(n.HashValue);
            val.emplace_back(w * (it != point.end() ? it->second : NaN), t);
            dependsOnVariable.push_back(true);
            continue;
        }

        std::vector<std::size_t> ch;
        bool dependent = false;
        for (auto c : Operon::Tree::Indices(nodes, i)) {
            ch.push_back(c);
            dependent = dependent || dependsOnVariable[c];
        }
        auto const fold = [&](RefJet acc, auto op) {
            for (std::size_t m = 0; m < ch.size(); ++m) {
                acc = op(acc, val[ch[m]]);
            }
            return acc;
        };
        RefJet r;
        auto const& x = val[ch[0]];
        switch (n.HashValue) {
        case Operon::Hash(BuiltinOp::Add):
            r = fold(RefJet { 0.0 }, [](RefJet const& a, RefJet const& b) { return a + b; });
            break;
        case Operon::Hash(BuiltinOp::Mul):
            r = fold(RefJet { 1.0 }, [](RefJet const& a, RefJet const& b) { return a * b; });
            break;
        case Operon::Hash(BuiltinOp::Sub):
            if (ch.size() == 1) {
                r = -x;
            } else {
                r = x;
                for (std::size_t m = 1; m < ch.size(); ++m) {
                    r = r - val[ch[m]];
                }
            }
            break;
        case Operon::Hash(BuiltinOp::Div):
            if (ch.size() == 1) {
                r = 1.0 / x;
            } else {
                r = x;
                for (std::size_t m = 1; m < ch.size(); ++m) {
                    r = r / val[ch[m]];
                }
            }
            break;
        case Operon::Hash(BuiltinOp::Fmin):
            r = x;
            for (std::size_t m = 1; m < ch.size(); ++m) {
                r = Operon::fmin(r, val[ch[m]]);
            }
            break;
        case Operon::Hash(BuiltinOp::Fmax):
            r = x;
            for (std::size_t m = 1; m < ch.size(); ++m) {
                r = Operon::fmax(r, val[ch[m]]);
            }
            break;
        case Operon::Hash(BuiltinOp::Pow):
            if (x.a <= 0.0 && dependsOnVariable[ch[1]]) {
                regular = false;
            }
            r = Operon::pow(x, val[ch[1]]);
            break;
        case Operon::Hash(BuiltinOp::Aq):
            r = Operon::aq(x, val[ch[1]]);
            break;
        case Operon::Hash(BuiltinOp::Powabs):
            r = Operon::powabs(x, val[ch[1]]);
            break;
        case Operon::Hash(BuiltinOp::Abs):
            r = Operon::abs(x);
            break;
        case Operon::Hash(BuiltinOp::Acos):
            r = Operon::acos(x);
            break;
        case Operon::Hash(BuiltinOp::Asin):
            r = Operon::asin(x);
            break;
        case Operon::Hash(BuiltinOp::Atan):
            r = Operon::atan(x);
            break;
        case Operon::Hash(BuiltinOp::Cbrt):
            r = Operon::cbrt(x);
            break;
        case Operon::Hash(BuiltinOp::Ceil):
            r = Operon::ceil(x);
            break;
        case Operon::Hash(BuiltinOp::Cos):
            r = Operon::cos(x);
            break;
        case Operon::Hash(BuiltinOp::Cosh):
            r = Operon::cosh(x);
            break;
        case Operon::Hash(BuiltinOp::Exp):
            r = Operon::exp(x);
            break;
        case Operon::Hash(BuiltinOp::Floor):
            r = Operon::floor(x);
            break;
        case Operon::Hash(BuiltinOp::Log):
            r = Operon::log(x);
            break;
        case Operon::Hash(BuiltinOp::Logabs):
            r = Operon::logabs(x);
            break;
        case Operon::Hash(BuiltinOp::Log1p):
            r = Operon::log1p(x);
            break;
        case Operon::Hash(BuiltinOp::Sin):
            r = Operon::sin(x);
            break;
        case Operon::Hash(BuiltinOp::Sinh):
            r = Operon::sinh(x);
            break;
        case Operon::Hash(BuiltinOp::Sqrt):
            r = Operon::sqrt(x);
            break;
        case Operon::Hash(BuiltinOp::Sqrtabs):
            r = Operon::sqrtabs(x);
            break;
        case Operon::Hash(BuiltinOp::Tan):
            r = Operon::tan(x);
            break;
        case Operon::Hash(BuiltinOp::Tanh):
            r = Operon::tanh(x);
            break;
        case Operon::Hash(BuiltinOp::Square):
            r = Operon::square(x);
            break;
        default:
            r = RefJet { NaN };
            break;
        }
        val.push_back(r * w);
        dependsOnVariable.push_back(dependent);
        regular = regular && finite(val.back());
    }
    if (nodeValues != nullptr) {
        *nodeValues = val;
    }
    return val.back();
}

// A bound as double endpoints (any scalar type of interval); empty bounds have lo > hi or NaN endpoints.
using Bnd = std::pair<double, double>;

template <typename T> auto ToBnd(pappus::interval<T> const& i) -> Bnd
{
    return { static_cast<double>(i.inf()), static_cast<double>(i.sup()) };
}

// Soundness bookkeeping of one bound family (jet or dag) against sampled reference values.
struct Containment {
    // A reference value may leave the bound by Slack * (1 + |reference|) before it counts as a violation: float
    // pappus arithmetic is only outward rounded to within a few ulp per operation (see report), which
    // cancellation can amplify; the double instantiation of the jet evaluator is checked with a tiny slack.
    double Slack { 1e-6 };
    explicit Containment(double slack = 1e-6) : Slack(slack) {}
    std::size_t Checked {}; // finite reference values/derivatives compared against a bound
    std::size_t Violations {}; // outside the bound by more than the slack
    std::size_t Strict {}; // outside the bound at all (informational: double reference vs float bound)
    double WorstExcess {}; // largest violation distance
    std::vector<std::string> Examples; // first few violations, for diagnosis

    // Returns whether `reference` violates `bound` by more than the slack.
    auto Check(Bnd const& bound, double reference) -> bool
    {
        if (!std::isfinite(reference)) {
            return false;
        }
        ++Checked;
        auto const empty = !(bound.first <= bound.second);
        auto const excess = empty ? std::numeric_limits<double>::infinity()
                                  : std::max(bound.first - reference, reference - bound.second);
        if (excess > 0.0) {
            ++Strict;
        }
        if (excess > Slack * (1.0 + std::abs(reference))) {
            ++Violations;
            WorstExcess = std::max(WorstExcess, excess);
            return true;
        }
        return false;
    }
};

// A set of enclosures under test: the value and, per direction, the derivative enclosure if it is claimed
// certain (nullopt: not claimed, nothing to check).
struct BoundSet {
    std::optional<Bnd> Value;
    std::array<std::optional<Bnd>, 4> Derivative {};
};

// Draws `samples` points of the box (corners, centre and uniform interior points; every variable of `domains`
// is sampled independently) and checks every claimed enclosure of `set` against the reference.
inline auto SampleCheck(Operon::Tree const& tree, Operon::Span<Operon::Scalar const> coeff,
    std::span<Operon::Hash const> vars, DomainMap const& domains, BoundSet const& set, std::mt19937_64& rng,
    std::size_t samples, Containment& valueStats, Containment& derivativeStats) -> void
{
    for (std::size_t s = 0; s < samples; ++s) {
        Operon::Map<Operon::Hash, double> point;
        for (auto const& [hash, bounds] : domains) {
            double const lo = bounds.first;
            double const hi = bounds.second;
            double x = lo;
            switch (s % 4) {
            case 0:
                x = lo + (hi - lo) * std::uniform_real_distribution<double> { 0.0, 1.0 }(rng);
                break;
            case 1:
                x = (s % 8 == 1) ? lo : hi;
                break;
            case 2:
                x = 0.5 * lo + 0.5 * hi;
                break;
            default:
                x = lo + (hi - lo) * std::uniform_real_distribution<double> { 0.0, 1.0 }(rng);
                break;
            }
            point.insert_or_assign(hash, std::clamp(x, lo, hi));
        }
        bool regular = true;
        auto const ref = Reference(tree, coeff, vars, point, regular);
        if (!regular) {
            continue;
        }
        auto const where = [&] {
            std::string text;
            for (auto const& [hash, x] : point) {
                text += fmt::format(" {:x}={:.9g}", hash, x);
            }
            return text;
        };
        if (set.Value && std::isfinite(ref.a)) {
            if (valueStats.Check(*set.Value, ref.a) && valueStats.Examples.size() < 3) {
                valueStats.Examples.push_back(fmt::format(
                    "value {:.9g} outside [{}, {}] at{}", ref.a, set.Value->first, set.Value->second, where()));
            }
        }
        if (!std::isfinite(ref.a)) {
            continue; // the reference function is undefined here: the box leaves the domain
        }
        for (std::size_t k = 0; k < vars.size(); ++k) {
            if (set.Derivative[k] && derivativeStats.Check(*set.Derivative[k], ref.v[k])
                && derivativeStats.Examples.size() < 3) {
                derivativeStats.Examples.push_back(fmt::format("d/dvar{} {:.9g} outside [{}, {}] at{}", k, ref.v[k],
                    set.Derivative[k]->first, set.Derivative[k]->second, where()));
            }
        }
    }
}

// Converts an IntervalJetBounds into a BoundSet claiming the certain directions.
template <typename T> auto ToBoundSet(Operon::IntervalJetBounds<T> const& jet) -> BoundSet
{
    BoundSet set;
    if (!jet.Value.is_empty()) {
        set.Value = ToBnd(jet.Value);
    }
    for (std::size_t k = 0; k < jet.Directions; ++k) {
        if (jet.Status[k] == Operon::IntervalJetStatus::Certain) {
            set.Derivative[k] = ToBnd(jet.Derivative[k]);
        }
    }
    return set;
}

} // namespace Operon::Test::JetSupport

#endif
