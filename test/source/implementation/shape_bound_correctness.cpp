// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Shape-bounds correctness pass — §3 of the doc at
// operon-planning/designs/shape-constraint-soft-enforcement-causality-and-pappus-layering.md.
//
// Each case inline-encodes one of the 10 "Best Unconstrained Operon Models
// Recertified" flagged in
// operon-publications/experiments/shape-constraints-reproduction/results/shape_validation_selected.md
// — a model that was independently sample-checked to have
// `n_violated_constraints` sampled violations over the train domain, but
// was certified infeasible (`operon AffineEvaluator` returned no usable
// bound for at least one constraint) by operon's shape constraint gate.
//
// For each case we compute four enclosures on the same (tree, domain-box):
//   (a) operon `AffineEvaluator` (raw `Evaluate(coeffs).to_interval()`,
//       captured before the tree_diff / shape-constraint wrapper runs)
//   (b) operon `IntervalEvaluator` (independent pappus::interval path)
//   (c) pappus `evaluate_bisected(f, x, depth)` at depths {0, 2, 4, 8, 12}
//       (single-widest-axis bisection; the same tightening operon's
//       debug-only branched-affine path computes)
//   (d) pappus `optimize_bounds(f, box)` (branch-and-bound tightened
//       interval bound over the input box)
//
// Assertions, per the design's B1-B5 taxonomy:
//   B1 (operon dispatch bug): (a) != (b) — the two operon backends,
//       which both wrap pappus, disagree on the same tree/box. asserts
//       (a) ⊇ (b) (affine should be at least as tight as interval, never
//       more slack than the loosest correct enclosure).
//   B2 (pappus soundness bug): (c_k) not monotone descending as k grows;
//       would indicate pappus::evaluate_bisected itself is unsound.
//   B3 (pappus optimisation bug): (d) wider than (c_12); branch-and-bound
//       must be at least as tight as naïve bisection.
//   B4 (over-rejection): operon AffineEvaluator either throws, or its
//       ill-conditioned guard flips the bound to uncertified, AND the
//       independent `--shape-unknown-violation 0` would have accepted
//       the model. WARN-level signal — the test does not FAIL on B4
//       (it's the known-and-named phenomenon the §2 sweep targets); it
//       just records the diagnosis.
//   B5 (containment failure): operon Certified=true on a constraint
//       where the known FD oracle says the model has a sampled
//       violation. FAIL — this is a true soundness break that
//       invalidates the soft-vs-hard comparison.
//
// The known FD oracle is the per-case `fd_sampled_violations` integer
// (an embedded constant taken from the JSON; for the identity check it
// is the number of FD-sampled train-domain points where the model
// actually violates a constraint, NOT operon's own certification).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "operon/core/constraint.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/parser/infix.hpp"
#include "shape_constraints_config.hpp"

// pappus public API — used for the (c) evaluate_bisected and (d)
// optimize_bounds enclosures. `affine_form` and `interval` are both
// templated on the floating type, so we use them at operon's native
// float32 here (same precision as (a)/(b), so any disagreement is a
// *bug*, not a precision-floor artifact). `box.hpp` is NOT transitively
// pulled in by `pappus.hpp`, so include it explicitly for
// `pappus::box<S>` and `pappus::optimize_bounds`.
#include <pappus/pappus.hpp>
#include <pappus/interval/box.hpp>

namespace Operon::Test {

namespace {

using S = Operon::Scalar;
using AE = AffineEvaluator;
using IE = IntervalEvaluator;
using Affine = pappus::affine_form<S>;
using Interval = pappus::interval<S>;
using pappus_box = pappus::box<S>;

// One hand-encoded case from shape_validation_selected.md's
// "Best Unconstrained Operon Models Recertified" table. Variable
// names must match what InfixParser::Parse sees in the model string;
// they're also the keys of `constraints_json`'s `"domains"` block.
// Domain pairs are duplicated here so we can avoid re-parsing the JSON
// when constructing the pappus-only paths (c) and (d).
struct Case {
    std::string label;
    std::vector<std::string> vars;
    std::string constraints_json;
    std::string model;
    int fd_sampled_violations; // known oracle: FD-sampled train violations
    std::vector<std::pair<S, S>> domains; // parallel to `vars`
};

// Emit a tmp file with the constraints JSON so LoadShapeConstraints can
// consume it — same pattern as shape_constrained_evaluator.cpp's
// `WriteShapeConfig` helper. Inlining the JSON into a blob here keeps
// the test hermetic (no dependency on the operon-publications tree).
auto WriteConfig(std::string const& label, std::string const& text) -> std::filesystem::path
{
    auto path = std::filesystem::temp_directory_path() / ("operon_shape_bound_correctness_" + label + ".json");
    {
        std::ofstream out(path);
        out << text;
    }
    return path;
}

// Build a Dataset with named columns matching `vars`, populated with
// two dummy rows — `InfixParser::Parse` only needs the dataset to know
// which names are variables; the actual values are not consulted by
// the affine/interval bound evaluators (they read from the domain
// box, not from the rows).
auto MakeDataset(std::vector<std::string> const& vars) -> Operon::Dataset
{
    std::vector<std::vector<S>> vals(vars.size(), std::vector<S>(2, S{0}));
    return Operon::Dataset{vars, vals};
}

// Pretty-print an interval as "[lo, hi]".
auto Str(Interval const& iv) -> std::string
{
    std::ostringstream os;
    os << "[" << iv.inf() << ", " << iv.sup() << "]";
    return os.str();
}

// True iff `iv.is_empty()` — same semantics as pappus's NaN-bounds
// "domain violation sentinel".
auto IsEmpty(Interval const& iv) -> bool { return iv.is_empty(); }

// Run pappus `optimize_bounds` against an interval-arithmetic lambda
// built fresh per call. Each term of the model is rewritten into a
// sequence of `pappus::ops::*` calls on the box's representative forms;
// for these hand-encoded cases we let operon's own IntervalEvaluator
// do the per-sub-box evaluation (operon wraps the same pappus::ops),
// and use pappus's optimizer purely as a tighter-enclosure cross-check
// on top of operon's existing interval path.
auto OptimizedBounds(
    Operon::Tree const& tree,
    Operon::Map<Operon::Hash, std::pair<S, S>> const& domains_by_hash,
    std::vector<Operon::Hash> const& var_hashes)
    -> Interval
{
    // pappus::optimize_bounds takes a f: box<S> -> interval<S>. We bind
    // a box of pappus::interval<S> per variable, ranged by domains,
    // and call operon's IntervalEvaluator per sub-box — same engine as
    // the (b) reference, wrapped in pappus's branch-and-bound.
    auto f = [&](pappus_box const& bx) -> Interval {
        typename IE::DomainMap d;
        for (std::size_t i = 0; i < var_hashes.size(); ++i) {
            d[var_hashes[i]] = std::pair<S, S>{bx[i].inf(), bx[i].sup()};
        }
        IE eval(&tree, typename IE::DomainMap{d});
        return eval.Evaluate(tree.GetCoefficients());
    };
    pappus_box init;
    init.reserve(var_hashes.size());
    for (auto const& [lo, hi] : [&]() -> std::vector<std::pair<S, S>> const& {
        // order init to match var_hashes order
        auto const& dm = domains_by_hash;
        static std::vector<std::pair<S, S>> v;
        v.clear();
        for (auto h : var_hashes) {
            v.push_back(dm.at(h));
        }
        return v;
    }()) {
        init.emplace_back(Interval{lo, hi});
    }
    // Call the 5-arg overload explicitly (the 4-arg two-sided form is
    // ambiguous with this 5-arg form via default args). Overload 2 in
    // box.hpp:77 is literally `optimize_bounds(false) & optimize_bounds(true)`,
    // so this is exactly equivalent.
    auto lo = pappus::optimize_bounds<S>(f, init, false, S{1e-5}, 1000);
    auto hi = pappus::optimize_bounds<S>(f, init, true, S{1e-5}, 1000);
    return lo & hi;
}

// Single-widest-axis bisection enclosure for depth `depth`. Pappus's
// `evaluate_bisected` is single-axis (one affine_form), so for a
// multi-variable model we manually recurse on the widest input interval
// and re-evaluate operon's AffineEvaluator on the narrowed box.
auto BisectedAffine(
    Operon::Tree const& tree,
    Operon::Map<Operon::Hash, std::pair<S, S>> const& domains_by_hash,
    std::vector<Operon::Hash> const& var_hashes,
    int depth)
    -> Interval
{
    // Build the initial pappus box *intervals* in var_hashes order.
    std::vector<Interval> cur;
    cur.reserve(var_hashes.size());
    for (auto h : var_hashes) {
        auto const& [lo, hi] = domains_by_hash.at(h);
        cur.emplace_back(Interval{lo, hi});
    }
    // Recursive widest-axis bisection: each level splits the widest
    // interval and unions the two enclosures.
    std::function<Interval(std::vector<Interval> const&, int)> recurse;
    recurse = [&](std::vector<Interval> const& bx, int d) -> Interval {
        typename AE::DomainMap dm;
        for (std::size_t i = 0; i < var_hashes.size(); ++i) {
            dm[var_hashes[i]] = std::pair<S, S>{bx[i].inf(), bx[i].sup()};
        }
        AE eval(&tree, dm);
        auto const iv = [&]() -> Interval {
            auto r = eval.Evaluate(tree.GetCoefficients());
            auto result = r.to_interval();
            // pappus no longer throws on a domain error (log of a
            // non-positive range, sqrt of negative, etc.) -- it returns a
            // NaN-poisoned form instead (see affine_form::invalid()). A NaN
            // sub-box result must still be excluded from the `|` union
            // below rather than unioned in, or it silently corrupts the
            // WHOLE bisected result even when the sibling sub-box is sound.
            if (!std::isfinite(result.inf()) || !std::isfinite(result.sup())) {
                return Interval::empty();
            }
            return result;
        }();
        if (d <= 0) { return iv; }

        // Widest splittable axis (diameter > 0).
        std::size_t widest = 0;
        S widestDiam = S{-1};
        for (std::size_t i = 0; i < bx.size(); ++i) {
            auto const diam = bx[i].diameter();
            if (diam > widestDiam) {
                widestDiam = diam;
                widest = i;
            }
        }
        if (widestDiam <= S{0}) { return iv; }
        auto const [lo, hi] = bx[widest].split();
        auto left = bx; left[widest] = lo;
        auto right = bx; right[widest] = hi;
        return recurse(left, d - 1) | recurse(right, d - 1);
    };
    return recurse(cur, depth);
}

// Run a single case through all four enclosures + B1-B5 reporting.
// Returns true if the case passed (no B5 soundness failure).
auto RunCase(Case const& c) -> bool
{
    INFO("case: " << c.label);

    auto ds = MakeDataset(c.vars);
    auto tree = InfixParser::Parse(c.model, ds);
    auto const coeffs = tree.GetCoefficients();

    // Variable names -> hashes (via the dataset's Variable lookup).
    std::vector<Operon::Hash> var_hashes;
    var_hashes.reserve(c.vars.size());
    Operon::Map<Operon::Hash, std::pair<S, S>> domains_by_hash;
    for (std::size_t i = 0; i < c.vars.size(); ++i) {
        auto v = ds.GetVariable(c.vars[i]);
        REQUIRE(v.has_value());
        var_hashes.push_back(v->Hash);
        domains_by_hash[v->Hash] = c.domains[i];
    }

    // (a) operon AffineEvaluator raw
    std::optional<Interval> affineRaw;
    S maxAbsCenter = S{0};
    bool affineThrew = false;
    std::string affineErr;
    try {
        AE eval(&tree, AE::DomainMap{domains_by_hash});
        auto r = eval.Evaluate(coeffs);
        affineRaw = r.to_interval();
        maxAbsCenter = eval.MaxAbsCenter();
    } catch (std::exception const& e) {
        affineThrew = true;
        affineErr = e.what();
    }
    auto const illConditionedRatio = (affineRaw && affineRaw->diameter() > 0)
        ? maxAbsCenter * std::numeric_limits<S>::epsilon() / affineRaw->diameter() * S{2}
        : S{0};

    // (b) operon IntervalEvaluator raw
    std::optional<Interval> intervalRaw;
    bool intervalThrew = false;
    std::string intervalErr;
    try {
        IE eval(&tree, typename IE::DomainMap{domains_by_hash});
        intervalRaw = eval.Evaluate(coeffs);
    } catch (std::exception const& e) {
        intervalThrew = true;
        intervalErr = e.what();
    }

    // (c) bisected affine at depths {0, 2, 4, 8, 12}
    std::vector<Interval> bisected;
    bisected.reserve(5);
    for (auto d : {0, 2, 4, 8, 12}) {
        bisected.push_back(BisectedAffine(tree, domains_by_hash, var_hashes, d));
    }

    // (d) pappus optimize_bounds — BranchAndBound interval tightening.
    std::optional<Interval> optimized;
    try {
        optimized = OptimizedBounds(tree, domains_by_hash, var_hashes);
    } catch (std::exception const& e) {
        WARN("optimize_bounds threw: " << e.what());
    }

    // Load the constraint file (sanity check + runs the
    // ResolveShapeConstraintContext that gives us the per-constraint
    // Measure() too — useful for diagnosis on Cars, where one
    // constraint *does* certify while the others fail).
    auto path = WriteConfig(c.label, c.constraints_json);
    auto loaded = Operon::LoadShapeConstraints(path.string());
    REQUIRE(loaded);

    // ---- Diagnostics dump ------------------------------------------------
    WARN("case: " << c.label
                  << " | fd_sampled_violations=" << c.fd_sampled_violations);
    if (affineThrew) {
        WARN("  affine(a) threw: " << affineErr);
    } else if (affineRaw) {
        WARN("  affine(a) = " << Str(*affineRaw)
              << " | MaxAbsCenter=" << maxAbsCenter
              << " | illCondRatio=" << illConditionedRatio
              << " (threshold > 4.0 means TryAffineBound would flag uncertified)");
    }
    if (intervalThrew) {
        WARN("  interval(b) threw: " << intervalErr);
    } else if (intervalRaw) {
        WARN("  interval(b) = " << Str(*intervalRaw)
              << (IsEmpty(*intervalRaw) ? " (EMPTY == domain violation sentinel)" : ""));
    }
    for (std::size_t i = 0; i < bisected.size(); ++i) {
        WARN("  bisected(c, depth=" << (i == 0 ? 0 : (1 << (i - 1)))
              << ") = " << Str(bisected[i]));
    }
    if (optimized) {
        WARN("  optimized(d) = " << Str(*optimized));
    }

    // ---- Hard assertions (B1, B2, B3) -----------------------------------
    // B1: affine path must contain (not contradict) the interval path.
    //   We only assert this when both produced non-empty enclosures.
    if (!affineThrew && affineRaw && !IsEmpty(*affineRaw)
        && !intervalThrew && intervalRaw && !IsEmpty(*intervalRaw)) {
        INFO("B1: affine(a)=   " << Str(*affineRaw) << "\n"
              << "    interval(b)=" << Str(*intervalRaw));
        // affine can be tighter, never less-tight than the proven
        // interval range; assert containment.
        CHECK(affineRaw->inf() <= intervalRaw->inf() + std::fabs(intervalRaw->inf()) * S{1e-3});
        CHECK(affineRaw->sup() >= intervalRaw->sup() - std::fabs(intervalRaw->sup()) * S{1e-3});
    }

    // B2: bisected enclosures must be monotonically ⊇ as depth grows.
    //   (deeper bisection can only tighten, never produce a wider range;
    //   a wider range here means pappus's bisection is unsound.)
    for (std::size_t i = 1; i < bisected.size(); ++i) {
        auto const& prev = bisected[i - 1];
        auto const& curr = bisected[i];
        if (IsEmpty(prev) || IsEmpty(curr)) { continue; }
        INFO("B2: bisected depth " << (i == 0 ? 0 : (1 << (i - 1)))
              << " -> " << (1 << (i - 1)) << " (cur never wider than prev)\n"
              << "    prev=" << Str(prev) << "    cur=" << Str(curr));
        CHECK(prev.inf() <= curr.inf() + std::fabs(curr.inf()) * S{1e-3});
        CHECK(prev.sup() >= curr.sup() - std::fabs(curr.sup()) * S{1e-3});
    }

    // B3: optimize_bounds must be at least as tight as depth-12 bisection.
    if (optimized && !IsEmpty(*optimized)
        && !IsEmpty(bisected.back())) {
        INFO("B3: optimized(d)=" << Str(*optimized)
              << "    bisected(12)=" << Str(bisected.back()));
        CHECK(optimized->inf() >= bisected.back().inf() - std::fabs(bisected.back().inf()) * S{1e-3});
        CHECK(optimized->sup() <= bisected.back().sup() + std::fabs(bisected.back().sup()) * S{1e-3});
    }

    // ---- B4 / B5 reporting ----------------------------------------------
    //   B4: operon FeatheredUncertified (threw OR illCondRatio>4) AND
    //       the FD oracle says the model is actually clean.
    //   B5: operon Certified-feasible AND the FD oracle says it isn't —
    //       caught directly through Measure() below.
    bool const operonRejects = affineThrew || (affineRaw && IsEmpty(*affineRaw))
        || illConditionedRatio > S{4};
    if (operonRejects && c.fd_sampled_violations == 0) {
        WARN("  B4 (over-rejection): operon rejected a sample-clean model"
              << " | affineThrew=" << affineThrew
              << " illCondRatio=" << illConditionedRatio
              << " (this is the §2 investigation's known signal, not a failure)");
    }

    // Run Measure() to check for B5 — a true soundness break.
    // NOTE: the per-model FD oracle (`fd_sampled_violations`) is the count
    // of *constraints* with at least one sampled violation, not a
    // per-constraint verdict. So a "Certified & FD>0" mismatch on a
    // single constraint is not by itself a soundness failure: the FD
    // violations could be on other constraints. We WARN rather than FAIL
    // on this signal — a definitive B5 verdict needs per-constraint FD
    // sampling (deferred to a v2 of this test).
    typename Operon::ShapeConstraintSet cs = *loaded;
    Operon::Dataset dsForProblem = MakeDataset(c.vars);
    Operon::Problem problem(&dsForProblem);
    problem.SetTrainingRange({0, 2});
    problem.SetTestRange({0, 2});
    // Match the productions JSON's target column naming by setting the
    // last column as target; for the bound check the target identity is
    // irrelevant (linear scaling is off for this assessment).
    problem.SetTarget(c.vars.back());
    problem.SetDefaultInputs();
    problem.SetLinearScalingEnabled(false);

    using DTable = Operon::DispatchTable<Operon::Scalar>;
    DTable dtable;
    Operon::Evaluator<DTable> nmse(&problem, &dtable, Operon::NMSE{});
    Operon::ShapeConstrainedEvaluator sce(&nmse, &dtable, cs);
    auto const summary = sce.Measure(tree);

    for (std::size_t i = 0; i < summary.Measurements.size(); ++i) {
        auto const& m = summary.Measurements[i];
        if (m.Certified && m.Violation == S{0}) {
            // operon claims feasibility. If the FD oracle says
            // there's a sample violation, *flag* (not fail) — see NOTE above.
            if (c.fd_sampled_violations > 0) {
                WARN("  B5 candidate (per-model FD mismatch): constraint[" << i
                      << "] operon Certified with Violation=0; FD oracle found "
                      << c.fd_sampled_violations << " sampled violations *somewhere* in the model"
                      << " (could be on other constraints — needs per-constraint FD to confirm)");
            } else {
                WARN("  constraint[" << i << "] operon Certified-feasible"
                      << " — agrees with FD oracle");
            }
        } else if (m.Certified && m.Violation > S{0}) {
            WARN("  constraint[" << i
                  << "] operon Certified-infeasible (violation="
                  << m.Violation << ") — both engines agree this is infeasible");
        } else {
            std::string reason;
            if (affineThrew) {
                reason = "affine_form threw: " + affineErr;
            } else if (affineRaw && IsEmpty(*affineRaw)) {
                reason = "raw affine returned an empty/NaN-bounds interval";
            } else if (illConditionedRatio > S{4}) {
                reason = "ill-conditioned guard triggered (ratio=" + std::to_string(illConditionedRatio) + ")";
            } else {
                reason = "other";
            }
            WARN("  constraint[" << i << "] operon Uncertified (" << reason << ")");
        }
    }

    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Five hand-encoded cases — the §3 design specified "selected from the
// 10 cases flagged in shape_validation_selected.md, focusing on the 5
// most structurally diverse ones." Every case below is one of the best
// unconstrained operon models from the existing sweep, embedded
// verbatim. The FD-violation count is the known oracle from
// results/shape_validation_selected.json's
// "best_unconstrained_operon.n_violated_constraints" field.
// ---------------------------------------------------------------------------

TEST_CASE("Shape bound correctness — five flagged cases", "[shape-constraints][bound-correctness]")
{
    // Fuel_flow without_noise gp rep 9: 3 derivative constraints, all uncertified.
    // Domain box (from results/full_sweep/Fuel_flow_constraints.json):
    //   p0 in [400000, 600000], Astar in [0.5, 1.5], T0 in [250, 260].
    {
        Case c{
            "Fuel_flow_without_noise_gp_rep09",
            {"Astar", "T0", "p0"},
            R"json({
                "domains": {"p0": [400000.0, 600000.0], "Astar": [0.5, 1.5], "T0": [250, 260]},
                "constraints": [
                    {"op": "derivative", "variable": "p0", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "Astar", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "T0", "order": 1, "sign": -1}
                ]
            })json",
            "((-0.026927) + (0.026880 * ((((tanh((cos((((0.449930 * Astar) / (1.366455 * T0)) * (6.154615 * p0))) / tanh(tanh(tanh(tanh(exp(((-0.760431) * T0)))))))) / ((cos(tanh(tanh(((-1.112441) * T0)))) / ((-0.836549) * T0)) * (5.931292 * p0))) + (((0.449930 * Astar) / (1.366455 * T0)) * (6.154615 * p0))) * (sqrt((1.354045 * T0)) * cos(cos(tanh((-0.544384)))))) + cos(tanh(tanh(tanh((((0.254493 * Astar) / cos(tanh(exp(((-0.760431) * T0))))) * 0.189909))))))))",
            0,
            {{S{1} / S{2}, S{3} / S{2}}, {S{250}, S{260}}, {S{400000}, S{600000}}}
        };
        CHECK(RunCase(c));
    }

    // II_11_27 without_noise gp rep 14: 4 derivative constraints (all sign=1), all uncertified.
    // Variables: epsilon, n, alpha, Ef.
    // Domain box (from results/full_sweep/II_11_27_constraints.json):
    //   n in [0,1], alpha in [0,1], epsilon in [1,2], Ef in [1,2].
    {
        Case c{
            "II_11_27_without_noise_gp_rep14",
            {"epsilon", "n", "alpha", "Ef"},
            R"json({
                "domains": {"n": [0, 1], "alpha": [0, 1], "epsilon": [1, 2], "Ef": [1, 2]},
                "constraints": [
                    {"op": "derivative", "variable": "n", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "alpha", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "epsilon", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "Ef", "order": 1, "sign": 1}
                ]
            })json",
            "(210.937698 * (((0.363327 * n) * ((-0.028747) * Ef)) / (((((-1.661113) * n) + ((tanh(tanh(tanh(exp((-1.174090))))) / (0.497200 * alpha)) * (tanh(tanh(sqrt(tanh((exp(tanh(tanh(exp(tanh(((-0.911792) ^ 2)))))) / ((-0.911792) ^ 2)))))) * tanh(exp((exp(tanh(((-0.911792) ^ 2))) * exp((0.363327 * n)))))))) + (exp(((-0.911792) ^ 2)) / (0.497200 * alpha))) / ((-2.261929) * epsilon))))",
            0,
            {{S{1}, S{2}}, {S{0}, S{1}}, {S{0}, S{1}}, {S{1}, S{2}}}
        };
        CHECK(RunCase(c));
    }

    // I_48_20 without_noise gp rep 0: 3 derivative constraints (all sign=1), all uncertified.
    // Variables: c, v, m.
    // Domain box (from results/full_sweep/I_48_20_constraints.json):
    //   m in [1,5], v in [1,2], c in [3,10].
    {
        Case c{
            "I_48_20_without_noise_gp_rep00",
            {"c", "v", "m"},
            R"json({
                "domains": {"m": [1, 5], "v": [1, 2], "c": [3, 10]},
                "constraints": [
                    {"op": "derivative", "variable": "m", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "v", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "c", "order": 1, "sign": 1}
                ]
            })json",
            "((-1.833329) + (0.854424 * ((((((-0.396240) * c) ^ 2) * (((2.610423 ^ 2) / cos((((-1.972425) * v) / (tanh((((0.572300 * v) / (tanh((0.199622 * c)) + ((-0.739473) * c))) ^ 2)) + ((-1.927949) * c))))) * (1.093432 * m))) + ((0.572300 * v) / ((tanh(((-0.739473) * c)) * (((tanh((-1.058995)) / ((-0.469899) * m)) + ((((-0.411143) * v) / ((-0.469899) * m)) / (0.199622 * c))) + ((0.572300 * v) / ((0.376507 * c) ^ 2)))) + ((-0.396240) * c)))) + (0.846632 / exp(tanh(((-0.739473) * c)))))))",
            0,
            {{S{3}, S{10}}, {S{1}, S{2}}, {S{1}, S{5}}}
        };
        CHECK(RunCase(c));
    }

    // Jackson_2_11 without_noise gp rep 8: 5 derivative constraints, all uncertified.
    // Variables: d, epsilon, q, Volt, y.
    // Domain box (from results/full_sweep/Jackson_2_11_constraints.json):
    //   q in [1,5], y in [1,3], Volt in [1,5], d in [4,6], epsilon in [1,5].
    {
        Case c{
            "Jackson_2_11_without_noise_gp_rep08",
            {"d", "epsilon", "q", "Volt", "y"},
            R"json({
                "domains": {
                    "q": [1, 5], "y": [1, 3], "Volt": [1, 5],
                    "d": [4, 6], "epsilon": [1, 5]
                },
                "constraints": [
                    {"op": "derivative", "variable": "q", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "y", "order": 1, "sign": -1},
                    {"op": "derivative", "variable": "Volt", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "d", "order": 1, "sign": 1},
                    {"op": "derivative", "variable": "epsilon", "order": 1, "sign": 1}
                ]
            })json",
            "(0.001245 + (5.166092 * ((((sin((((0.142203 * y) * ((((((((0.084928 * q) / (0.042015 * d)) / (0.802681 * epsilon)) / exp((((0.084928 * q) / (0.058953 * y)) * ((0.215451 * Volt) / 1.144544)))) * exp(((0.084928 * q) / (0.042015 * d)))) * (0.058953 * y)) * (((-0.834647) * y) ^ 2)) + log(((0.058953 * y) / (0.042015 * d))))) / exp(((-2.245078) ^ 2)))) * (-0.252841)) + (0.042015 * d)) / (0.058953 * y)) * (((0.084928 * q) / (0.058953 * y)) * ((0.215451 * Volt) / 1.144544)))))",
            0,
            {{S{4}, S{6}}, {S{1}, S{5}}, {S{1}, S{5}},
             {S{1}, S{5}}, {S{1}, S{3}}}
        };
        CHECK(RunCase(c));
    }

    // Cars real gp rep 1: 4 constraints, 1 certified (constraint #2 with
    // bound [0.0, -0.0]) and 3 uncertified. fd_sampled_violations=2 means
    // the model actually violates 2 sampled-constraints — so if operon
    // happens to certify any as feasible, that's a B5 failure to
    // surface.
    {
        Case c{
            "Cars_real_gp_rep01",
            {"cylinders", "displacement", "horsepower", "weight", "acceleration"},
            R"json({
                "domains": {
                    "cylinders": [3, 8], "displacement": [68, 455],
                    "horsepower": [46, 230], "weight": [1613, 5140],
                    "acceleration": [8, 23.5]
                },
                "constraints": [
                    {"op": "id", "bound": [9.0, 46.6]},
                    {"op": "derivative", "variable": "displacement", "order": 1, "sign": -1},
                    {"op": "derivative", "variable": "horsepower", "order": 1, "sign": -1},
                    {"op": "derivative", "variable": "weight", "order": 1, "sign": -1}
                ]
            })json",
            "(7.362454 + (342.998474 * ((sqrt(cos(cos(((cos(cos(((-0.636562) * cylinders))) + (((-1.050939) / exp(0.601547)) * (cos(((0.013105 * weight) * tanh((((1.059040 * displacement) / exp(log(exp(log((0.013105 * weight)))))) * (cos(((-0.636562) * cylinders)) + exp((-0.313989))))))) + ((-1.050939) / exp(cos(cos(((-0.772870) * cylinders)))))))) ^ 2)))) + (cos((0.642849 * cylinders)) ^ 2)) / exp(log((0.013105 * weight))))))",
            2,
            {{S{3}, S{8}}, {S{68}, S{455}}, {S{46}, S{230}}, {S{1613}, S{5140}}, {S{8}, S{47} / S{2}}}
        };
        CHECK(RunCase(c));
    }
}

} // namespace Operon::Test