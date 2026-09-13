// Throwaway pilot: does per-node collapse (HybridEvaluator) actually
// improve the ROOT bound over today's root-only `combined` mode, on real
// trees? Not permanent code.
//
// Revision: fixed two bugs an implementation review found in the first
// draft: (1) the non-overlapping-intersection fallback for `combined` used
// min(affine_width, interval_width), but production TryAffineBoundDirect
// treats non-overlap as "one of the two is unsound" and falls back to
// affine alone, not whichever is narrower -- fixed to match. (2) models
// were excluded entirely whenever EITHER backend's plain root evaluation
// was non-finite, before even trying Hybrid -- this discarded exactly the
// cases Hybrid's own force-collapse-on-invalid-affine policy is meant to
// help with. Fixed: now include (affine invalid, interval valid) cases,
// mirroring production's own affine-then-interval-fallback semantics for
// what "combined" means when affine alone fails.
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <limits>

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/parser/infix.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/hybrid_evaluator.hpp"

using namespace Operon;

struct BoundInfo { bool valid = false; double lo = 0, hi = 0; };

auto MakeBoundInfo(pappus::interval<Scalar> const& iv) -> BoundInfo {
    BoundInfo b;
    if (iv.is_empty() || !std::isfinite(iv.inf()) || !std::isfinite(iv.sup())) { return b; }
    b.valid = true; b.lo = static_cast<double>(iv.inf()); b.hi = static_cast<double>(iv.sup());
    return b;
}

// Mirrors TryAffineBoundDirect's own combined-mode policy exactly:
// - affine invalid -> interval alone (production's non-finite-affine fallback)
// - interval invalid/unavailable -> affine alone (production's "interval
//   fallback failed" branch, which just returns the affine bound)
// - both valid, overlapping -> intersection
// - both valid, NOT overlapping -> affine alone (production comment:
//   "a non-overlapping result would mean one of the two is unsound... fall
//   back to the affine bound alone rather than construct an inverted interval")
// - both invalid -> no result
auto Combined(BoundInfo const& a, BoundInfo const& iv) -> BoundInfo {
    if (!a.valid && !iv.valid) { return {}; }
    if (!a.valid) { return iv; }
    if (!iv.valid) { return a; }
    auto lo = std::max(a.lo, iv.lo);
    auto hi = std::min(a.hi, iv.hi);
    if (lo <= hi) { return {true, lo, hi}; }
    return a; // non-overlap: affine alone, matching production, NOT min-width
}

auto Width(BoundInfo const& b) -> double { return b.hi - b.lo; }

struct Result {
    BoundInfo affineOnly, intervalOnly, hybrid, combined;
    bool parsedOk = false;
};

auto RunOne(std::string const& infix, Dataset const& ds,
            AffineEvaluator::DomainMap const& adom, IntervalEvaluator::DomainMap const& idom,
            std::size_t& collapses, std::size_t& internalNodes) -> Result {
    Result res;
    Tree tree;
    try { tree = InfixParser::Parse(infix, ds); }
    catch (...) { return res; }
    res.parsedOk = true;
    auto coeff = tree.GetCoefficients();

    BoundInfo affineForCombined; // affine, but treated invalid if ill-conditioned (see below)
    try {
        AffineEvaluator ae(&tree, adom);
        auto ares = ae.Evaluate(coeff);
        res.affineOnly = MakeBoundInfo(ares.to_interval());
        // Mirror TryAffineBoundDirect's own ill-conditioning guard exactly
        // (shape_constrained_evaluator.cpp): a finite-looking affine bound
        // can still be untrustworthy if an intermediate center implies a
        // rounding-error floor exceeding the tracked radius. Production
        // falls back to interval-alone in that case; the pilot's combined
        // baseline must do the same, or it's more optimistic about affine
        // than production's own combined mode actually is.
        constexpr auto eps = std::numeric_limits<Scalar>::epsilon();
        auto const impliedErrorFloor = ae.MaxAbsCenter() * eps;
        auto const r = ares.radius();
        bool const illConditioned = (r > Scalar{0} && impliedErrorFloor > Scalar{4} * r);
        affineForCombined = illConditioned ? BoundInfo{} : res.affineOnly;
    } catch (...) {}
    try {
        IntervalEvaluator ie(&tree, idom);
        res.intervalOnly = MakeBoundInfo(ie.Evaluate(coeff));
    } catch (...) {}
    try {
        HybridEvaluator he(&tree, adom);
        res.hybrid = MakeBoundInfo(he.Evaluate(coeff).to_interval());
        collapses += he.CollapseCount();
        internalNodes += he.InternalNodeCount();
    } catch (...) {}

    res.combined = Combined(affineForCombined, res.intervalOnly);
    return res;
}

auto main(int argc, char** argv) -> int {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <models_file>\n", argv[0]); return 1; }
    std::vector<std::string> varnames = {"Astar","B","Bx","By","Bz","C_La","C_Lde","Ef","G","Gamma",
        "R","SHT","Sref","T","T0","Vinf","Volt","acceleration","alpha","c","cylinders","d","de",
        "displacement","epsilon","h","horsepower","kb","lambd","m","m1","m2","mom","n","n_rho",
        "omega","omega_0","p","p0","p_d","phi","phi_dot","q","r","sigma","t","theta","u","v",
        "weight","x","x1","x2","y","y1","y2","z","z1","z2"};
    std::vector<std::vector<Scalar>> vals(varnames.size(), std::vector<Scalar>(2, Scalar{1}));
    Dataset ds(varnames, vals);
    AffineEvaluator::DomainMap adom;
    IntervalEvaluator::DomainMap idom;
    for (auto const& name : varnames) {
        if (auto v = ds.GetVariable(name)) {
            adom[v->Hash] = {Scalar{0.1}, Scalar{10}};
            idom[v->Hash] = {Scalar{0.1}, Scalar{10}};
        }
    }

    FILE* f = std::fopen(argv[1], "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    char line[1 << 20];
    int n = 0, parsed = 0;
    long hybridBeatsCombined = 0, combinedBeatsHybrid = 0, tiedHC = 0, noCombined = 0, noHybrid = 0;
    long hybridBeatsAffine = 0, affineBeatsHybrid = 0, tiedHA = 0, noAffine = 0;
    std::size_t collapses = 0, internalNodes = 0;
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
        if (s.empty()) { continue; }
        ++n;
        auto r = RunOne(s, ds, adom, idom, collapses, internalNodes);
        if (!r.parsedOk) { continue; }
        ++parsed;

        if (!r.hybrid.valid) { ++noHybrid; continue; }
        if (!r.combined.valid) { ++noCombined; }
        else {
            double reltol = 1e-6 * std::max({1.0, Width(r.hybrid), Width(r.combined)});
            if (std::fabs(Width(r.hybrid) - Width(r.combined)) <= reltol) { ++tiedHC; }
            else if (Width(r.hybrid) < Width(r.combined)) { ++hybridBeatsCombined; }
            else { ++combinedBeatsHybrid; }
        }

        if (!r.affineOnly.valid) { ++noAffine; }
        else {
            double reltol = 1e-6 * std::max({1.0, Width(r.hybrid), Width(r.affineOnly)});
            if (std::fabs(Width(r.hybrid) - Width(r.affineOnly)) <= reltol) { ++tiedHA; }
            else if (Width(r.hybrid) < Width(r.affineOnly)) { ++hybridBeatsAffine; }
            else { ++affineBeatsHybrid; }
        }
    }
    std::fclose(f);

    std::printf("models=%d parsed=%d hybrid_invalid=%ld combined_unavailable=%ld affine_unavailable=%ld\n",
        n, parsed, noHybrid, noCombined, noAffine);
    std::printf("collapses=%zu / internal_nodes=%zu (%.1f%%)\n", collapses, internalNodes,
        internalNodes ? 100.0 * static_cast<double>(collapses) / static_cast<double>(internalNodes) : 0.0);
    std::printf("ROOT: hybrid vs combined   : hybrid_tighter=%ld tied=%ld combined_tighter=%ld\n",
        hybridBeatsCombined, tiedHC, combinedBeatsHybrid);
    std::printf("ROOT: hybrid vs affine-only: hybrid_tighter=%ld tied=%ld affine_tighter=%ld\n",
        hybridBeatsAffine, tiedHA, affineBeatsHybrid);
    return 0;
}
