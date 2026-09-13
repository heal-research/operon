// Throwaway pilot: does per-node collapse (HybridEvaluator) actually
// improve the ROOT bound over today's root-only `combined` mode, on real
// trees? Not permanent code.
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/parser/infix.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/hybrid_evaluator.hpp"

using namespace Operon;

struct Result { double affineOnly=0, intervalOnly=0, combined=0, hybrid=0; bool ok=false; };

auto RunOne(std::string const& infix, Dataset const& ds,
            AffineEvaluator::DomainMap const& adom, IntervalEvaluator::DomainMap const& idom,
            std::size_t& collapses, std::size_t& internalNodes) -> Result {
    Result res;
    Tree tree;
    try { tree = InfixParser::Parse(infix, ds); }
    catch (...) { return res; }
    auto coeff = tree.GetCoefficients();

    double aw, iw, hw;
    try {
        AffineEvaluator ae(&tree, adom);
        auto a = ae.Evaluate(coeff).to_interval();
        if (!std::isfinite(a.inf()) || !std::isfinite(a.sup())) { return res; }
        aw = static_cast<double>(a.sup() - a.inf());
        res.affineOnly = aw;

        IntervalEvaluator ie(&tree, idom);
        auto iv = ie.Evaluate(coeff);
        if (iv.is_empty() || !std::isfinite(iv.inf()) || !std::isfinite(iv.sup())) { return res; }
        iw = static_cast<double>(iv.sup() - iv.inf());
        res.intervalOnly = iw;

        auto lo = std::max(a.inf(), iv.inf());
        auto hi = std::min(a.sup(), iv.sup());
        res.combined = (lo <= hi) ? static_cast<double>(hi - lo) : std::min(aw, iw);

        HybridEvaluator he(&tree, adom);
        auto h = he.Evaluate(coeff).to_interval();
        if (!std::isfinite(h.inf()) || !std::isfinite(h.sup())) { return res; }
        hw = static_cast<double>(h.sup() - h.inf());
        res.hybrid = hw;
        collapses += he.CollapseCount();
        internalNodes += he.InternalNodeCount();
    } catch (std::exception const& e) {
        return res;
    }
    res.ok = true;
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
    int n = 0, ok = 0;
    long hybridBeatsCombined = 0, combinedBeatsHybrid = 0, tiedHC = 0;
    long hybridBeatsAffine = 0, affineBeatsHybrid = 0, tiedHA = 0;
    std::size_t collapses = 0, internalNodes = 0;
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
        if (s.empty()) { continue; }
        ++n;
        auto r = RunOne(s, ds, adom, idom, collapses, internalNodes);
        if (!r.ok) { continue; }
        ++ok;
        double reltol = 1e-6 * std::max({1.0, r.hybrid, r.combined, r.affineOnly});
        if (std::fabs(r.hybrid - r.combined) <= reltol) { ++tiedHC; }
        else if (r.hybrid < r.combined) { ++hybridBeatsCombined; }
        else { ++combinedBeatsHybrid; }

        if (std::fabs(r.hybrid - r.affineOnly) <= reltol) { ++tiedHA; }
        else if (r.hybrid < r.affineOnly) { ++hybridBeatsAffine; }
        else { ++affineBeatsHybrid; }
    }
    std::fclose(f);

    std::printf("models=%d ok=%d\n", n, ok);
    std::printf("collapses=%zu / internal_nodes=%zu (%.1f%%)\n", collapses, internalNodes,
        internalNodes ? 100.0 * static_cast<double>(collapses) / static_cast<double>(internalNodes) : 0.0);
    std::printf("ROOT: hybrid vs combined   : hybrid_tighter=%ld tied=%ld combined_tighter=%ld\n",
        hybridBeatsCombined, tiedHC, combinedBeatsHybrid);
    std::printf("ROOT: hybrid vs affine-only: hybrid_tighter=%ld tied=%ld affine_tighter=%ld\n",
        hybridBeatsAffine, tiedHA, affineBeatsHybrid);
    return 0;
}
