// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Hidden comparison + benchmark harness of the forward-mode interval-jet first-derivative bounds against the
// production symbolic path (BuildVariableGradientDag + IntervalEvaluator); see interval_jet.cpp for permanent tests.
//
// Reproduce the recorded Release result with:
//   nix develop -c cmake -S . -B build-shape-jetbench -Doperon_DEVELOPER_MODE=ON -DBUILD_TESTING=ON
//   -DCMAKE_BUILD_TYPE=Release nix develop -c cmake --build build-shape-jetbench --target operon_interval_jet_compare
//   -j2 nix develop -c ./build-shape-jetbench/test/operon_interval_jet_compare "[.interval-jet-bench]"
//
// Workload: fixed seed 20260612, 300 balanced trees per (length, directions) cell, 8 timed repetitions after one
// warm-up. Lengths are 10/20/30 and directions are 1..4. The corpus uses the supported production primitive pool.
// The 2026-10-13 Release run recorded first-order Jet speedups of 2.41x--3.86x and five allocations/evaluation
// versus 9.0--9.3 for the dag. Per-cell median/p95 raw timings are printed by the executable; CPU scaling can affect
// them.
//
//   operon_test "[.interval-jet-compare]"   reproducible random corpus: bound-quality statistics, pointwise
//                                           containment of both bounds, feasibility-decision effect
//   operon_test "[.interval-jet-bench]"     time per tree (median / p95) of both paths for K = 1..4
//   operon_test "[.interval-jet-scaling]"   time per tree vs the number K of constrained variables (1..32) and the
//                                           tree length, with the crossover where the symbolic dag wins
//                                           (env: OPERON_JET_SCALING_TREES, OPERON_JET_SCALING_DOMAINS=1|2|3)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "interval_jet_support.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/interval_jet_evaluator.hpp"
#include "operon/interpreter/interval_second_order_jet_evaluator.hpp"
#include "operon/operators/creator.hpp"
#if defined(__unix__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifdef OPERON_JET_COUNT_ALLOCATIONS
namespace {
std::atomic<std::size_t> gAllocations { 0 };
} // namespace

void* operator new(std::size_t n)
{
    gAllocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n == 0 ? 1 : n)) {
        return p;
    }
    throw std::bad_alloc {};
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#endif

namespace Operon::Test {
namespace {
    using namespace JetSupport;
    using Status = Operon::IntervalJetStatus;
    using JE = Operon::IntervalJetEvaluator<S>;
    using IE = Operon::IntervalEvaluator<S>;

    constexpr std::uint64_t CorpusSeed = 20260612;
    constexpr std::size_t CorpusSize = 6000;
    constexpr std::size_t SamplesPerTree = 16;
    constexpr auto NoGrad = std::numeric_limits<std::size_t>::max();

    auto AllocationCount() -> std::size_t
    {
#ifdef OPERON_JET_COUNT_ALLOCATIONS
        return gAllocations.load(std::memory_order_relaxed);
#else
        return 0;
#endif
    }

    // ---------------------------------------------------------------- corpus

    struct Item {
        Operon::Tree Tree;
        std::vector<Operon::Hash> Vars; // constrained variables, in direction order
        DomainMap Domains; // box over every dataset variable
        std::vector<S> Coeff;
        bool Unsupported {}; // contains an op the symbolic dag cannot differentiate
    };

    auto PoolConfig(bool withUnsupported) -> PrimitiveSetConfig
    {
        auto cfg = PrimitiveSet::Arithmetic | BuiltinOp::Pow | BuiltinOp::Exp | BuiltinOp::Log | BuiltinOp::Sin
            | BuiltinOp::Cos | BuiltinOp::Square | BuiltinOp::Tan | BuiltinOp::Tanh | BuiltinOp::Sqrt | BuiltinOp::Cbrt
            | BuiltinOp::Asin | BuiltinOp::Acos | BuiltinOp::Atan | BuiltinOp::Sinh | BuiltinOp::Cosh | BuiltinOp::Log1p
            | BuiltinOp::Logabs;
        if (withUnsupported) {
            cfg |= BuiltinOp::Abs;
            cfg |= BuiltinOp::Sqrtabs;
            cfg |= BuiltinOp::Floor;
            cfg |= BuiltinOp::Ceil;
            cfg |= BuiltinOp::Fmin;
            cfg |= BuiltinOp::Fmax;
            cfg |= BuiltinOp::Aq;
            cfg |= BuiltinOp::Powabs;
        }
        return cfg;
    }

    auto MakePset(bool withUnsupported, std::size_t arithmeticFrequency, std::size_t variableFrequency) -> PrimitiveSet
    {
        PrimitiveSet pset { PoolConfig(withUnsupported) };
        for (auto op : { BuiltinOp::Add, BuiltinOp::Sub, BuiltinOp::Mul, BuiltinOp::Div }) {
            pset.SetFrequency(Operon::Hash(op), arithmeticFrequency);
        }
        pset.SetMinMaxArity(Operon::Hash(BuiltinOp::Add), 2, 4);
        pset.SetMinMaxArity(Operon::Hash(BuiltinOp::Mul), 2, 4);
        pset.SetMinMaxArity(Operon::Hash(BuiltinOp::Sub), 1, 2);
        pset.SetMinMaxArity(Operon::Hash(BuiltinOp::Div), 1, 2);
        pset.SetFrequency(Node(NodeType::Variable).HashValue, variableFrequency);
        return pset;
    }

    auto HasUnsupportedOp(Operon::Tree const& tree) -> bool
    {
        return std::ranges::any_of(tree.Nodes(), [](Node const& n) {
            return n.IsOp<BuiltinOp::Abs, BuiltinOp::Sqrtabs, BuiltinOp::Floor, BuiltinOp::Ceil, BuiltinOp::Fmin,
                BuiltinOp::Fmax, BuiltinOp::Aq, BuiltinOp::Powabs>();
        });
    }

    // Box families: narrow/wide positive, touching 0 from either side, straddling 0, negative, point, unit.
    // `benign` restricts the mix to boxes bounded away from zero (narrow/wide positive, positive point).
    auto RandomBox(std::mt19937_64& rng, bool benign) -> std::pair<S, S>
    {
        auto const u = [&](double a, double b) { return std::uniform_real_distribution<double> { a, b }(rng); };
        double lo = 0;
        double hi = 0;
        constexpr std::array<int, 4> benignKinds { 0, 0, 2, 8 };
        auto kind = std::uniform_int_distribution<int> { 0, 9 }(rng);
        if (benign) {
            kind = benignKinds[std::uniform_int_distribution<std::size_t> { 0, benignKinds.size() - 1 }(rng)];
        }
        switch (kind) {
        case 0:
        case 1: // narrow positive
            lo = u(0.1, 3.0);
            hi = lo + u(0.01, 0.5);
            break;
        case 2: // wide positive
            lo = u(0.01, 1.0);
            hi = lo + u(1.0, 6.0);
            break;
        case 3: // touching zero from above
            lo = 0.0;
            hi = u(0.05, 3.0);
            break;
        case 4: // touching zero from below
            hi = 0.0;
            lo = -u(0.05, 3.0);
            break;
        case 5:
        case 6: // straddling zero
            lo = -u(0.05, 3.0);
            hi = u(0.05, 3.0);
            break;
        case 7: // negative
            hi = -u(0.1, 3.0);
            lo = hi - u(0.01, 2.0);
            break;
        case 8: // point
            lo = benign ? u(0.2, 3.0) : u(-3.0, 3.0);
            hi = lo;
            break;
        default: // unit box
            lo = -1.0;
            hi = 1.0;
            break;
        }
        return { static_cast<S>(lo), std::max(static_cast<S>(lo), static_cast<S>(hi)) };
    }

    auto DistinctVariables(Operon::Tree const& tree) -> std::vector<Operon::Hash>
    {
        std::vector<Operon::Hash> out;
        for (auto const& n : tree.Nodes()) {
            if (n.IsVariable() && std::ranges::find(out, n.HashValue) == out.end()) {
                out.push_back(n.HashValue);
            }
        }
        return out;
    }

    auto MakeCorpus(Operon::Dataset const& ds, std::uint64_t seed, std::size_t count, bool benign) -> std::vector<Item>
    {
        auto const supported = MakePset(false, 6, 3);
        auto const mixed = MakePset(true, 6, 3);
        std::vector<Operon::Hash> allVars;
        for (std::size_t i = 1; i <= 5; ++i) {
            allVars.push_back(VarHash(ds, i));
        }

        Operon::RandomGenerator treeRng { seed };
        std::mt19937_64 rng { seed ^ 0x9E3779B97F4A7C15ULL };
        std::vector<Item> corpus;
        corpus.reserve(count);
        while (corpus.size() < count) {
            auto const useMixed = std::uniform_real_distribution<double> { 0.0, 1.0 }(rng) < 0.15;
            auto const nvars = std::uniform_int_distribution<std::size_t> { 2, 5 }(rng);
            std::vector<Operon::Hash> vars(allVars.begin(), allVars.begin() + static_cast<std::ptrdiff_t>(nvars));
            auto const length = std::uniform_int_distribution<std::size_t> { 5, 40 }(rng);
            auto tree = useMixed ? BalancedTreeCreator { &mixed, vars, 0.0, 40 }(treeRng, length, 0, 40)
                                 : BalancedTreeCreator { &supported, vars, 0.0, 40 }(treeRng, length, 0, 40);

            auto present = DistinctVariables(tree);
            if (present.empty()) {
                continue; // constant-only tree
            }
            std::ranges::shuffle(present, rng);
            present.resize(
                std::uniform_int_distribution<std::size_t> { 1, std::min<std::size_t>(4, present.size()) }(rng));

            Item item;
            item.Vars = present;
            item.Coeff = tree.GetCoefficients();
            auto const integerMode = std::uniform_real_distribution<double> { 0.0, 1.0 }(rng) < 0.25;
            for (auto& c : item.Coeff) {
                auto v = std::uniform_real_distribution<double> { -2.0, 2.0 }(rng);
                if (integerMode) {
                    v = std::round(v);
                    v = v == 0.0 ? 1.0 : v;
                }
                c = static_cast<S>(v);
            }
            for (auto const h : vars) {
                item.Domains.insert_or_assign(h, RandomBox(rng, benign));
            }
            item.Unsupported = HasUnsupportedOp(tree);
            item.Tree = std::move(tree);
            corpus.push_back(std::move(item));
        }
        return corpus;
    }

    // -------------------------------------------------------------- bounds

    enum class Cls : std::uint8_t { Finite, Unbounded, Failed, Uncertain };
    constexpr std::array<char const*, 4> ClsName { "finite", "unbounded", "failed", "uncertain" };

    auto Classify(Interval const& b) -> Cls
    {
        if (b.is_empty()) {
            return Cls::Failed;
        }
        return b.is_finite() ? Cls::Finite : Cls::Unbounded;
    }

    struct Dir {
        Cls Class { Cls::Failed };
        Interval B {};
    };

    struct PathResult {
        Interval Value {};
        std::vector<Dir> Dirs;
        bool Error {};
    };

    // Production sequence of MeasureConstraints' fused interval-only path.
    auto RunDag(Item const& item) -> PathResult
    {
        PathResult out;
        out.Dirs.resize(item.Vars.size());
        auto const dag = BuildVariableGradientDag(item.Tree, item.Coeff, item.Vars);
        IE ie(&item.Tree, item.Domains);
        ie.SetNodes(dag.Nodes);
        auto const ok = ie.EvaluateAll(item.Coeff, [&](Operon::Span<Interval const> v) {
            out.Value = v[dag.OriginalSize - 1];
            for (std::size_t k = 0; k < item.Vars.size(); ++k) {
                auto const at = std::ranges::find(dag.Variables, item.Vars[k]);
                if (at == dag.Variables.end()) {
                    out.Dirs[k] = { Cls::Finite, Interval::zero() };
                    continue;
                }
                auto const idx = static_cast<std::size_t>(at - dag.Variables.begin());
                if (!dag.Certain[idx]) {
                    out.Dirs[k] = { Cls::Uncertain, Interval::infinite() };
                } else if (dag.Roots[idx] == NoGrad) {
                    out.Dirs[k] = { Cls::Finite, Interval::zero() };
                } else {
                    auto const b = v[dag.Roots[idx]];
                    out.Dirs[k] = { Classify(b), b };
                }
            }
        });
        out.Error = !ok;
        return out;
    }

    // `relaxed`: a direction whose derivative enclosure is non-empty is reported by that enclosure even when the
    // identity value is empty (the symbolic dag does not look at the identity either).
    auto RunJet(Item const& item, bool relaxed = false) -> PathResult
    {
        PathResult out;
        out.Dirs.resize(item.Vars.size());
        JE je(&item.Tree, item.Domains, item.Vars);
        auto const r = je(item.Coeff);
        if (!r) {
            out.Error = true;
            return out;
        }
        out.Value = r->Value;
        for (std::size_t k = 0; k < item.Vars.size(); ++k) {
            switch (r->Status[k]) {
            case Status::Certain:
                out.Dirs[k] = { Classify(r->Derivative[k]), r->Derivative[k] };
                break;
            case Status::Uncertain:
                out.Dirs[k] = { Cls::Uncertain, Interval::infinite() };
                break;
            case Status::Failed:
                if (relaxed && !r->Derivative[k].is_empty()) {
                    out.Dirs[k] = { Classify(r->Derivative[k]), r->Derivative[k] };
                } else {
                    out.Dirs[k] = { Cls::Failed, r->Derivative[k] };
                }
                break;
            }
        }
        return out;
    }

    // ----------------------------------------------------------- statistics

    auto Tol(double a, double b) -> double { return 1e-4 * std::max({ 1.0, std::abs(a), std::abs(b) }); }

    auto SameEnd(double a, double b) -> bool
    {
        return (std::isinf(a) || std::isinf(b)) ? a == b : std::abs(a - b) <= Tol(a, b);
    }

    // a contains b (up to the comparison tolerance)
    auto ContainsTol(Interval const& a, Interval const& b) -> bool
    {
        auto const lo = [&](double x, double y) { return std::isinf(x) ? x <= y : x <= y + Tol(x, y); };
        auto const hi = [&](double x, double y) { return std::isinf(x) ? x >= y : x >= y - Tol(x, y); };
        return lo(a.inf(), b.inf()) && hi(a.sup(), b.sup());
    }

    auto Width(Interval const& b) -> double { return static_cast<double>(b.sup()) - static_cast<double>(b.inf()); }

    struct Relation {
        std::size_t Equal {};
        std::size_t JetTighter {};
        std::size_t JetLooser {};
        std::size_t Incomparable {};
        [[nodiscard]] auto Total() const -> std::size_t { return Equal + JetTighter + JetLooser + Incomparable; }

        void Add(Interval const& jet, Interval const& dag)
        {
            if (SameEnd(jet.inf(), dag.inf()) && SameEnd(jet.sup(), dag.sup())) {
                ++Equal;
            } else if (ContainsTol(dag, jet)) {
                ++JetTighter;
            } else if (ContainsTol(jet, dag)) {
                ++JetLooser;
            } else {
                ++Incomparable;
            }
        }
    };

    auto Quantile(std::vector<double> v, double q) -> double
    {
        if (v.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::ranges::sort(v);
        return v[static_cast<std::size_t>(q * static_cast<double>(v.size() - 1))];
    }

    // Sign-constraint feasibility, as MeasureConstraints decides it: both endpoints finite and the sign holds.
    auto Certifies(Interval const& b, int sign) -> bool
    {
        return b.is_finite() && (sign > 0 ? b.inf() >= S { 0 } : b.sup() <= S { 0 });
    }

    struct Feasibility {
        std::size_t Both {}, DagOnly {}, JetOnly {}, Neither {};
        std::size_t Intersection {}; // jet ∩ dag (or the single certain bound) certifies
        std::size_t IntersectionOnly {}; // ... and neither alone does
        std::size_t EmptyIntersections {}; // two bounds with a gap larger than 1e-4 (relative): soundness alarm
        std::size_t UlpGaps {}; // two disjoint bounds whose gap is rounding-level (pappus float outward rounding)
        std::size_t DagOnlyJetValueEmpty {}; // dag-only certifications where the jet's identity value is empty
        std::vector<std::string> Examples;

        template <typename Where>
        void Add(PathResult const& jet, PathResult const& dag, std::size_t k, int sign, Where&& where)
        {
            auto const jc = jet.Dirs[k].Class == Cls::Finite || jet.Dirs[k].Class == Cls::Unbounded;
            auto const dc = dag.Dirs[k].Class == Cls::Finite || dag.Dirs[k].Class == Cls::Unbounded;
            auto const j = jc && Certifies(jet.Dirs[k].B, sign);
            auto const d = dc && Certifies(dag.Dirs[k].B, sign);
            ++(j && d ? Both : (d ? DagOnly : (j ? JetOnly : Neither)));
            DagOnlyJetValueEmpty += (d && !j && jet.Value.is_empty()) ? 1 : 0;
            bool inter = false;
            if (jc && dc) {
                auto const i = jet.Dirs[k].B & dag.Dirs[k].B;
                auto const& jb = jet.Dirs[k].B;
                auto const& db = dag.Dirs[k].B;
                auto const gap
                    = std::max(static_cast<double>(jb.inf()) - db.sup(), static_cast<double>(db.inf()) - jb.sup());
                auto const scale = std::max(
                    { 1.0, std::abs(static_cast<double>(jb.inf())), std::abs(static_cast<double>(db.sup())) });
                if (i.is_empty() && gap <= 1e-4 * scale) {
                    ++UlpGaps;
                } else if (i.is_empty()) {
                    ++EmptyIntersections;
                    if (Examples.size() < 4) {
                        Examples.push_back(
                            fmt::format("jet [{}, {}] vs dag [{}, {}] (direction {}) in {}", jet.Dirs[k].B.inf(),
                                jet.Dirs[k].B.sup(), dag.Dirs[k].B.inf(), dag.Dirs[k].B.sup(), k, where()));
                    }
                } else {
                    inter = Certifies(i, sign);
                }
            } else if (jc) {
                inter = j;
            } else if (dc) {
                inter = d;
            }
            Intersection += inter ? 1 : 0;
            IntersectionOnly += (inter && !j && !d) ? 1 : 0;
        }
    };

    auto Describe(Item const& item, Operon::Dataset const& ds) -> std::string
    {
        std::string s
            = fmt::format("{}\n    boxes:", fmt::format("{}", Operon::Fmt::TreeFormatArgs { item.Tree, ds, 7 }));
        for (auto const& [h, b] : item.Domains) {
            s += fmt::format(" {}=[{}, {}]", ds.GetVariable(h)->Name, b.first, b.second);
        }
        s += fmt::format("\n    coefficients (Optimize order): {:.9g}", fmt::join(item.Coeff, ", "));
        return s;
    }

    auto Median(std::vector<double> v) -> double { return Quantile(std::move(v), 0.5); }
} // namespace

namespace {
    void RunComparison(char const* label, Operon::Dataset const& ds, std::vector<Item> const& corpus)
    {

        std::size_t pairs = 0;
        std::size_t jetErrors = 0;
        std::size_t dagErrors = 0;
        std::size_t treesWithUnsupported = 0;
        std::array<std::array<std::size_t, 4>, 4> matrix {}; // [jet][dag]
        Relation relAll;
        Relation relFinite;
        std::vector<double> ratios;
        std::size_t zeroWidthDag = 0;
        std::size_t jetUncertainDagCertain = 0;
        std::size_t dagUncertainJetCertain = 0;
        std::size_t jetFailedIdentityEmpty = 0;
        std::size_t jetFailedDagFinite = 0;
        std::array<Feasibility, 2> feas {}; // [+1, -1]
        std::array<Feasibility, 2> feasRelaxed {}; // jet reported by its derivative alone
        Containment jetValue { 1e-4 };
        Containment jetDeriv { 1e-4 };
        Containment dagValue { 1e-4 };
        Containment dagDeriv { 1e-4 };
        Containment jetDValue { 1e-9 }; // the jet evaluator instantiated for double: formula soundness
        Containment jetDDeriv { 1e-9 };
        std::size_t lengthSum = 0;
        std::array<std::size_t, 5> kHistogram {};
        std::size_t violationReports = 0;
        std::vector<std::string> failureExamples;
        std::vector<std::string> dagFailureExamples;

        std::mt19937_64 sampleRng { CorpusSeed + 1 };
        for (auto const& item : corpus) {
            lengthSum += item.Tree.Length();
            treesWithUnsupported += item.Unsupported ? 1 : 0;
            ++kHistogram[item.Vars.size()];
            auto const dag = RunDag(item);
            auto const jet = RunJet(item);
            auto const jetRelaxed = RunJet(item, /*relaxed=*/true);
            if (dag.Error) {
                ++dagErrors;
            }
            if (jet.Error) {
                ++jetErrors;
            }
            if (dag.Error || jet.Error) {
                continue;
            }

            for (std::size_t k = 0; k < item.Vars.size(); ++k) {
                ++pairs;
                auto const& j = jet.Dirs[k];
                auto const& d = dag.Dirs[k];
                ++matrix[static_cast<std::size_t>(j.Class)][static_cast<std::size_t>(d.Class)];
                jetUncertainDagCertain += (j.Class == Cls::Uncertain && d.Class != Cls::Uncertain) ? 1 : 0;
                dagUncertainJetCertain += (d.Class == Cls::Uncertain && j.Class != Cls::Uncertain) ? 1 : 0;
                if (j.Class == Cls::Failed && (d.Class == Cls::Finite || d.Class == Cls::Unbounded)) {
                    ++jetFailedDagFinite;
                    jetFailedIdentityEmpty += jet.Value.is_empty() ? 1 : 0;
                    if (!jet.Value.is_empty() && failureExamples.size() < 4) {
                        failureExamples.push_back(
                            fmt::format("jet derivative empty while dag has [{}, {}] (direction {}) in {}", d.B.inf(),
                                d.B.sup(), k, Describe(item, ds)));
                    }
                }
                if (d.Class == Cls::Failed && (j.Class == Cls::Finite || j.Class == Cls::Unbounded)
                    && dagFailureExamples.size() < 4) {
                    dagFailureExamples.push_back(
                        fmt::format("dag derivative empty (identity {}) while jet has [{}, {}] (direction {}) in {}",
                            dag.Value.is_empty() ? "empty" : "non-empty", j.B.inf(), j.B.sup(), k, Describe(item, ds)));
                }
                auto const comparable = [](Cls c) { return c == Cls::Finite || c == Cls::Unbounded; };
                if (comparable(j.Class) && comparable(d.Class)) {
                    relAll.Add(j.B, d.B);
                    if (j.Class == Cls::Finite && d.Class == Cls::Finite) {
                        relFinite.Add(j.B, d.B);
                        if (Width(d.B) > 1e-9) {
                            ratios.push_back(Width(j.B) / Width(d.B));
                        } else {
                            ++zeroWidthDag;
                        }
                    }
                }
                auto const where = [&] { return Describe(item, ds); };
                feas[0].Add(jet, dag, k, +1, where);
                feas[1].Add(jet, dag, k, -1, where);
                feasRelaxed[0].Add(jetRelaxed, dag, k, +1, where);
                feasRelaxed[1].Add(jetRelaxed, dag, k, -1, where);
            }

            // pointwise containment of both bounds against sampled true derivatives
            auto const check = [&](PathResult const& r, Containment& value, Containment& deriv, char const* who) {
                BoundSet set;
                if (!r.Value.is_empty()) {
                    set.Value = ToBnd(r.Value);
                }
                for (std::size_t k = 0; k < item.Vars.size(); ++k) {
                    if (r.Dirs[k].Class == Cls::Finite || r.Dirs[k].Class == Cls::Unbounded) {
                        set.Derivative[k] = ToBnd(r.Dirs[k].B);
                    }
                }
                auto const before = value.Violations + deriv.Violations;
                SampleCheck(
                    item.Tree, item.Coeff, item.Vars, item.Domains, set, sampleRng, SamplesPerTree, value, deriv);
                if (value.Violations + deriv.Violations != before && violationReports++ < 20) {
                    fmt::print("!!! SOUNDNESS VIOLATION ({}):\n    {}\n", who, Describe(item, ds));
                    for (auto const& e : value.Examples) {
                        fmt::print("    value: {}\n", e);
                    }
                    for (auto const& e : deriv.Examples) {
                        fmt::print("    derivative: {}\n", e);
                    }
                    value.Examples.clear();
                    deriv.Examples.clear();
                    for (std::size_t k = 0; k < item.Vars.size(); ++k) {
                        fmt::print("    dir {} ({}): bound [{}, {}] class {}\n", k, ds.GetVariable(item.Vars[k])->Name,
                            r.Dirs[k].B.inf(), r.Dirs[k].B.sup(), ClsName[static_cast<std::size_t>(r.Dirs[k].Class)]);
                    }
                }
            };
            check(jet, jetValue, jetDeriv, "JET");
            check(dag, dagValue, dagDeriv, "DAG");
            {
                Operon::IntervalJetEvaluator<double> jd(&item.Tree, item.Domains, item.Vars);
                auto const r = jd(item.Coeff);
                if (r) {
                    auto const before = jetDDeriv.Violations;
                    SampleCheck(item.Tree, item.Coeff, item.Vars, item.Domains, ToBoundSet(*r), sampleRng,
                        SamplesPerTree, jetDValue, jetDDeriv);
                    if (jetDDeriv.Violations != before && violationReports++ < 20) {
                        fmt::print("!!! SOUNDNESS VIOLATION (JET double):\n    {}\n", Describe(item, ds));
                        for (auto const& e : jetDDeriv.Examples) {
                            fmt::print("    derivative: {}\n", e);
                        }
                        jetDDeriv.Examples.clear();
                    }
                    jetDValue.Examples.clear();
                }
            }
        }

        auto const pct = [&](std::size_t n, std::size_t of) {
            return of == 0 ? 0.0 : 100.0 * static_cast<double>(n) / static_cast<double>(of);
        };
        fmt::print("\n==================== interval-jet vs symbolic-dag comparison: {} ====================\n", label);
        fmt::print("scalar: {}, seed {}, corpus {} trees ({} with an unsupported op), mean length {:.1f}\n",
            sizeof(S) == 4 ? "float" : "double", CorpusSeed, corpus.size(), treesWithUnsupported,
            static_cast<double>(lengthSum) / static_cast<double>(corpus.size()));
        fmt::print("constrained-variable count K histogram: K=1:{} K=2:{} K=3:{} K=4:{}\n", kHistogram[1],
            kHistogram[2], kHistogram[3], kHistogram[4]);
        fmt::print("(tree, variable) pairs: {}   jet evaluation errors: {}   dag evaluation errors: {}\n\n", pairs,
            jetErrors, dagErrors);

        fmt::print("class matrix (rows: jet, columns: dag), pair counts\n{:>12}", "");
        for (auto const* name : ClsName) {
            fmt::print("{:>11}", name);
        }
        fmt::print("\n");
        for (std::size_t r = 0; r < 4; ++r) {
            fmt::print("{:>12}", ClsName[r]);
            for (std::size_t c = 0; c < 4; ++c) {
                fmt::print("{:>11}", matrix[r][c]);
            }
            fmt::print("\n");
        }
        fmt::print("jet uncertain but dag certain: {}    dag uncertain but jet certain: {}\n", jetUncertainDagCertain,
            dagUncertainJetCertain);
        fmt::print("jet failed while dag produced a bound: {} (of which the jet's identity value is empty: {})\n",
            jetFailedDagFinite, jetFailedIdentityEmpty);
        for (auto const& e : failureExamples) {
            fmt::print("    {}\n", e);
        }
        for (auto const& e : dagFailureExamples) {
            fmt::print("    {}\n", e);
        }
        fmt::print("\n");

        auto const printRel = [&](char const* title, Relation const& r) {
            fmt::print("{}: {} comparable pairs\n", title, r.Total());
            fmt::print("    equal (rel. tol 1e-4)   {:>7} ({:5.1f}%)\n", r.Equal, pct(r.Equal, r.Total()));
            fmt::print("    jet tighter             {:>7} ({:5.1f}%)\n", r.JetTighter, pct(r.JetTighter, r.Total()));
            fmt::print("    jet looser              {:>7} ({:5.1f}%)\n", r.JetLooser, pct(r.JetLooser, r.Total()));
            fmt::print(
                "    incomparable            {:>7} ({:5.1f}%)\n", r.Incomparable, pct(r.Incomparable, r.Total()));
        };
        printRel("all certain & non-empty (finite or unbounded)", relAll);
        printRel("both finite", relFinite);
        {
            auto const mean = ratios.empty()
                ? 0.0
                : std::accumulate(ratios.begin(), ratios.end(), 0.0) / static_cast<double>(ratios.size());
            auto logSum = 0.0;
            for (auto const r : ratios) {
                logSum += std::log(std::max(r, 1e-12));
            }
            auto const geo = ratios.empty() ? 0.0 : std::exp(logSum / static_cast<double>(ratios.size()));
            fmt::print(
                "width ratio jet/dag over {} finite pairs (dag width > 0; {} dag-zero-width skipped): mean {:.3f}, "
                "geometric mean {:.3f}, median {:.3f}, p10 {:.3f}, p90 {:.3f}\n\n",
                ratios.size(), zeroWidthDag, mean, geo, Median(ratios), Quantile(ratios, 0.1), Quantile(ratios, 0.9));
        }
        fmt::print(
            "finite-vs-unbounded: jet finite & dag unbounded {}, jet unbounded & dag finite {}, both unbounded {}\n\n",
            matrix[static_cast<std::size_t>(Cls::Finite)][static_cast<std::size_t>(Cls::Unbounded)],
            matrix[static_cast<std::size_t>(Cls::Unbounded)][static_cast<std::size_t>(Cls::Finite)],
            matrix[static_cast<std::size_t>(Cls::Unbounded)][static_cast<std::size_t>(Cls::Unbounded)]);

        fmt::print("feasibility under a sign constraint (certified = both endpoints finite and the sign holds)\n");
        fmt::print("{:>22}{:>10}{:>10}{:>10}{:>10}{:>14}{:>14}{:>14}\n", "constraint", "both", "dag only", "jet only",
            "neither", "jet∩dag", "∩ only", "|jet∪dag|");
        Feasibility total;
        for (std::size_t i = 0; i < 2; ++i) {
            auto const& f = feas[i];
            fmt::print("{:>22}{:>10}{:>10}{:>10}{:>10}{:>14}{:>14}{:>14}\n",
                i == 0 ? "increasing (d >= 0)" : "decreasing (d <= 0)", f.Both, f.DagOnly, f.JetOnly, f.Neither,
                f.Intersection, f.IntersectionOnly, f.Both + f.DagOnly + f.JetOnly);
            total.Both += f.Both;
            total.DagOnly += f.DagOnly;
            total.JetOnly += f.JetOnly;
            total.Neither += f.Neither;
            total.Intersection += f.Intersection;
            total.IntersectionOnly += f.IntersectionOnly;
            total.EmptyIntersections += f.EmptyIntersections;
            total.UlpGaps += f.UlpGaps;
            total.DagOnlyJetValueEmpty += f.DagOnlyJetValueEmpty;
        }
        fmt::print("{:>22}{:>10}{:>10}{:>10}{:>10}{:>14}{:>14}{:>14}\n", "total", total.Both, total.DagOnly,
            total.JetOnly, total.Neither, total.Intersection, total.IntersectionOnly,
            total.Both + total.DagOnly + total.JetOnly);
        {
            Feasibility r;
            for (auto const& f : feasRelaxed) {
                r.Both += f.Both;
                r.DagOnly += f.DagOnly;
                r.JetOnly += f.JetOnly;
                r.Neither += f.Neither;
                r.Intersection += f.Intersection;
                r.IntersectionOnly += f.IntersectionOnly;
            }
            fmt::print(
                "{:>22}{:>10}{:>10}{:>10}{:>10}{:>14}{:>14}{:>14}   (jet judged by its derivative alone, identity "
                "emptiness ignored)\n",
                "total, relaxed jet", r.Both, r.DagOnly, r.JetOnly, r.Neither, r.Intersection, r.IntersectionOnly,
                r.Both + r.DagOnly + r.JetOnly);
        }

        fmt::print("dag-only certifications where the jet's identity value is empty (function undefined on the whole "
                   "box): {}\n",
            total.DagOnlyJetValueEmpty);
        fmt::print(
            "disjoint jet/dag bounds (empty intersection): {} with a gap above 1e-4 relative (soundness alarm), {} at "
            "rounding level (gap <= 1e-4 relative)\n\n",
            total.EmptyIntersections, total.UlpGaps);
        for (auto const& f : feas) {
            for (auto const& e : f.Examples) {
                fmt::print("    {}\n", e);
            }
        }

        auto const printContainment = [&](char const* name, Containment const& v, Containment const& d) {
            fmt::print(
                "{:>4}: derivative samples checked {:>8}, violations {:>5} (strict {:>6}), worst excess {:.3g};   "
                "value samples {:>8}, violations {:>5}\n",
                name, d.Checked, d.Violations, d.Strict, d.WorstExcess, v.Checked, v.Violations);
        };
        fmt::print("pointwise containment vs double-precision reference ({} samples per tree)\n", SamplesPerTree);
        printContainment("jet", jetValue, jetDeriv);
        printContainment("dag", dagValue, dagDeriv);
        printContainment("jetD", jetDValue, jetDDeriv);
        fmt::print("(float slack 1e-4 relative: pappus float arithmetic is only outward rounded to a few ulp per op; "
                   "jetD = IntervalJetEvaluator<double>, slack 1e-9)\n");
        fmt::print("=================================================================================\n");

        CHECK(jetErrors == 0);
        CHECK(dagErrors == 0);
        CHECK(jetDeriv.Violations == 0);
        CHECK(dagDeriv.Violations == 0);
        CHECK(jetDDeriv.Violations == 0);
        // identity-value violations belong to the shared IntervalEvaluator/pappus value semantics (identical for both
        // paths); they are reported above but not asserted here.
        CHECK(total.EmptyIntersections == 0);
    }
} // namespace

TEST_CASE("Interval jet vs symbolic dag: bound quality and soundness over a random corpus", "[.interval-jet-compare]")
{
    auto const ds = MakeDataset(5);
    RunComparison("mixed boxes (singularities, zero-touching, straddling, negative, point, unit)", ds,
        MakeCorpus(ds, CorpusSeed, CorpusSize, /*benign=*/false));
    RunComparison(
        "benign boxes (positive, away from zero)", ds, MakeCorpus(ds, CorpusSeed + 7, CorpusSize, /*benign=*/true));
}

// ------------------------------------------------------------------ benchmark

namespace {
    struct Timing {
        double Median {};
        double P95 {};
        double Allocations {};
    };

    auto Summarize(std::vector<double> ns, std::size_t allocations, std::size_t calls) -> Timing
    {
        return { Quantile(ns, 0.5), Quantile(ns, 0.95),
            calls == 0 ? 0.0 : static_cast<double>(allocations) / static_cast<double>(calls) };
    }

    template <typename F> auto Measure(std::vector<Item> const& items, std::size_t reps, F&& f) -> Timing
    {
        std::vector<double> ns;
        ns.reserve(items.size() * reps);
        std::size_t allocations = 0;
        std::size_t calls = 0;
        // warm-up pass: first-touch of thread-local scratch, registries, allocator caches
        for (auto const& item : items) {
            f(item);
        }
        for (std::size_t r = 0; r < reps; ++r) {
            for (auto const& item : items) {
                auto const a0 = AllocationCount();
                auto const t0 = std::chrono::steady_clock::now();
                f(item);
                auto const t1 = std::chrono::steady_clock::now();
                allocations += AllocationCount() - a0;
                ++calls;
                ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
        }
        return Summarize(std::move(ns), allocations, calls);
    }

    volatile double gSink = 0.0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace

TEST_CASE("Interval jet vs symbolic dag: time per tree", "[.interval-jet-bench]")
{
    auto const ds = MakeDataset(5);
    std::vector<Operon::Hash> allVars;
    for (std::size_t i = 1; i <= 5; ++i) {
        allVars.push_back(VarHash(ds, i));
    }
    auto const pset = MakePset(false, 6, 6);
    Operon::RandomGenerator treeRng { CorpusSeed };
    std::mt19937_64 rng { CorpusSeed };
    constexpr std::size_t TreesPerCell = 300;
    constexpr std::size_t Reps = 8;

#ifdef NDEBUG
    constexpr auto BuildMode = "Release (-O3 -DNDEBUG)";
#else
    constexpr auto BuildMode = "NOT optimized / assertions on -- numbers are not representative";
#endif
    fmt::print("\n==================== time per tree, nanoseconds ====================\n");
    fmt::print("build: {}; scalar {}; corpus seed {}; {} trees per cell x {} repetitions; allocations {}\n", BuildMode,
        sizeof(S) == 4 ? "float" : "double", CorpusSeed, TreesPerCell, Reps,
#ifdef OPERON_JET_COUNT_ALLOCATIONS
        "counted (global operator new)"
#else
        "not counted"
#endif
    );
    fmt::print("dag = BuildVariableGradientDag + IntervalEvaluator(SetNodes, EvaluateAll); jet = IntervalJetEvaluator "
               "(construct + evaluate)\n");
    fmt::print(
        "dag-build / dag-eval / jet-eval split the same work with the evaluator already constructed; plain = one "
        "IntervalEvaluator pass over the tree (value only)\n\n");
    fmt::print("{:>4}{:>3}{:>7}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>9}{:>9}\n", "len", "K",
        "avgLen", "dag med", "dag p95", "jet med", "jet p95", "speedup", "dagBuild", "dagEval", "jetEval", "plain",
        "dag alloc", "jet alloc", "");

    for (std::size_t length : { 10UL, 20UL, 30UL }) {
        for (std::size_t K = 1; K <= 4; ++K) {
            std::vector<Item> items;
            std::size_t lengthSum = 0;
            std::size_t attempts = 0;
            while (items.size() < TreesPerCell && attempts++ < 2'000'000) {
                auto tree = BalancedTreeCreator { &pset, allVars, 0.0, 40 }(treeRng, length, 0, 40);
                auto present = DistinctVariables(tree);
                if (present.size() < K) {
                    continue;
                }
                Item item;
                item.Vars.assign(present.begin(), present.begin() + static_cast<std::ptrdiff_t>(K));
                item.Coeff = tree.GetCoefficients();
                for (auto& c : item.Coeff) {
                    c = static_cast<S>(std::uniform_real_distribution<double> { 0.5, 2.0 }(rng));
                }
                for (auto const h : allVars) {
                    auto const lo = std::uniform_real_distribution<double> { 0.5, 2.0 }(rng);
                    item.Domains.insert_or_assign(h,
                        std::pair { static_cast<S>(lo),
                            static_cast<S>(lo + std::uniform_real_distribution<double> { 0.05, 1.0 }(rng)) });
                }
                lengthSum += tree.Length();
                item.Tree = std::move(tree);
                items.push_back(std::move(item));
            }
            REQUIRE(items.size() == TreesPerCell);

            auto const dagFull = Measure(items, Reps, [&](Item const& it) {
                auto const dag = BuildVariableGradientDag(it.Tree, it.Coeff, it.Vars);
                IE ie(&it.Tree, it.Domains);
                ie.SetNodes(dag.Nodes);
                auto sink = 0.0;
                static_cast<void>(ie.EvaluateAll(it.Coeff, [&](Operon::Span<Interval const> v) {
                    sink += static_cast<double>(v[dag.OriginalSize - 1].sup());
                    for (auto const root : dag.Roots) {
                        if (root != NoGrad) {
                            sink += static_cast<double>(v[root].sup());
                        }
                    }
                }));
                gSink = gSink + sink;
            });
            auto const jetFull = Measure(items, Reps, [&](Item const& it) {
                JE je(&it.Tree, it.Domains, it.Vars);
                auto const r = je(it.Coeff);
                if (r) {
                    auto sink = static_cast<double>(r->Value.sup());
                    for (std::size_t k = 0; k < K; ++k) {
                        sink += static_cast<double>(r->Derivative[k].sup());
                    }
                    gSink = gSink + sink;
                }
            });

            // split timings with prebuilt state
            std::vector<VariableGradientDag> dags;
            std::vector<IE> ies;
            std::vector<IE> plains; // untouched by SetNodes: the original tree
            std::vector<JE> jes;
            dags.reserve(items.size());
            ies.reserve(items.size());
            plains.reserve(items.size());
            jes.reserve(items.size());
            for (auto const& it : items) {
                dags.push_back(BuildVariableGradientDag(it.Tree, it.Coeff, it.Vars));
                ies.emplace_back(&it.Tree, it.Domains);
                ies.back().SetNodes(dags.back().Nodes);
                plains.emplace_back(&it.Tree, it.Domains);
                jes.emplace_back(&it.Tree, it.Domains, it.Vars);
            }
            std::vector<Item> const& refItems = items;
            auto const indexOf = [&](Item const& it) { return static_cast<std::size_t>(&it - refItems.data()); };
            auto const dagBuild = Measure(items, Reps, [&](Item const& it) {
                auto const dag = BuildVariableGradientDag(it.Tree, it.Coeff, it.Vars);
                gSink = gSink + static_cast<double>(dag.Nodes.size());
            });
            auto const dagEval = Measure(items, Reps, [&](Item const& it) {
                auto const i = indexOf(it);
                auto sink = 0.0;
                static_cast<void>(ies[i].EvaluateAll(it.Coeff, [&](Operon::Span<Interval const> v) {
                    sink += static_cast<double>(v[dags[i].OriginalSize - 1].sup());
                    for (auto const root : dags[i].Roots) {
                        if (root != NoGrad) {
                            sink += static_cast<double>(v[root].sup());
                        }
                    }
                }));
                gSink = gSink + sink;
            });
            auto const jetEval = Measure(items, Reps, [&](Item const& it) {
                auto const r = jes[indexOf(it)](it.Coeff);
                if (r) {
                    gSink = gSink + static_cast<double>(r->Value.sup()) + static_cast<double>(r->Derivative[0].sup());
                }
            });
            auto const plain = Measure(items, Reps, [&](Item const& it) {
                auto const r = plains[indexOf(it)](it.Coeff);
                if (r) {
                    gSink = gSink + static_cast<double>(r->sup());
                }
            });
            fmt::print("{:>4}{:>3}{:>7.1f}{:>10.0f}{:>10.0f}{:>10.0f}{:>10.0f}{:>9.2f}x{:>10.0f}{:>10.0f}{:>10.0f}{:>"
                       "10.0f}{:>10.1f}{:>9.1f}\n",
                length, K, static_cast<double>(lengthSum) / static_cast<double>(items.size()), dagFull.Median,
                dagFull.P95, jetFull.Median, jetFull.P95, dagFull.Median / jetFull.Median, dagBuild.Median,
                dagEval.Median, jetEval.Median, plain.Median, dagFull.Allocations, jetFull.Allocations);
        }
    }
    fmt::print("=====================================================================\n");
}

} // namespace Operon::Test

namespace Operon::Test {
TEST_CASE(
    "pappus float elementary functions: point-interval enclosure of the true value", "[.interval-jet-pappus-rounding]")
{
    // How often does a point interval of a pappus<float> elementary function miss the true value? (soundness of the
    // float outward rounding itself, independent of the jet code)
    std::mt19937_64 rng { 1 };
    auto const probe = [&](char const* name, auto fn, auto ref, double lo, double hi) {
        std::size_t bad = 0;
        std::size_t badNeg = 0;
        std::size_t n = 20000;
        double worst = 0;
        for (std::size_t i = 0; i < n; ++i) {
            auto const x = static_cast<S>(std::uniform_real_distribution<double> { lo, hi }(rng));
            auto const iv = fn(Interval { x });
            auto const r = ref(static_cast<double>(x));
            if (!std::isfinite(r)) {
                continue;
            }
            auto const ex = std::max(static_cast<double>(iv.inf()) - r, r - static_cast<double>(iv.sup()));
            if (ex > 0) {
                ++bad;
                badNeg += x < 0 ? 1 : 0;
                worst = std::max(worst, ex / std::max(1.0, std::abs(r)));
            }
        }
        fmt::print("{:>8}: {:>6} / {} point-interval enclosures miss the true value (negative x: {}), worst rel excess "
                   "{:.3g}\n",
            name, bad, n, badNeg, worst);
    };
    auto const probe2 = [&](char const* name, auto fn, auto ref) {
        std::size_t bad = 0;
        constexpr std::size_t n = 20000;
        for (std::size_t i = 0; i < n; ++i) {
            auto const a = static_cast<S>(std::uniform_real_distribution<double> { 0.1, 10.0 }(rng));
            auto const b = static_cast<S>(std::uniform_real_distribution<double> { 0.1, 10.0 }(rng));
            auto const interval = fn(Interval { a }, Interval { b });
            auto const value = ref(static_cast<double>(a), static_cast<double>(b));
            bad += static_cast<std::size_t>(value < interval.inf() || value > interval.sup());
        }
        fmt::print("{:>8}: {:>6} / {} binary point-interval enclosures miss the true value\n", name, bad, n);
    };
    probe("cbrt", [](Interval const& v) { return v.cbrt(); }, [](double x) { return std::cbrt(x); }, -10, 10);
    probe("tan", [](Interval const& v) { return v.tan(); }, [](double x) { return std::tan(x); }, -1.5, 1.5);
    probe("sin", [](Interval const& v) { return v.sin(); }, [](double x) { return std::sin(x); }, -3, 3);
    probe("cos", [](Interval const& v) { return v.cos(); }, [](double x) { return std::cos(x); }, -3, 3);
    probe("exp", [](Interval const& v) { return v.exp(); }, [](double x) { return std::exp(x); }, -5, 5);
    probe("log", [](Interval const& v) { return v.log(); }, [](double x) { return std::log(x); }, 0.001, 10);
    probe("log1p", [](Interval const& v) { return v.log1p(); }, [](double x) { return std::log1p(x); }, -0.9, 10);
    probe("sqrt", [](Interval const& v) { return v.sqrt(); }, [](double x) { return std::sqrt(x); }, 0, 10);
    probe("asin", [](Interval const& v) { return v.asin(); }, [](double x) { return std::asin(x); }, -1, 1);
    probe("acos", [](Interval const& v) { return v.acos(); }, [](double x) { return std::acos(x); }, -1, 1);
    probe("atan", [](Interval const& v) { return v.atan(); }, [](double x) { return std::atan(x); }, -10, 10);
    probe("sinh", [](Interval const& v) { return v.sinh(); }, [](double x) { return std::sinh(x); }, -5, 5);
    probe("cosh", [](Interval const& v) { return v.cosh(); }, [](double x) { return std::cosh(x); }, -5, 5);
    probe2(
        "a + b", [](Interval const& a, Interval const& b) { return a + b; }, [](double a, double b) { return a + b; });
    probe2(
        "a * b", [](Interval const& a, Interval const& b) { return a * b; }, [](double a, double b) { return a * b; });
    probe2(
        "a / b", [](Interval const& a, Interval const& b) { return a / b; }, [](double a, double b) { return a / b; });
}
} // namespace Operon::Test

// ------------------------------------------------------- K-scaling benchmark

namespace Operon::Test {
namespace {
    // Direction counts the wide jet path supports (see detail::IntervalJetWideDirections).
    constexpr std::array<std::size_t, 10> ScalingK { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 };

    template <std::size_t... I>
    constexpr auto JetNodeBytesImpl(std::size_t k, std::index_sequence<I...> /*unused*/) -> std::size_t
    {
        std::size_t bytes = 0;
        ((k == ScalingK[I] ? (bytes = sizeof(Operon::detail::IntervalJetValue<S, static_cast<int>(ScalingK[I])>))
                           : bytes),
            ...);
        return bytes;
    }

    // Size of one post-order jet value (value + K tangents + masks): what the evaluator stores per tree node.
    constexpr auto JetNodeBytes(std::size_t k) -> std::size_t
    {
        return JetNodeBytesImpl(k, std::make_index_sequence<ScalingK.size()> {});
    }

    template <typename B> auto BoundsSink(B const& b) -> double
    {
        auto sink = static_cast<double>(b.Value.sup());
        for (std::size_t k = 0; k < b.Directions; ++k) {
            sink += static_cast<double>(b.Derivative[k].sup());
        }
        return sink;
    }

    // Construct + evaluate the jet over `vars` (1..4: regular entry point; beyond: EvaluateWide).
    auto JetRun(Item const& it, Operon::Span<Operon::Hash const> vars) -> double
    {
        JE je(&it.Tree, it.Domains, vars);
        if (vars.size() <= 4) {
            auto const r = je(it.Coeff);
            return r ? BoundsSink(*r) : 0.0;
        }
        auto const r = je.EvaluateWide(it.Coeff);
        return r ? BoundsSink(*r) : 0.0;
    }

    // The K variables evaluated as ceil(K / group) independent jets of at most `group` directions.
    auto GroupedJetRun(Item const& it, std::size_t group) -> double
    {
        auto sink = 0.0;
        for (std::size_t begin = 0; begin < it.Vars.size(); begin += group) {
            auto const n = std::min(group, it.Vars.size() - begin);
            sink += JetRun(it, Operon::Span<Operon::Hash const> { it.Vars.data() + begin, n });
        }
        return sink;
    }

    auto DagSink(VariableGradientDag const& dag, Operon::Span<Interval const> v) -> double
    {
        auto sink = static_cast<double>(v[dag.OriginalSize - 1].sup());
        for (auto const root : dag.Roots) {
            if (root != NoGrad) {
                sink += static_cast<double>(v[root].sup());
            }
        }
        return sink;
    }

    struct ScalingRow {
        std::size_t Dataset {};
        bool Mixed {};
        std::size_t Length {};
        std::size_t K {};
        std::size_t Trees {};
        double DagMedian {};
        double JetMedian {};
        double Hybrid4 {}; // jets of <= 4 directions each (NaN when K <= 4)
        double Hybrid8 {}; // jets of <= 8 directions each (NaN when K <= 8)
        double DagNodesPerLength {};
        double DagBuild {}; // BuildVariableGradientDag only
        double DagEval {}; // evaluator pre-built over the DAG
        double JetEval {}; // jet evaluator pre-built
        double Plain {}; // one IntervalEvaluator pass over the original tree
    };

    auto Nan() -> double { return std::numeric_limits<double>::quiet_NaN(); }

    // Pappus asserts (abort) when an interval operation produces inverted bounds, which a few random trees over
    // mixed boxes trigger. Each batch of candidates runs every evaluation path once in a forked child; a batch whose
    // child does not exit cleanly is bisected tree by tree and the aborting trees are excluded from the corpus (and
    // counted). `stage` -1 runs every path; 0 the plain IntervalEvaluator pass, 1 the dag path, 2 the jet path.
    auto SurvivesEvaluation(Operon::Span<Item const> batch, int stage = -1) -> bool
    {
#if defined(__unix__)
        std::fflush(nullptr);
        auto const pid = fork();
        if (pid < 0) {
            return true;
        }
        if (pid == 0) {
            if (std::freopen("/dev/null", "w", stderr) == nullptr
                || std::freopen("/dev/null", "w", stdout) == nullptr) {
                _exit(2);
            }
            for (auto const& it : batch) {
                if (stage < 0 || stage == 0) {
                    IE plain(&it.Tree, it.Domains);
                    static_cast<void>(plain(it.Coeff));
                }
                if (stage < 0 || stage == 1) {
                    static_cast<void>(RunDag(it));
                }
                if (stage < 0 || stage == 2) {
                    static_cast<void>(JetRun(it, it.Vars));
                }
            }
            _exit(0);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#else
        static_cast<void>(batch);
        static_cast<void>(stage);
        return true;
#endif
    }

    auto EnvCount(char const* name, std::size_t fallback) -> std::size_t
    {
        if (auto const* s = std::getenv(name); s != nullptr && *s != '\0') {
            return static_cast<std::size_t>(std::strtoull(s, nullptr, 10));
        }
        return fallback;
    }
} // namespace

TEST_CASE("Interval jet vs symbolic dag: scaling with the number of constrained variables K", "[.interval-jet-scaling]")
{
    constexpr std::array<std::size_t, 3> DatasetVars { 8, 16, 32 };
    constexpr std::array<std::size_t, 6> Lengths { 20, 40, 80, 160, 320, 640 };
    auto const treesPerCell = EnvCount("OPERON_JET_SCALING_TREES", 100);
    auto const domainMode = EnvCount("OPERON_JET_SCALING_DOMAINS", 3); // bit 0: positive boxes, bit 1: mixed boxes
    constexpr std::size_t MinTrees = 30;
    constexpr std::size_t MaxAttempts = 60'000;

#ifdef NDEBUG
    constexpr auto BuildMode = "Release (-O3 -DNDEBUG)";
#else
    constexpr auto BuildMode = "NOT optimized / assertions on -- numbers are not representative";
#endif
    fmt::print("\n==================== K-scaling: time per tree, nanoseconds ====================\n");
    fmt::print("build: {}; scalar {}; seed {}; up to {} trees per cell; allocations {}\n", BuildMode,
        sizeof(S) == 4 ? "float" : "double", CorpusSeed, treesPerCell,
#ifdef OPERON_JET_COUNT_ALLOCATIONS
        "counted (global operator new)"
#else
        "not counted"
#endif
    );
    fmt::print("trees: BalancedTreeCreator of exactly `len` nodes over the D dataset variables (every other node "
               "kind: arithmetic 6, unary/pow 1, constant 1, variable 12); the K constrained variables are a random "
               "subset of the distinct variables present in the tree, so a cell needs a tree with >= K of them\n");
    fmt::print("dag = BuildVariableGradientDag(tree, coeff, K vars) + IntervalEvaluator SetNodes/EvaluateAll; "
               "jet = IntervalJetEvaluator construct + evaluate (K<=4: operator(), else EvaluateWide)\n");
    fmt::print("hyb4/hyb8 = the same K directions as ceil(K/4) / ceil(K/8) independent jets (construct + evaluate); "
               "dagB/dagE/jetE/plain = dag build only / dag eval only / jet eval only (evaluator pre-built) / one "
               "IntervalEvaluator value pass\n");
    fmt::print("dagNodes = DAG node count (x len = expansion over the tree); jetKB = len x sizeof(jet value); dagKB = "
               "dagNodes x sizeof(Interval)\n\n");

    std::vector<ScalingRow> rows;
    std::size_t aborted = 0;
    std::string abortedExample;
    Operon::RandomGenerator treeRng { CorpusSeed };
    std::mt19937_64 rng { CorpusSeed };

    for (auto const dsVars : DatasetVars) {
        auto const ds = MakeDataset(dsVars);
        std::vector<Operon::Hash> allVars;
        for (std::size_t i = 1; i <= dsVars; ++i) {
            allVars.push_back(VarHash(ds, i));
        }
        auto pset = MakePset(false, 6, 12);
        pset.SetFrequency(Node(NodeType::Constant).HashValue, 1);

        for (bool const mixed : { false, true }) {
            if ((domainMode & (mixed ? 2U : 1U)) == 0) {
                continue;
            }
            fmt::print("--- dataset variables D = {}, boxes: {} ---\n", dsVars,
                mixed ? "mixed (singularities, zero-touching, straddling, negative, point, unit)"
                      : "positive (lo in [0.5, 2.5], width in [0.05, 1])");
            fmt::print(
                "{:>4}{:>3}{:>5}{:>6}{:>6}{:>8}{:>6}{:>9}{:>9}{:>9}{:>9}{:>8}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>7}{:>7}"
                "{:>7}{:>7}\n",
                "len", "K", "n", "avgL", "dist", "dagNode", "xLen", "dag med", "dag p95", "jet med", "jet p95",
                "dag/jet", "hyb4", "hyb8", "dagB", "dagE", "jetE", "plain", "alloc.d", "alloc.j", "jetKB", "dagKB");

            for (auto const length : Lengths) {
                BalancedTreeCreator creator { &pset, allVars, 0.0, length + 10 };
                for (auto const K : ScalingK) {
                    if (K > dsVars) {
                        continue;
                    }
                    std::vector<Item> items;
                    std::size_t lengthSum = 0;
                    std::size_t distinctSum = 0;
                    std::size_t attempts = 0;
                    std::vector<std::size_t> candidateDistinct; // realized distinct variables per item
                    while (items.size() < treesPerCell && attempts++ < MaxAttempts) {
                        auto tree = creator(treeRng, length, 0, length);
                        auto present = DistinctVariables(tree);
                        if (present.size() < K) {
                            continue;
                        }
                        std::ranges::shuffle(present, rng);
                        Item item;
                        item.Vars.assign(present.begin(), present.begin() + static_cast<std::ptrdiff_t>(K));
                        item.Coeff = tree.GetCoefficients();
                        for (auto& c : item.Coeff) {
                            c = static_cast<S>(std::uniform_real_distribution<double> { 0.5, 2.0 }(rng));
                        }
                        for (auto const h : allVars) {
                            if (mixed) {
                                item.Domains.insert_or_assign(h, RandomBox(rng, /*benign=*/false));
                            } else {
                                auto const lo = std::uniform_real_distribution<double> { 0.5, 2.5 }(rng);
                                item.Domains.insert_or_assign(h,
                                    std::pair { static_cast<S>(lo),
                                        static_cast<S>(
                                            lo + std::uniform_real_distribution<double> { 0.05, 1.0 }(rng)) });
                            }
                        }
                        item.Tree = std::move(tree);
                        candidateDistinct.push_back(present.size());
                        items.push_back(std::move(item));
                    }
                    if (!SurvivesEvaluation(items)) {
                        // some tree aborts in pappus: find it (them) one by one
                        std::vector<Item> kept;
                        std::vector<std::size_t> keptDistinct;
                        for (std::size_t i = 0; i < items.size(); ++i) {
                            Operon::Span<Item const> const one { &items[i], 1 };
                            if (SurvivesEvaluation(one)) {
                                kept.push_back(std::move(items[i]));
                                keptDistinct.push_back(candidateDistinct[i]);
                                continue;
                            }
                            ++aborted;
                            if (abortedExample.empty()) {
                                abortedExample = fmt::format("D={} boxes={} len={} K={} (aborting paths: plain {}, dag "
                                                             "{}, jet {}): {}",
                                    dsVars, mixed ? "mixed" : "positive", length, K,
                                    SurvivesEvaluation(one, 0) ? "no" : "YES",
                                    SurvivesEvaluation(one, 1) ? "no" : "YES",
                                    SurvivesEvaluation(one, 2) ? "no" : "YES", Describe(items[i], ds));
                            }
                        }
                        items = std::move(kept);
                        candidateDistinct = std::move(keptDistinct);
                    }
                    for (std::size_t i = 0; i < items.size(); ++i) {
                        lengthSum += items[i].Tree.Length();
                        distinctSum += candidateDistinct[i];
                    }
                    if (items.size() < MinTrees) {
                        fmt::print(
                            "{:>4}{:>3}{:>5}   (fewer than {} trees with >= K distinct variables in {} attempts)\n",
                            length, K, items.size(), MinTrees, MaxAttempts);
                        continue;
                    }
                    auto const n = static_cast<double>(items.size());
                    auto const reps = length >= 80 ? std::size_t { 3 } : std::size_t { 6 };

                    auto const dagFull = Measure(items, reps, [&](Item const& it) {
                        auto const dag = BuildVariableGradientDag(it.Tree, it.Coeff, it.Vars);
                        IE ie(&it.Tree, it.Domains);
                        ie.SetNodes(dag.Nodes);
                        auto sink = 0.0;
                        static_cast<void>(
                            ie.EvaluateAll(it.Coeff, [&](Operon::Span<Interval const> v) { sink = DagSink(dag, v); }));
                        gSink = gSink + sink;
                    });
                    auto const jetFull
                        = Measure(items, reps, [&](Item const& it) { gSink = gSink + JetRun(it, it.Vars); });
                    auto const hyb4 = K > 4
                        ? Measure(items, reps, [&](Item const& it) { gSink = gSink + GroupedJetRun(it, 4); })
                        : Timing { Nan(), Nan(), Nan() };
                    auto const hyb8 = K > 8
                        ? Measure(items, reps, [&](Item const& it) { gSink = gSink + GroupedJetRun(it, 8); })
                        : Timing { Nan(), Nan(), Nan() };

                    // split timings with prebuilt state
                    std::vector<VariableGradientDag> dags;
                    std::vector<IE> ies;
                    std::vector<IE> plains; // untouched by SetNodes: the original tree
                    std::vector<JE> jes;
                    dags.reserve(items.size());
                    ies.reserve(items.size());
                    plains.reserve(items.size());
                    jes.reserve(items.size());
                    double dagNodeSum = 0.0;
                    for (auto const& it : items) {
                        dags.push_back(BuildVariableGradientDag(it.Tree, it.Coeff, it.Vars));
                        dagNodeSum += static_cast<double>(dags.back().Nodes.size());
                        ies.emplace_back(&it.Tree, it.Domains);
                        ies.back().SetNodes(dags.back().Nodes);
                        plains.emplace_back(&it.Tree, it.Domains);
                        jes.emplace_back(&it.Tree, it.Domains, it.Vars);
                    }
                    auto const indexOf = [&](Item const& it) { return static_cast<std::size_t>(&it - items.data()); };
                    auto const dagBuild = Measure(items, reps, [&](Item const& it) {
                        auto const dag = BuildVariableGradientDag(it.Tree, it.Coeff, it.Vars);
                        gSink = gSink + static_cast<double>(dag.Nodes.size());
                    });
                    auto const dagEval = Measure(items, reps, [&](Item const& it) {
                        auto const i = indexOf(it);
                        auto sink = 0.0;
                        static_cast<void>(ies[i].EvaluateAll(
                            it.Coeff, [&](Operon::Span<Interval const> v) { sink = DagSink(dags[i], v); }));
                        gSink = gSink + sink;
                    });
                    auto const jetEval = Measure(items, reps, [&](Item const& it) {
                        auto const& je = jes[indexOf(it)];
                        if (K <= 4) {
                            auto const r = je(it.Coeff);
                            gSink = gSink + (r ? BoundsSink(*r) : 0.0);
                        } else {
                            auto const r = je.EvaluateWide(it.Coeff);
                            gSink = gSink + (r ? BoundsSink(*r) : 0.0);
                        }
                    });
                    auto const plain = Measure(items, reps, [&](Item const& it) {
                        auto const r = plains[indexOf(it)](it.Coeff);
                        if (r) {
                            gSink = gSink + static_cast<double>(r->sup());
                        }
                    });

                    auto const avgLen = static_cast<double>(lengthSum) / n;
                    auto const dagNodes = dagNodeSum / n;
                    auto const jetKB = avgLen * static_cast<double>(JetNodeBytes(K)) / 1024.0;
                    auto const dagKB = dagNodes * static_cast<double>(sizeof(Interval)) / 1024.0;
                    fmt::print(
                        "{:>4}{:>3}{:>5}{:>6.0f}{:>6.1f}{:>8.0f}{:>5.1f}x{:>9.0f}{:>9.0f}{:>9.0f}{:>9.0f}{:>7.2f}x"
                        "{:>9.0f}{:>9.0f}{:>9.0f}{:>9.0f}{:>9.0f}{:>9.0f}{:>7.1f}{:>7.1f}{:>7.1f}{:>7.1f}\n",
                        length, K, items.size(), avgLen, static_cast<double>(distinctSum) / n, dagNodes,
                        dagNodes / avgLen, dagFull.Median, dagFull.P95, jetFull.Median, jetFull.P95,
                        dagFull.Median / jetFull.Median, hyb4.Median, hyb8.Median, dagBuild.Median, dagEval.Median,
                        jetEval.Median, plain.Median, dagFull.Allocations, jetFull.Allocations, jetKB, dagKB);
                    rows.push_back({ dsVars, mixed, length, K, items.size(), dagFull.Median, jetFull.Median,
                        hyb4.Median, hyb8.Median, dagNodes / avgLen, dagBuild.Median, dagEval.Median, jetEval.Median,
                        plain.Median });
                }
            }
            fmt::print("\n");
        }
    }

    // ---- crossover summary: first K at which the dag is no slower than the jet
    fmt::print("---- crossover: smallest measured K with dag median <= jet median (none: the jet wins at every "
               "measured K) ----\n");
    fmt::print("{:>4}{:>7}{:>5}{:>12}{:>12}{:>12}{:>12}   speedup dag/jet at K = 1, 2, 3, 4, 8, 16, 32\n", "D", "boxes",
        "len", "jet K*", "hyb4 K*", "hyb8 K*", "best K*");
    for (auto const dsVars : DatasetVars) {
        for (bool const mixed : { false, true }) {
            for (auto const length : Lengths) {
                std::vector<ScalingRow const*> cells;
                for (auto const& row : rows) {
                    if (row.Dataset == dsVars && row.Mixed == mixed && row.Length == length) {
                        cells.push_back(&row);
                    }
                }
                if (cells.empty()) {
                    continue;
                }
                auto const crossover = [&](auto metric) -> std::string {
                    for (auto const* row : cells) {
                        auto const value = metric(*row);
                        if (!std::isnan(value) && row->DagMedian <= value) {
                            return fmt::format("{}", row->K);
                        }
                    }
                    return "none";
                };
                auto const jetOf = [](ScalingRow const& r) { return r.JetMedian; };
                auto const hyb4Of = [](ScalingRow const& r) { return std::isnan(r.Hybrid4) ? r.JetMedian : r.Hybrid4; };
                auto const hyb8Of = [](ScalingRow const& r) { return std::isnan(r.Hybrid8) ? r.JetMedian : r.Hybrid8; };
                auto const bestOf = [&](ScalingRow const& r) { return std::min({ jetOf(r), hyb4Of(r), hyb8Of(r) }); };
                std::string speedups;
                for (auto const k : { 1UL, 2UL, 3UL, 4UL, 8UL, 16UL, 32UL }) {
                    auto const it = std::ranges::find_if(cells, [&](auto const* r) { return r->K == k; });
                    speedups
                        += it == cells.end() ? "      -" : fmt::format("{:>7.2f}", (*it)->DagMedian / (*it)->JetMedian);
                }
                fmt::print("{:>4}{:>7}{:>5}{:>12}{:>12}{:>12}{:>12}   {}\n", dsVars, mixed ? "mixed" : "pos", length,
                    crossover(jetOf), crossover(hyb4Of), crossover(hyb8Of), crossover(bestOf), speedups);
            }
        }
    }

    // ---- eval-only view and multi-box amortization. The production path (MeasureConstraints) builds one dag per
    // (tree, coefficients) and evaluates it once, so `dag/jet` above is the relevant number; if the same coefficients
    // were evaluated over B boxes the dag could be built once and only its evaluation repeated.
    fmt::print("\n---- eval-only (evaluators pre-built): smallest K with dag eval <= jet eval; break-even box count "
               "B* = dagBuild / (jetEval - dagEval) beyond which build-once-evaluate-B-times dag beats B jet "
               "evaluations (jet: the dag eval is slower at that K, so the jet wins for any B) ----\n");
    fmt::print(
        "{:>4}{:>7}{:>5}{:>9}   dagEval/jetEval and B* at K = 1, 2, 4, 8, 16, 32\n", "D", "boxes", "len", "evalK*");
    for (auto const dsVars : DatasetVars) {
        for (bool const mixed : { false, true }) {
            for (auto const length : Lengths) {
                std::vector<ScalingRow const*> cells;
                for (auto const& row : rows) {
                    if (row.Dataset == dsVars && row.Mixed == mixed && row.Length == length) {
                        cells.push_back(&row);
                    }
                }
                if (cells.empty()) {
                    continue;
                }
                std::string evalCross = "none";
                for (auto const* row : cells) {
                    if (row->DagEval <= row->JetEval) {
                        evalCross = fmt::format("{}", row->K);
                        break;
                    }
                }
                std::string detail;
                for (auto const k : { 1UL, 2UL, 4UL, 8UL, 16UL, 32UL }) {
                    auto const it = std::ranges::find_if(cells, [&](auto const* r) { return r->K == k; });
                    if (it == cells.end()) {
                        detail += "              -";
                        continue;
                    }
                    auto const& r = **it;
                    auto const gain = r.JetEval - r.DagEval;
                    detail += gain > 0 ? fmt::format("  {:>4.2f} B*={:>5.1f}", r.DagEval / r.JetEval, r.DagBuild / gain)
                                       : fmt::format("  {:>4.2f} jet      ", r.DagEval / r.JetEval);
                }
                fmt::print("{:>4}{:>7}{:>5}{:>9}  {}\n", dsVars, mixed ? "mixed" : "pos", length, evalCross, detail);
            }
        }
    }
    fmt::print("{} candidate trees aborted inside pappus (libassert failure in interval construction) and were "
               "excluded from every cell{}\n",
        aborted, aborted == 0 ? "" : "; first one:\n" + abortedExample);
    fmt::print("==============================================================================\n");
}

// Second-order bench: the production symbolic route for a SecondDerivative constraint on variable v is
// dag1 = BuildVariableGradientDag(tree) (shared by all constraints of the set), then per constraint: slice the
// first-derivative tree d1, dag2 = BuildVariableGradientDag(d1, v), evaluate the root. The jet route is one
// IntervalSecondOrderJetEvaluator pass for all K directions. Each cell certifies K second-derivative constraints (one
// per variable).
TEST_CASE("Second-order interval jet vs twice-differentiated dag: time per tree", "[.second-order-jet-bench]")
{
    using J2 = Operon::IntervalSecondOrderJetEvaluator<S>;
    auto const ds = MakeDataset(5);
    std::vector<Operon::Hash> allVars;
    for (std::size_t i = 1; i <= 5; ++i) {
        allVars.push_back(VarHash(ds, i));
    }
    auto const pset = MakePset(false, 6, 6);
    Operon::RandomGenerator treeRng { CorpusSeed };
    std::mt19937_64 rng { CorpusSeed };
    constexpr std::size_t TreesPerCell = 300;
    constexpr std::size_t Reps = 8;
    fmt::print(
        "\n============ second-order: time per tree (K second-derivative constraints), nanoseconds ============\n");
    fmt::print("seed {}; {} trees per cell x {} reps; dag = dag1 + K x (slice d1, dag2, IntervalEvaluator pass); jet = "
               "IntervalSecondOrderJetEvaluator "
               "construct + evaluate\n",
        CorpusSeed, TreesPerCell, Reps);
    fmt::print("{:>4}{:>3}{:>10}{:>10}{:>10}{:>10}{:>9}{:>10}{:>10}\n", "len", "K", "dag med", "dag p95", "jet med",
        "jet p95", "speedup", "dag alloc", "jet alloc");
    for (std::size_t length : { 10UL, 20UL, 30UL }) {
        for (std::size_t K = 1; K <= 4; ++K) {
            std::vector<Item> items;
            std::size_t attempts = 0;
            while (items.size() < TreesPerCell && attempts++ < 2'000'000) {
                auto tree = BalancedTreeCreator { &pset, allVars, 0.0, 40 }(treeRng, length, 0, 40);
                auto present = DistinctVariables(tree);
                if (present.size() < K) {
                    continue;
                }
                Item item;
                item.Vars.assign(present.begin(), present.begin() + static_cast<std::ptrdiff_t>(K));
                item.Coeff = tree.GetCoefficients();
                for (auto& c : item.Coeff) {
                    c = static_cast<S>(std::uniform_real_distribution<double> { 0.5, 2.0 }(rng));
                }
                tree.SetCoefficients(item.Coeff);
                for (auto const h : allVars) {
                    auto const lo = std::uniform_real_distribution<double> { 0.5, 2.0 }(rng);
                    item.Domains.insert_or_assign(h,
                        std::pair { static_cast<S>(lo),
                            static_cast<S>(lo + std::uniform_real_distribution<double> { 0.05, 1.0 }(rng)) });
                }
                item.Tree = std::move(tree);
                items.push_back(std::move(item));
            }
            REQUIRE(items.size() == TreesPerCell);

            auto const dagFull = Measure(items, Reps, [&](Item const& it) {
                auto const dag1 = BuildVariableGradientDag(it.Tree, it.Coeff, it.Vars);
                auto sink = 0.0;
                for (std::size_t k = 0; k < it.Vars.size(); ++k) {
                    auto const root = dag1.Roots[k];
                    if (root == NoGrad) {
                        continue;
                    }
                    auto const d1 = Operon::SliceDagRoot(dag1.Nodes, root);
                    auto const dag2 = BuildVariableGradientDag(d1, d1.GetCoefficients(), std::vector { it.Vars[k] });
                    IE ie(&d1, it.Domains);
                    ie.SetNodes(dag2.Nodes);
                    static_cast<void>(ie.EvaluateAll(d1.GetCoefficients(), [&](Operon::Span<Interval const> v) {
                        for (auto const r2 : dag2.Roots) {
                            if (r2 != NoGrad) {
                                sink += static_cast<double>(v[r2].sup());
                            }
                        }
                    }));
                }
                gSink = gSink + sink;
            });
            auto const jetFull = Measure(items, Reps, [&](Item const& it) {
                J2 je(&it.Tree, it.Domains, it.Vars);
                auto const r = je(it.Coeff);
                if (r) {
                    auto sink = static_cast<double>(r->Value.sup());
                    for (std::size_t k = 0; k < K; ++k) {
                        sink += static_cast<double>(r->Second[k].sup());
                    }
                    gSink = gSink + sink;
                }
            });
            fmt::print("{:>4}{:>3}{:>10.0f}{:>10.0f}{:>10.0f}{:>10.0f}{:>8.2f}x{:>10.1f}{:>10.1f}\n", length, K,
                dagFull.Median, dagFull.P95, jetFull.Median, jetFull.P95, dagFull.Median / jetFull.Median,
                dagFull.Allocations, jetFull.Allocations);
        }
    }
    fmt::print(
        "=======================================================================================================\n");
}

} // namespace Operon::Test
