// Simple GPU evaluator unit test.
// Constructs known trees, uploads the first variable (X1) as the synthetic
// target, and checks that the fused eval+fitness kernel returns the correct
// R² value for each tree.
//
// Expected results (R², with linear scaling enabled):
//   X1        vs X1  →  R²=1  → fitness ≈ -1.000  (exact)
//   2*X1      vs X1  →  R²=1  → fitness ≈ -1.000  (perfect linear)
//   const 3.14 vs X1 →  R²=0  → fitness ≈  0.000  (zero variance)
//   X1+X2     vs X1  →  R²<1  → fitness in (-1, 0)
//   X1-X2     vs X1  →  R²<1  → fitness in (-1, 0)
//
// Compile from the SYCL cmake build target (add_sycl_to_target wires acpp).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/range.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/backend/sycl/gpu_kernel.hpp"
#include "operon/interpreter/backend/sycl/population_encoder.hpp"

namespace {

auto makeVar(Operon::Hash hash, float coeff) -> Operon::Node {
    Operon::Node n(Operon::NodeType::Variable);
    n.HashValue = hash;
    n.Value     = coeff;
    n.Arity     = 0;
    n.Length    = 0;
    return n;
}
auto makeConst(float val) -> Operon::Node {
    auto n   = Operon::Node::Constant(val);
    n.Length = 0;
    return n;
}
auto makeOp(Operon::NodeType t, uint16_t arity, uint32_t subtreeLen) -> Operon::Node {
    Operon::Node n(t);
    n.Arity  = arity;
    n.Length = subtreeLen;
    n.Value  = 1.0;
    return n;
}

struct TestCase {
    char const*                   name;
    std::vector<Operon::Node>     nodes;
    float                         expectedFitness; // -R² with linear scaling
    float                         tolerance;
};

} // namespace

int main() {
    // Use a small slice so the test is fast; correctness is what we check.
    auto ds    = Operon::Dataset("data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Operon::Range{0, 100};

    auto vars  = ds.GetVariables();
    auto hashOf = [&](std::string const& name) -> Operon::Hash {
        for (auto const& v : vars) { if (v.Name == name) return v.Hash; }
        return 0;
    };
    Operon::Hash hX1 = hashOf("X1");
    Operon::Hash hX2 = hashOf("X2");
    Operon::Hash hX3 = hashOf("X3");

    printf("=== GPU fused eval+fitness kernel tests (Poly-10, %zu rows) ===\n\n",
           range.Size());

    // Fetch X1 and use it as the synthetic target
    auto x1span = ds.GetValues(hX1).subspan(range.Start(), range.Size());
    std::vector<float> targetF(range.Size());
    std::transform(x1span.begin(), x1span.end(), targetF.begin(),
                   [](auto v) { return static_cast<float>(v); });

    std::vector<TestCase> tests = {
        // Perfect match — X1 evaluated against X1 target
        { "X1",
          { makeVar(hX1, 1.0f) },
          -1.0f, 1e-4f },
        // Linearly related — with scaling R² = 1
        { "2*X1",
          { makeVar(hX1, 2.0f) },
          -1.0f, 1e-4f },
        // Constant has zero variance; linear scaling is undefined → R²≈0
        // (GPU kernel sets fitness to 0 when varX==0 and doScale==true)
        { "const(3.14)",
          { makeConst(3.14f) },
          0.0f, 1e-3f },
        // X1+X2 linearly related to X1 only if cov large; just check it is
        // a finite value in the expected range
        { "X1+X2",
          { makeVar(hX2, 1.0f), makeVar(hX1, 1.0f),
            makeOp(Operon::NodeType::Add, 2, 2) },
          -1.0f, 1.0f },  // wide tolerance — just checks finite + ≤ 0
        // X1-X2 similarly
        { "X1-X2",
          { makeVar(hX2, 1.0f), makeVar(hX1, 1.0f),
            makeOp(Operon::NodeType::Sub, 2, 2) },
          -1.0f, 1.0f },
        // Composite (X1+X2)-X3
        { "(X1+X2)-X3",
          { makeVar(hX3, 1.0f), makeVar(hX2, 1.0f), makeVar(hX1, 1.0f),
            makeOp(Operon::NodeType::Add, 2, 2),
            makeOp(Operon::NodeType::Sub, 2, 4) },
          -1.0f, 1.0f },
    };

    // Build population
    std::vector<Operon::Individual> pop(tests.size());
    for (std::size_t i = 0; i < tests.size(); ++i) {
        pop[i].Genotype = Operon::Tree(tests[i].nodes);
        pop[i].Genotype.UpdateNodes();
    }

    auto enc = Operon::Sycl::EncodePopulation(
        Operon::Span<Operon::Individual const>{pop.data(), pop.size()},
        ds, range);

    auto* ctx = Operon::Sycl::GpuContextCreate();
    Operon::Sycl::GpuContextUploadDataset(ctx, enc.DataBuffer.data(), enc.NVars, enc.NRows);
    Operon::Sycl::GpuContextUploadTarget(ctx, targetF.data(), enc.NRows);

    std::vector<float> fitness(enc.PopSize);
    Operon::Sycl::GpuContextEvaluate(ctx,
                                      enc.Ops.data(), enc.Lengths.data(),
                                      enc.PopSize, enc.MaxLen,
                                      Operon::Sycl::GpuFitType::R2,
                                      /*doScale=*/true,
                                      fitness.data());
    Operon::Sycl::GpuContextDestroy(ctx);

    // Build sorted-index → original map
    std::vector<uint32_t> origToSi(enc.PopSize);
    for (uint32_t si = 0; si < enc.PopSize; ++si) {
        origToSi[static_cast<uint32_t>(enc.SortedIndices[si])] = si;
    }

    bool allPassed = true;
    for (std::size_t ti = 0; ti < tests.size(); ++ti) {
        auto const& tc  = tests[ti];
        auto const  si  = origToSi[static_cast<uint32_t>(ti)];
        float const fit = fitness[si];

        bool finite  = std::isfinite(fit);
        bool inRange = fit >= -1.0f - 1e-4f && fit <= 0.0f + 1e-4f;
        bool close   = std::abs(fit - tc.expectedFitness) <= tc.tolerance;
        bool passed  = finite && inRange && close;

        printf("[%s] %-20s  fitness=%.6f  expected≈%.3f±%.3f  finite=%s range=%s\n",
               passed ? "PASS" : "FAIL",
               tc.name,
               static_cast<double>(fit),
               static_cast<double>(tc.expectedFitness),
               static_cast<double>(tc.tolerance),
               finite  ? "yes" : "no",
               inRange ? "yes" : "no");

        if (!passed) allPassed = false;
    }

    printf("\n%s\n", allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return allPassed ? 0 : 1;
}
