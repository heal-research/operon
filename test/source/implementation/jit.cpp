// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifdef HAVE_ASMJIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#include "operon/core/problem.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/parser/infix.hpp"
#include "operon/optimizer/jit_lm_cost_function.hpp"
#include "operon/optimizer/optimizer.hpp"

#include "../operon_test.hpp"

namespace nb = ankerl::nanobench;

namespace Operon::Test {

namespace {

using DTable = DispatchTable<Operon::Scalar>;

// Evaluate tree via a compiled function; returns vector of nRows results.
auto EvalCompiled(JIT::CompileMeta const& compiled,
                  Operon::Tree const& tree,
                  Operon::Dataset const& ds, Operon::Range range) -> std::vector<float>
{
    auto const nRows    = static_cast<int32_t>(range.Size());
    auto const nRowsPad = (nRows + 7) & ~7;

    auto const varOrder = JIT::VarOrder(tree);
    std::vector<float const*> colPtrs(varOrder.size());
    for (std::size_t i = 0; i < varOrder.size(); ++i) {
        colPtrs[i] = ds.GetPaddedValues(varOrder[i]) + range.Start();
    }
    auto coeff = tree.GetCoefficients();
    std::vector<float> scratch(nRowsPad);
    compiled.fn(scratch.data(), colPtrs.data(), nRowsPad,
                coeff.empty() ? nullptr : coeff.data());
    return {scratch.begin(), scratch.begin() + nRows};
}

auto EvalJIT_AVX2(JIT::TreeCompiler& compiler, Operon::Tree const& tree,
                  Operon::Dataset const& ds, Operon::Range range) -> std::vector<float>
{
    auto compiled = compiler.CompileAVX2(tree);
    REQUIRE(compiled != nullptr);
    REQUIRE(compiled->fn != nullptr);
    return EvalCompiled(*compiled, tree, ds, range);
}

// Evaluate tree via reference Interpreter.
auto EvalRef(Operon::Tree& tree, Operon::Dataset const& ds, Operon::Range range) -> std::vector<float>
{
    DTable dtable;
    auto coeff = tree.GetCoefficients();
    auto result = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(coeff, range);
    return {result.begin(), result.end()};
}

constexpr float Tol = 1e-4f;

} // namespace

TEST_CASE("JIT AVX2 correctness", "[jit][avx2]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    // Use a row count that is not a multiple of 8 to exercise the tail loop.
    auto range = Range{0, 201};

    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};

    // Skip if AVX2 is not available on this CPU.
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    auto check = [&](std::string_view expr) {
        INFO("expression: " << expr);
        auto tree = InfixParser::Parse(std::string(expr), ds);
        auto ref  = EvalRef(tree, ds, range);
        auto avx2 = EvalJIT_AVX2(compiler, tree, ds, range);

        REQUIRE(ref.size() == avx2.size());
        for (std::size_t i = 0; i < ref.size(); ++i) {
            INFO("row " << i << ": ref=" << ref[i] << " avx2=" << avx2[i]);
            if (std::isnan(ref[i])) {
                CHECK(std::isnan(avx2[i]));
            } else if (std::isinf(ref[i])) {
                CHECK(std::isinf(avx2[i]));
                CHECK((ref[i] > 0) == (avx2[i] > 0));
            } else {
                CHECK(avx2[i] == Catch::Approx(ref[i]).epsilon(Tol));
            }
        }
    };

    SECTION("Add")         { check("X1 + X2 + X3"); }
    SECTION("Sub")         { check("X1 - X2"); }
    SECTION("Mul")         { check("X1 * X2 * X3"); }
    SECTION("Div")         { check("X1 / X2"); }
    SECTION("Nested")      { check("X1 + X2 * X3 - X4"); }
    SECTION("Square")      { check("X1 * X1 + X2 * X2"); }
    SECTION("Unary neg")   { check("0 - X1"); }
    SECTION("Sqrt")        { check("sqrt(X1 * X1)"); }
    SECTION("Sqrtabs")     { check("sqrt(abs(X1))"); }
    SECTION("Abs")         { check("abs(X1 - X2)"); }
    SECTION("Sin")         { check("sin(X1)"); }
    SECTION("Cos")         { check("cos(X1)"); }
    SECTION("Tan")         { check("tan(X1)"); }
    SECTION("Asin")        { check("asin(X1)"); }
    SECTION("Acos")        { check("acos(X1)"); }
    SECTION("Atan")        { check("atan(X1)"); }
    SECTION("Sinh")        { check("sinh(X1)"); }
    SECTION("Cosh")        { check("cosh(X1)"); }
    SECTION("Cbrt")        { check("cbrt(X1)"); }
    SECTION("Log1p")       { check("log1p(abs(X1))"); }
    SECTION("Exp")         { check("exp(X1)"); }
    SECTION("Log")         { check("log(X1)"); }
    SECTION("Logabs")      { check("log(abs(X1))"); }
    SECTION("Tanh")        { check("tanh(X1)"); }
    SECTION("Aq")          { check("X1 / sqrt(1 + X2 * X2)"); }
    SECTION("Powabs")      { check("powabs(X1, 2)"); }
    SECTION("Composite")   { check("sin(X1) * cos(X2) + exp(0 - X3 * X3)"); }
    SECTION("Pow")         { check("X1 ^ X2"); }
    // Rows=201: 25 full AVX2 iterations (200 rows) + 1 scalar tail row
    SECTION("Tail rows")   { check("X1 * X2 + X3"); }
    // Many simultaneous transcendentals — exercises ymm spill/fill at invoke sites.
    SECTION("Many transcendentals") {
        check("sin(X1) + cos(X2) + exp(X3) + log(abs(X4)) + tanh(X5)");
        check("sin(X1) * cos(X2) + exp(X3) * tanh(X4) + log(abs(X5)) * sin(X6)");
        check("(sin(X1) + cos(X2)) * (exp(X3) + tanh(X4)) + (log(abs(X5)) * sin(X6) + cos(X7))");
    }
}

// Floor/Ceil have no infix syntax (infix-parser's node_type enum has no
// floor/ceil variant, a pre-existing gap unrelated to the JIT registry), so
// unlike the other unary ops above they can't go through check()'s
// InfixParser::Parse path — built directly instead.
TEST_CASE("JIT AVX2 correctness: floor/ceil", "[jit][avx2]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, 201};

    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) { SKIP("AVX2 not available"); }

    auto x1Hash = ds.GetVariable("X1").value().Hash;

    auto checkOp = [&](Operon::BuiltinOp op) {
        Node v1(NodeType::Variable); v1.HashValue = v1.CalculatedHashValue = x1Hash; v1.Value = 1.0F;
        auto tree = Tree({v1, Node::Function(static_cast<Operon::Hash>(op), 1)}).UpdateNodes();
        auto ref  = EvalRef(tree, ds, range);
        auto avx2 = EvalJIT_AVX2(compiler, tree, ds, range);

        REQUIRE(ref.size() == avx2.size());
        for (std::size_t i = 0; i < ref.size(); ++i) {
            INFO("row " << i << ": ref=" << ref[i] << " avx2=" << avx2[i]);
            CHECK(avx2[i] == Catch::Approx(ref[i]).epsilon(Tol));
        }
    };

    SECTION("Floor") { checkOp(Operon::BuiltinOp::Floor); }
    SECTION("Ceil")  { checkOp(Operon::BuiltinOp::Ceil); }
}

// Smoke test for Ref node handling in the JIT compiler. The explicit register
// copy (vmovaps/vmovups) is preventive — it shortens live ranges and simplifies
// the use graph for asmjit's RA, but no wrong-result bug was observed with the
// old aliasing code. This test exercises a tree shape with two independent use
// chains from a shared sub-expression to verify the code path end-to-end.
TEST_CASE("JIT Ref node forward-pass correctness", "[jit][ref]")
{
    auto ds    = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, std::min(ds.Rows<std::size_t>(), std::size_t{200})};

    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};

    // Postfix layout:
    //   0: X1                            (Variable, Len=0)
    //   1: X2                            (Variable, Len=0)
    //   2: Add(X1,X2)   Arity=2 Len=2   S = X1+X2 (shared sub-expression)
    //   3: X3                            (Variable, Len=0)
    //   4: Ref(2)                        copy of S (Len=0)
    //   5: Sub(Ref,X3)  Arity=2 Len=2   S - X3   (consumes Ref + X3)
    //   6: Mul(Add,Sub) Arity=2 Len=6   S * (S - X3) (consumes node 2 + node 5)
    //
    // Node 2's value is consumed independently by Mul (directly) and Sub (via Ref
    // at node 4), creating two live use chains. With the old register-aliasing
    // code the Ref shared node 2's virtual register, extending its liveness across
    // the Sub emit region — exactly the pattern that triggered RA mis-scheduling.
    //
    // Semantically: (X1+X2) * ((X1+X2) - X3)
    auto x1Hash = ds.GetVariable("X1").value().Hash;
    auto x2Hash = ds.GetVariable("X2").value().Hash;
    auto x3Hash = ds.GetVariable("X3").value().Hash;

    Operon::Vector<Node> nodes;
    {
        Node v1(NodeType::Variable); v1.HashValue = v1.CalculatedHashValue = x1Hash; v1.Value = 1.0F;
        Node v2(NodeType::Variable); v2.HashValue = v2.CalculatedHashValue = x2Hash; v2.Value = 1.0F;
        auto add = Util::MakeOp<BuiltinOp::Add>(); add.Length = 2;
        Node v3(NodeType::Variable); v3.HashValue = v3.CalculatedHashValue = x3Hash; v3.Value = 1.0F;
        Node ref = Node::Ref(2);
        auto sub = Util::MakeOp<BuiltinOp::Sub>(); sub.Length = 2;
        auto mul = Util::MakeOp<BuiltinOp::Mul>(); mul.Length = 6;
        nodes.push_back(v1);
        nodes.push_back(v2);
        nodes.push_back(add);
        nodes.push_back(v3);
        nodes.push_back(ref);
        nodes.push_back(sub);
        nodes.push_back(mul);
    }
    Tree tree{nodes};

    // Reference: interpreter
    auto ref = EvalRef(tree, ds, range);

    // Verify against the known formula: (X1+X2) * ((X1+X2) - X3)
    {
        auto const* x1 = ds.GetPaddedValues(x1Hash);
        auto const* x2 = ds.GetPaddedValues(x2Hash);
        auto const* x3 = ds.GetPaddedValues(x3Hash);
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto s = x1[i] + x2[i];
            auto expected = s * (s - x3[i]);
            INFO("row " << i << ": expected=" << expected << " ref=" << ref[i]);
            CHECK(ref[i] == Catch::Approx(expected).epsilon(Tol));
        }
    }

    SECTION("AVX2") {
        if (!compiler.HasAVX2()) { SKIP("AVX2 not available"); }
        auto avx2 = EvalJIT_AVX2(compiler, tree, ds, range);
        REQUIRE(ref.size() == avx2.size());
        for (std::size_t i = 0; i < ref.size(); ++i) {
            INFO("row " << i << ": ref=" << ref[i] << " avx2=" << avx2[i]);
            CHECK(avx2[i] == Catch::Approx(ref[i]).epsilon(Tol));
        }
    }
}

TEST_CASE("JitEvaluator correctness", "[jit][evaluator]")
{
    auto ds    = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, std::min(ds.Rows<std::size_t>(), std::size_t{200})};

    Problem problem{&ds};
    problem.SetTarget("Y");
    problem.SetTrainingRange(range);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    problem.SetInputs(inputs);

    RandomGenerator rng(1234);
    JIT::JitZobrist zobrist(rng, /*maxLength=*/50, inputs);

    DTable dtable;
    Evaluator<DTable> refEval(&problem, &dtable, MSE{}, /*linearScaling=*/true);
    JIT::JitEvaluator  jitEval(&problem, &zobrist, MSE{}, /*linearScaling=*/true);

    auto check = [&](std::string_view expr) {
        INFO("expression: " << expr);
        auto tree = InfixParser::Parse(std::string(expr), ds);
        Individual refInd(1);
        Individual jitInd(1);
        refInd.Genotype = tree;
        jitInd.Genotype = tree;

        auto refFit = refEval(rng, refInd)[0];
        auto jitFit = jitEval(rng, jitInd)[0];

        INFO("ref=" << refFit << " jit=" << jitFit);
        if (std::isfinite(refFit)) {
            CHECK(jitFit == Catch::Approx(refFit).epsilon(1e-3F));
        } else {
            CHECK(!std::isfinite(jitFit));
        }
    };

    SECTION("Linear")     { check("X1 + X2 + X3"); }
    SECTION("Product")    { check("X1 * X2 * X3"); }
    SECTION("Sin")        { check("sin(X1)"); }
    SECTION("Composite")  { check("sin(X1) * cos(X2) + exp(0 - X3 * X3)"); }

    SECTION("Cache reuse") {
        auto tree  = InfixParser::Parse("X1 * X2 + X3", ds);
        Individual ind(1);
        ind.Genotype = tree;

        jitEval.operator()(rng, ind);
        jitEval.operator()(rng, ind);

        CHECK(jitEval.CacheSize()  == 1);
        CHECK(jitEval.CacheHits() >= 1);
    }

    SECTION("CacheMisses and ResetCounters") {
        // Length gate forces a miss.
        jitEval.SetMaxLength(1);

        auto tree = InfixParser::Parse("X1 * X2 + X3", ds);  // length > 1
        auto const* c = jitEval.GetOrCompile(tree);
        CHECK(c == nullptr);
        CHECK(jitEval.CacheMisses() >= 1);

        // ResetCounters clears hit/miss/fail counters without touching the cache.
        jitEval.SetMaxLength(0);
        auto const* c2 = jitEval.GetOrCompile(tree);  // this will compile and increment hits
        CHECK(c2 != nullptr);

        jitEval.ResetCounters();
        CHECK(jitEval.CacheHits()   == 0);
        CHECK(jitEval.CacheMisses() == 0);
        // Cache itself is unaffected — the compiled entry is still there.
        CHECK(jitEval.CacheSize() >= 1);
    }
}

// Same oversized-scratch-buffer bug class as issue #114/#116 (fixed there for
// Evaluator<DTable>), but for JitEvaluator's own operator(): the compiled
// path only ever writes range.Size() rows, and the fallback path's
// Interpreter::Evaluate only writes when given a span sized exactly to the
// range - so an oversized caller-owned buffer's tail must never leak into
// scaling/the error metric. The tail is poisoned with NaN so a wrong slice
// shows up as a non-finite result rather than passing by coincidence.
TEST_CASE("JitEvaluator: oversized buffer matches exact-size buffer", "[jit][evaluator]")
{
    auto ds    = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, std::min(ds.Rows<std::size_t>(), std::size_t{200})};

    Problem problem{&ds};
    problem.SetTarget("Y");
    problem.SetTrainingRange(range);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    problem.SetInputs(inputs);

    RandomGenerator rng(1234);
    auto tree = InfixParser::Parse("X1 + X2 + X3", ds);
    Individual ind(1);
    ind.Genotype = tree;

    auto makeOversizedBuf = [&] {
        std::vector<Scalar> buf(range.Size() + 50, std::numeric_limits<Scalar>::quiet_NaN());
        return buf;
    };

    SECTION("Compiled path") {
        JIT::JitZobrist   zobrist(rng, /*maxLength=*/50, inputs);
        JIT::JitEvaluator jitEval(&problem, &zobrist, MSE{}, /*linearScaling=*/true);
        jitEval.SetMinVisits(1); // compile on first call

        std::vector<Scalar> exactBuf(range.Size());
        auto const exactResult = jitEval(rng, ind, exactBuf);
        REQUIRE(jitEval.GetOrCompile(tree) != nullptr); // sanity: compiled path was actually taken

        auto oversizedBuf = makeOversizedBuf();
        auto const oversizedResult = jitEval(rng, ind, oversizedBuf);

        REQUIRE(exactResult.size() == 1);
        REQUIRE(oversizedResult.size() == 1);
        CHECK(std::isfinite(oversizedResult[0]));
        CHECK(oversizedResult[0] == exactResult[0]);
    }

    SECTION("Fallback (uncompiled) path") {
        JIT::JitZobrist   zobrist(rng, /*maxLength=*/50, inputs);
        JIT::JitEvaluator jitEval(&problem, &zobrist, MSE{}, /*linearScaling=*/true);
        jitEval.SetMaxLength(1); // forces GetOrCompile to return nullptr for this tree

        std::vector<Scalar> exactBuf(range.Size());
        auto const exactResult = jitEval(rng, ind, exactBuf);
        REQUIRE(jitEval.GetOrCompile(tree) == nullptr); // sanity: fallback path was actually taken

        auto oversizedBuf = makeOversizedBuf();
        auto const oversizedResult = jitEval(rng, ind, oversizedBuf);

        REQUIRE(exactResult.size() == 1);
        REQUIRE(oversizedResult.size() == 1);
        CHECK(std::isfinite(oversizedResult[0]));
        CHECK(oversizedResult[0] == exactResult[0]);
    }
}

// ============================================================
// Zobrist structural hash: constants and variable weights are ignored
// ============================================================

TEST_CASE("Zobrist hash is structural: constant values do not affect the hash", "[jit][zobrist]")
{
    auto ds       = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto allHashes = ds.VariableHashes();

    constexpr int maxLen = 50;
    RandomGenerator rng(42); // NOLINT(cert-msc51-cpp)
    JIT::JitZobrist zobrist(rng, maxLen, allHashes);

    SECTION("same-structure trees with different literal constants get the same hash") {
        // "X1 + 1.0" → [Variable(X1), Constant(1.0), Add]
        auto tree1 = InfixParser::Parse("X1 + 1.0", ds);
        auto tree2 = InfixParser::Parse("X1 + 1.0", ds);
        for (auto& nd : tree2.Nodes()) {
            if (nd.IsConstant()) { nd.Value = 99.0F; }
        }
        CHECK(zobrist.ComputeHash(tree1) == zobrist.ComputeHash(tree2));
    }

    SECTION("same-structure trees with different variable weights get the same hash") {
        auto tree1 = InfixParser::Parse("X1 + X2", ds);
        auto tree2 = InfixParser::Parse("X1 + X2", ds);
        for (auto& nd : tree2.Nodes()) {
            if (nd.IsVariable()) { nd.Value = 42.0F; }
        }
        CHECK(zobrist.ComputeHash(tree1) == zobrist.ComputeHash(tree2));
    }

    SECTION("different-structure trees get different hashes") {
        auto tree1 = InfixParser::Parse("X1 + X2", ds);
        auto tree2 = InfixParser::Parse("X1 * X2 + X3", ds);
        CHECK(zobrist.ComputeHash(tree1) != zobrist.ComputeHash(tree2));
    }

    SECTION("GetOrCompile reuses the same compiled function for same-structure trees") {
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable("Y").value().Hash);

        Problem problem{&ds};
        problem.SetTarget("Y");
        problem.SetTrainingRange({0, 100});
        problem.SetInputs(inputs);

        JIT::JitEvaluator jitEval(&problem, &zobrist, MSE{}, /*linearScaling=*/false);
        jitEval.SetBudget(std::numeric_limits<std::size_t>::max());

        auto tree1 = InfixParser::Parse("X1 + 1.0", ds);
        auto tree2 = InfixParser::Parse("X1 + 1.0", ds);
        for (auto& nd : tree2.Nodes()) {
            if (nd.IsConstant()) { nd.Value = 7.0F; }
        }

        auto const* c1 = jitEval.GetOrCompile(tree1);
        auto const* c2 = jitEval.GetOrCompile(tree2);

        REQUIRE(c1 != nullptr);
        REQUIRE(c2 != nullptr);
        CHECK(c1 == c2);           // same compiled function pointer
        CHECK(jitEval.CacheSize()  == 1);
        CHECK(jitEval.CacheHits() >= 1);
    }
}

// ============================================================
// Population-level correctness: JitEvaluator vs interpreter on random GP trees
// ============================================================

TEST_CASE("JitEvaluator vs interpreter on random population", "[jit][evaluator][population]")
{
    auto ds    = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, std::min(ds.Rows<std::size_t>(), std::size_t{200})};

    Problem problem{&ds};
    problem.SetTarget("Y");
    problem.SetTrainingRange(range);

    // Full symbol set matching the failing comparison runs (plus Variable as terminal).
    PrimitiveSet pset;
    pset.SetConfig(BuiltinOp::Add | BuiltinOp::Sub | BuiltinOp::Mul | BuiltinOp::Div |
                   BuiltinOp::Sin | BuiltinOp::Cos | BuiltinOp::Exp | BuiltinOp::Log |
                   BuiltinOp::Pow | BuiltinOp::Sqrt | BuiltinOp::Tanh |
                   NodeType::Variable);

    // Pass all dataset variable hashes since the creator may produce any variable.
    RandomGenerator rng(42);
    JIT::JitZobrist zobrist(rng, /*maxLength=*/50, ds.VariableHashes());

    DTable dtable;
    Evaluator<DTable>  refEval(&problem, &dtable, MSE{}, /*linearScaling=*/true);
    JIT::JitEvaluator  jitEval(&problem, &zobrist, MSE{}, /*linearScaling=*/true);

    // Generate 200 random trees and compare fitness.
    constexpr int MaxLength = 50;
    BalancedTreeCreator creator{&pset, ds.VariableHashes(), 0.0, MaxLength};
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    std::size_t fitMismatch = 0;
    std::size_t compileFailAVX2 = 0;
    for (int i = 0; i < 200; ++i) {
        auto tree = creator(rng, MaxLength, 1, 1000);
        tree.Reduce();

        // Check compilation directly.
        auto avx2 = compiler.CompileAVX2(tree);
        if (!avx2) { ++compileFailAVX2; }

        Individual refInd(1), jitInd(1);
        refInd.Genotype = jitInd.Genotype = tree;

        auto refFit = refEval(rng, refInd)[0];
        auto jitFit = jitEval(rng, jitInd)[0];

        if (std::isfinite(refFit) && std::isfinite(jitFit)) {
            if (std::abs(refFit - jitFit) > 1e-3F * std::max(1.0F, std::abs(refFit))) {
                INFO("tree " << i << " fitness mismatch: ref=" << refFit << " jit=" << jitFit);
                ++fitMismatch;
            }
        } else if (std::isfinite(refFit) && !std::isfinite(jitFit)) {
            INFO("tree " << i << " jit got non-finite when ref=" << refFit);
            ++fitMismatch;
        }
    }
    INFO("compile failures: AVX2=" << compileFailAVX2);
    CHECK(compileFailAVX2 == 0);
    CHECK(fitMismatch == 0);
}

// ============================================================
// CompileJacobian tests
// ============================================================

namespace {

// Evaluate a compiled Jacobian over `range` and return an Eigen matrix matching
// the JacRev layout: shape (nRows, nConsts), column-major.
auto EvalCompiledJacobian(
    JIT::CompileMeta const& compiled,
    Operon::Tree const& tree,
    Operon::Span<Operon::Scalar const> coeff,
    Dataset const& ds,
    Range range
) -> Eigen::Array<Operon::Scalar, -1, -1>
{
    auto const nRows    = static_cast<int32_t>(range.Size());
    auto const nRowsPad = (nRows + 7) & ~7;
    auto const nConsts  = static_cast<Eigen::Index>(tree.CoefficientsCount());

    std::vector<std::vector<float>> colStorage(static_cast<std::size_t>(nConsts),
                                               std::vector<float>(static_cast<std::size_t>(nRowsPad)));
    std::vector<float*> outPtrs(static_cast<std::size_t>(nConsts));
    for (std::size_t k = 0; k < static_cast<std::size_t>(nConsts); ++k) {
        outPtrs[k] = colStorage[k].data();
    }

    auto const varOrder = JIT::VarOrder(tree);
    std::vector<float const*> colPtrs(varOrder.size());
    for (std::size_t i = 0; i < varOrder.size(); ++i) {
        colPtrs[i] = ds.GetPaddedValues(varOrder[i]) + range.Start();
    }

    compiled.jacFn(outPtrs.data(), colPtrs.data(), nRowsPad,
                   coeff.empty() ? nullptr : coeff.data());

    Eigen::Array<Operon::Scalar, -1, -1> jac(nRows, nConsts);
    for (Eigen::Index k = 0; k < nConsts; ++k) {
        for (Eigen::Index r = 0; r < nRows; ++r) {
            jac(r, k) = colStorage[static_cast<std::size_t>(k)][static_cast<std::size_t>(r)];
        }
    }
    return jac;
}

// Same supported pset as in tree_diff.cpp tests (excludes nodes with zero-derivative support).
auto MakeSupportedPsetJit() -> PrimitiveSet {
    PrimitiveSet ps;
    ps.SetConfig(
        BuiltinOp::Add | BuiltinOp::Mul | BuiltinOp::Sub | BuiltinOp::Div |
        BuiltinOp::Exp | BuiltinOp::Log | BuiltinOp::Logabs | BuiltinOp::Log1p |
        BuiltinOp::Sin | BuiltinOp::Cos | BuiltinOp::Tan  |
        BuiltinOp::Asin | BuiltinOp::Acos | BuiltinOp::Atan |
        BuiltinOp::Sinh | BuiltinOp::Cosh | BuiltinOp::Tanh |
        BuiltinOp::Sqrt | BuiltinOp::Cbrt | BuiltinOp::Square |
        BuiltinOp::Pow  |
        NodeType::Constant
    );
    return ps;
}

} // namespace

TEST_CASE("CompileJacobian - single constant", "[jit][jacobian]")
{
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, 100};

    // Build tree: a single constant c (Optimize=true)
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(2.5F); c.Optimize = true;
    nodes.push_back(c);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);
    auto compiled = compiler.CompileJacobian(dag);
    REQUIRE(compiled != nullptr);
    REQUIRE(compiled->jacFn != nullptr);
    REQUIRE(tree.CoefficientsCount() == 1);

    auto coeff = tree.GetCoefficients();
    auto jit = EvalCompiledJacobian(*compiled, tree, coeff, ds, range);

    // d(c)/dc = 1 for all rows
    REQUIRE(jit.rows() == static_cast<Eigen::Index>(range.Size()));
    REQUIRE(jit.cols() == 1);
    for (Eigen::Index r = 0; r < jit.rows(); ++r) {
        CHECK(jit(r, 0) == Catch::Approx(1.0F));
    }
}

TEST_CASE("CompileJacobian correctness vs JacRev - random trees", "[jit][jacobian]")
{
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    constexpr auto nRows  = 100;
    constexpr auto nCols  = 5;
    constexpr auto nTrees = 200;
    constexpr auto maxLen = 30;
    constexpr auto eps    = 1e-3F;
    constexpr auto maxDivergeRate = 0.02;

    Operon::RandomGenerator rng(99UL);
    auto ds = Operon::Test::Util::RandomDataset(rng, nRows, nCols);
    DTable dtable;
    Range const range{0, static_cast<std::size_t>(nRows)};

    auto pset = MakeSupportedPsetJit();
    std::uniform_real_distribution<Operon::Scalar> valDist(-2.F, +2.F);
    std::uniform_int_distribution<std::size_t> lenDist(1, maxLen);
    BalancedTreeCreator const btc{&pset, ds.VariableHashes(), 0.0, maxLen};

    std::size_t finiteMismatch = 0;
    std::size_t finiteDiverge  = 0;
    std::size_t totalCols      = 0;

    for (int t = 0; t < nTrees; ++t) {
        auto tree = btc(rng, lenDist(rng), 1, 1000);
        for (auto& nd : tree.Nodes()) {
            nd.Optimize = nd.IsConstant();
            if (nd.IsConstant()) { nd.Value = valDist(rng); }
        }

        auto const coeff = tree.GetCoefficients();
        if (coeff.empty()) { continue; }

        // Reference: JacRev via interpreter
        Interpreter<Operon::Scalar, DTable> const interp{&dtable, &ds, &tree};
        auto const jrev = interp.JacRev(coeff, range);

        // JIT Jacobian
        auto const dag = BuildJacobianDag(tree);
        auto compiled  = compiler.CompileJacobian(dag);
        REQUIRE(compiled != nullptr);
        auto const jjit = EvalCompiledJacobian(*compiled, tree, coeff, ds, range);

        auto const nk = jrev.cols();
        totalCols += static_cast<std::size_t>(nk);

        for (Eigen::Index k = 0; k < nk; ++k) {
            auto const colRev = jrev.col(k);
            auto const colJit = jjit.col(k);
            bool const revFin = std::isfinite(colRev.sum());
            bool const jitFin = std::isfinite(colJit.sum());
            if (revFin && !jitFin) {
                ++finiteMismatch;
            } else if (revFin && jitFin && !colRev.isApprox(colJit, eps)) {
                ++finiteDiverge;
            }
        }
    }

    INFO("finite mismatch: " << finiteMismatch << " / " << totalCols);
    INFO("finite diverge:  " << finiteDiverge  << " / " << totalCols);
    CHECK(finiteMismatch == 0);
    CHECK(static_cast<double>(finiteDiverge) / static_cast<double>(std::max(totalCols, std::size_t{1})) < maxDivergeRate);
}

TEST_CASE("CompileJacobian correctness - variable weights", "[jit][jacobian]")
{
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    constexpr auto nRows  = 100;
    constexpr auto nCols  = 5;
    constexpr auto nTrees = 500;
    constexpr auto maxLen = 30;
    constexpr auto eps    = 1e-3F;
    constexpr auto maxDivergeRate = 0.02;

    Operon::RandomGenerator rng(42UL);
    auto ds = Operon::Test::Util::RandomDataset(rng, nRows, nCols);
    DTable dtable;
    Range const range{0, static_cast<std::size_t>(nRows)};

    auto pset = MakeSupportedPsetJit();
    std::uniform_int_distribution<std::size_t> lenDist(1, maxLen);
    BalancedTreeCreator const btc{&pset, ds.VariableHashes(), 0.0, maxLen};

    std::size_t finiteMismatch = 0;
    std::size_t finiteDiverge  = 0;
    std::size_t totalCols      = 0;

    for (int t = 0; t < nTrees; ++t) {
        auto tree = btc(rng, lenDist(rng), 1, 1000);
        // Use the default Optimize=IsLeaf() setting — Variables get Optimize=true.
        // This matches the real optimizer behaviour.

        auto const coeff = tree.GetCoefficients();
        if (coeff.empty()) { continue; }

        Interpreter<Operon::Scalar, DTable> const interp{&dtable, &ds, &tree};
        auto const jrev = interp.JacRev(coeff, range);

        auto const dag = BuildJacobianDag(tree);
        auto compiled  = compiler.CompileJacobian(dag);
        REQUIRE(compiled != nullptr);
        auto const jjit = EvalCompiledJacobian(*compiled, tree, coeff, ds, range);

        auto const nk = jrev.cols();
        totalCols += static_cast<std::size_t>(nk);

        for (Eigen::Index k = 0; k < nk; ++k) {
            auto const colRev = jrev.col(k);
            auto const colJit = jjit.col(k);
            bool const revFin = std::isfinite(colRev.sum());
            bool const jitFin = std::isfinite(colJit.sum());
            if (revFin && !jitFin) {
                ++finiteMismatch;
            } else if (revFin && jitFin && !colRev.isApprox(colJit, eps)) {
                ++finiteDiverge;
            }
        }
    }

    INFO("finite mismatch: " << finiteMismatch << " / " << totalCols);
    INFO("finite diverge:  " << finiteDiverge  << " / " << totalCols);
    CHECK(finiteMismatch == 0);
    CHECK(static_cast<double>(finiteDiverge) / static_cast<double>(std::max(totalCols, std::size_t{1})) < maxDivergeRate);
}

TEST_CASE("CompileJacobian performance vs JacRev", "[jit][jacobian][performance]")
{
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    constexpr auto nRows  = 1000;
    constexpr auto nCols  = 10;
    constexpr auto nTrees = 200;
    constexpr auto maxLen = 100;

    Operon::RandomGenerator rng(0UL);
    auto ds = Operon::Test::Util::RandomDataset(rng, nRows, nCols);
    DTable dtable;
    Range const range{0, static_cast<std::size_t>(nRows)};

    auto pset = MakeSupportedPsetJit();
    std::uniform_real_distribution<Operon::Scalar> valDist(-2.F, +2.F);
    std::uniform_int_distribution<std::size_t> lenDist(1, maxLen);
    BalancedTreeCreator const btc{&pset, ds.VariableHashes(), 0.0, maxLen};

    std::vector<Tree> trees;
    trees.reserve(nTrees);
    for (int t = 0; t < nTrees; ++t) {
        auto tree = btc(rng, lenDist(rng), 1, 1000);
        for (auto& nd : tree.Nodes()) {
            nd.Optimize = nd.IsConstant();
            if (nd.IsConstant()) { nd.Value = valDist(rng); }
        }
        trees.push_back(std::move(tree));
    }

    std::vector<std::vector<Operon::Scalar>> coeffs;
    coeffs.reserve(trees.size());
    for (auto const& tree : trees) { coeffs.push_back(tree.GetCoefficients()); }

    // Pre-build dags and compile
    std::vector<JacobianDag> dags;
    dags.reserve(trees.size());
    for (auto const& tree : trees) { dags.push_back(BuildJacobianDag(tree)); }

    std::vector<std::unique_ptr<JIT::CompileMeta>> compiled;
    compiled.reserve(dags.size());
    for (auto const& dag : dags) { compiled.push_back(compiler.CompileJacobian(dag)); }

    nb::Bench bench;
    bench.timeUnit(std::chrono::milliseconds(1), "ms");
    bench.relative(true);

    // Baseline: JacRev
    bench.run("JacRev", [&]() {
        for (std::size_t i = 0; i < trees.size(); ++i) {
            if (coeffs[i].empty()) { continue; }
            nb::doNotOptimizeAway(
                Interpreter<Operon::Scalar, DTable>{&dtable, &ds, &trees[i]}.JacRev(coeffs[i], range));
        }
    });

    // CompileJacobian only (construction cost)
    bench.run("CompileJacobian", [&]() {
        for (auto const& dag : dags) {
            nb::doNotOptimizeAway(compiler.CompileJacobian(dag));
        }
    });

    // Evaluate prebuilt compiled Jacobian
    bench.run("EvalCompiledJac (prebuilt)", [&]() {
        for (std::size_t i = 0; i < compiled.size(); ++i) {
            if (coeffs[i].empty() || !compiled[i]) { continue; }
            nb::doNotOptimizeAway(EvalCompiledJacobian(*compiled[i], trees[i], coeffs[i], ds, range));
        }
    });

    bench.render(nb::templates::csv(), std::cout);
}

// ============================================================
// JitLMCostFunction tests
// ============================================================

TEST_CASE("JitLMCostFunction residuals vs interpreter", "[jit][lm]")
{
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    Range const range{0, 100};

    Problem problem{&ds};
    problem.SetTarget("Y");
    problem.SetTrainingRange(range);

    auto const target = problem.TargetValues(range);
    DTable dtable;

    // Build JitLMCostFunction for an expression, evaluate at `evalCoeff`, and
    // compare residuals + Jacobian against the interpreter reference.
    auto checkCostFn = [&](std::string_view exprStr, std::vector<Operon::Scalar> evalCoeff) {
        INFO("expression: " << exprStr);

        auto tree = InfixParser::Parse(std::string(exprStr), ds);
        for (auto& nd : tree.Nodes()) { nd.Optimize = nd.IsConstant(); }

        auto coeff = tree.GetCoefficients();
        if (coeff.empty()) { return; }
        REQUIRE(evalCoeff.size() == coeff.size());

        auto compiled = compiler.CompileAVX2(tree);
        REQUIRE(compiled != nullptr);

        auto dag         = BuildJacobianDag(tree);
        auto compiledJac = compiler.CompileJacobian(dag);
        REQUIRE(compiledJac != nullptr);
        REQUIRE(tree.CoefficientsCount() == static_cast<int>(coeff.size()));

        auto const varOrder = JIT::VarOrder(tree);
        auto makeColPtrs = [&](std::vector<Operon::Hash> const& order) {
            std::vector<float const*> ptrs(order.size());
            for (std::size_t i = 0; i < order.size(); ++i) {
                ptrs[i] = ds.GetPaddedValues(order[i]) + range.Start();
            }
            return ptrs;
        };

        Interpreter<Operon::Scalar, DTable> interp{&dtable, &ds, &tree};

        JitLMCostFunction<> cf{
            gsl::not_null<InterpreterBase<Operon::Scalar> const*>{&interp},
            compiled->fn,
            makeColPtrs(varOrder),
            target, range,
            compiledJac->jacFn,
            makeColPtrs(varOrder)};

        auto const nRes = static_cast<Eigen::Index>(cf.NumResiduals());
        auto const nPar = static_cast<Eigen::Index>(cf.NumParameters());
        REQUIRE(nRes == static_cast<Eigen::Index>(range.Size()));
        REQUIRE(nPar == static_cast<Eigen::Index>(coeff.size()));

        std::vector<Operon::Scalar> jitResiduals(static_cast<std::size_t>(nRes));
        std::vector<Operon::Scalar> jitJacobian(static_cast<std::size_t>(nRes * nPar));

        bool ok = cf.Evaluate(evalCoeff.data(), jitResiduals.data(), jitJacobian.data());
        REQUIRE(ok);

        // Reference residuals: interpreter predict - target
        auto predVec = interp.Evaluate(evalCoeff, range);
        Eigen::Map<const Eigen::Array<Operon::Scalar, -1, 1>> tgtArr(target.data(), nRes);
        Eigen::Map<const Eigen::Array<Operon::Scalar, -1, 1>> predArr(predVec.data(), nRes);
        Eigen::Array<Operon::Scalar, -1, 1> refResiduals = predArr - tgtArr;

        // Reference Jacobian: (nRes, nPar) col-major
        auto refJac = interp.JacRev(evalCoeff, range);
        REQUIRE(refJac.rows() == nRes);
        REQUIRE(refJac.cols() == nPar);

        constexpr float Eps = 1e-4F;

        for (Eigen::Index r = 0; r < nRes; ++r) {
            if (!std::isfinite(refResiduals(r))) { continue; }
            INFO("residual row " << r << ": ref=" << refResiduals(r) << " jit=" << jitResiduals[r]);
            CHECK(jitResiduals[r] == Catch::Approx(refResiduals(r)).epsilon(Eps));
        }

        // jitJacobian is col-major: column k at offset k*nRes
        for (Eigen::Index k = 0; k < nPar; ++k) {
            for (Eigen::Index r = 0; r < nRes; ++r) {
                if (!std::isfinite(refJac(r, k))) { continue; }
                auto jitVal = jitJacobian[static_cast<std::size_t>(k * nRes + r)];
                INFO("jac(" << r << "," << k << "): ref=" << refJac(r, k) << " jit=" << jitVal);
                CHECK(jitVal == Catch::Approx(refJac(r, k)).epsilon(Eps));
            }
        }
    };

    SECTION("linear a*X1 + b")          { checkCostFn("1.5 * X1 + 2.0",             {1.5F, 2.0F}); }
    SECTION("quadratic a*X1^2 + b*X1 + c") { checkCostFn("1.0 * X1 * X1 + 0.5 * X1 + 3.0", {1.0F, 0.5F, 3.0F}); }
    SECTION("trig a*sin + b*cos")        { checkCostFn("1.0 * sin(X1) + 1.5 * cos(X2)",     {1.0F, 1.5F}); }
    SECTION("exp a*exp(b*X1)")           { checkCostFn("2.0 * exp(0.5 * X1)",                {2.0F, 0.5F}); }
    SECTION("composite")                 { checkCostFn("1.0 * X1 * X2 + 0.5 * sin(X3) + 2.0", {1.0F, 0.5F, 2.0F}); }
}

TEST_CASE("JitLMCostFunction respects consts parameter", "[jit][lm]")
{
    // Verify that Evaluate uses the supplied coefficient values, not the tree's
    // stored values.  Evaluating at two different parameter sets must produce
    // two different residual/Jacobian results.
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    Range const range{0, 100};

    Problem problem{&ds};
    problem.SetTarget("Y");
    problem.SetTrainingRange(range);

    auto const target = problem.TargetValues(range);
    DTable dtable;

    // Build tree: a * X1 + b, tree stores a=1.0, b=0.0
    auto tree = InfixParser::Parse("1.0 * X1 + 0.0", ds);
    for (auto& nd : tree.Nodes()) { nd.Optimize = nd.IsConstant(); }

    auto compiled    = compiler.CompileAVX2(tree);
    auto dag         = BuildJacobianDag(tree);
    auto compiledJac = compiler.CompileJacobian(dag);
    REQUIRE(compiled    != nullptr);
    REQUIRE(compiledJac != nullptr);

    auto const varOrder = JIT::VarOrder(tree);
    auto makeColPtrs = [&](std::vector<Operon::Hash> const& order) {
        std::vector<float const*> ptrs(order.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            ptrs[i] = ds.GetPaddedValues(order[i]) + range.Start();
        }
        return ptrs;
    };

    Interpreter<Operon::Scalar, DTable> interp{&dtable, &ds, &tree};

    JitLMCostFunction<> cf{
        gsl::not_null<InterpreterBase<Operon::Scalar> const*>{&interp},
        compiled->fn,
        makeColPtrs(varOrder),
        target, range,
        compiledJac->jacFn,
        makeColPtrs(varOrder)};

    auto const nRes = static_cast<std::size_t>(cf.NumResiduals());
    auto const nPar = static_cast<std::size_t>(cf.NumParameters());
    REQUIRE(nPar == 2);

    std::vector<Operon::Scalar> res1(nRes), res2(nRes);
    std::vector<Operon::Scalar> jac1(nRes * nPar), jac2(nRes * nPar);

    std::vector<Operon::Scalar> p1 = {1.0F,  0.0F};
    std::vector<Operon::Scalar> p2 = {2.0F, -1.0F};

    cf.Evaluate(p1.data(), res1.data(), jac1.data());
    cf.Evaluate(p2.data(), res2.data(), jac2.data());

    // The two residual vectors must differ (different a,b => different predictions)
    bool differ = false;
    for (std::size_t r = 0; r < nRes; ++r) {
        if (std::abs(res1[r] - res2[r]) > 1e-6F) { differ = true; break; }
    }
    CHECK(differ);

    // Verify each set against the interpreter
    auto check = [&](std::vector<Operon::Scalar>& p,
                     std::vector<Operon::Scalar>& res,
                     std::vector<Operon::Scalar>& jac) {
        auto predVec = interp.Evaluate(p, range);
        auto refJac  = interp.JacRev(p, range);

        Eigen::Map<const Eigen::Array<Operon::Scalar, -1, 1>> tgtArr(target.data(), static_cast<Eigen::Index>(nRes));
        Eigen::Map<const Eigen::Array<Operon::Scalar, -1, 1>> predArr(predVec.data(), static_cast<Eigen::Index>(nRes));
        auto refRes = (predArr - tgtArr).eval();

        constexpr float Eps = 1e-4F;
        for (std::size_t r = 0; r < nRes; ++r) {
            if (!std::isfinite(refRes(static_cast<Eigen::Index>(r)))) { continue; }
            CHECK(res[r] == Catch::Approx(refRes(static_cast<Eigen::Index>(r))).epsilon(Eps));
        }
        for (std::size_t k = 0; k < nPar; ++k) {
            for (std::size_t r = 0; r < nRes; ++r) {
                auto ei = static_cast<Eigen::Index>(r);
                auto ek = static_cast<Eigen::Index>(k);
                if (!std::isfinite(refJac(ei, ek))) { continue; }
                CHECK(jac[k * nRes + r] == Catch::Approx(refJac(ei, ek)).epsilon(Eps));
            }
        }
    };

    SECTION("at p1") { check(p1, res1, jac1); }
    SECTION("at p2") { check(p2, res2, jac2); }
}

TEST_CASE("JitLMCostFunction residuals only (no Jacobian)", "[jit][lm]")
{
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    Range const range{0, 100};

    Problem problem{&ds};
    problem.SetTarget("Y");
    problem.SetTrainingRange(range);

    auto const target = problem.TargetValues(range);
    DTable dtable;

    auto tree = InfixParser::Parse("1.5 * X1 + 2.0 * X2 + 0.5", ds);
    for (auto& nd : tree.Nodes()) { nd.Optimize = nd.IsConstant(); }

    auto compiled = compiler.CompileAVX2(tree);
    REQUIRE(compiled != nullptr);

    auto const varOrder = JIT::VarOrder(tree);
    std::vector<float const*> colPtrs(varOrder.size());
    for (std::size_t i = 0; i < colPtrs.size(); ++i) {
        colPtrs[i] = ds.GetPaddedValues(varOrder[i]) + range.Start();
    }

    Interpreter<Operon::Scalar, DTable> interp{&dtable, &ds, &tree};

    JitLMCostFunction<> cf{
        gsl::not_null<InterpreterBase<Operon::Scalar> const*>{&interp},
        compiled->fn,
        std::move(colPtrs),
        target, range};  // no Jacobian

    auto const nRes = static_cast<std::size_t>(cf.NumResiduals());
    std::vector<Operon::Scalar> evalCoeff = {1.5F, 2.0F, 0.5F};

    std::vector<Operon::Scalar> jitResiduals(nRes);
    bool ok = cf.Evaluate(evalCoeff.data(), jitResiduals.data(), nullptr);
    REQUIRE(ok);

    auto predVec = interp.Evaluate(evalCoeff, range);
    Eigen::Map<const Eigen::Array<Operon::Scalar, -1, 1>> tgtArr(target.data(), static_cast<Eigen::Index>(nRes));
    Eigen::Map<const Eigen::Array<Operon::Scalar, -1, 1>> predArr(predVec.data(), static_cast<Eigen::Index>(nRes));
    auto refRes = (predArr - tgtArr).eval();

    constexpr float Eps = 1e-4F;
    for (std::size_t r = 0; r < nRes; ++r) {
        auto ei = static_cast<Eigen::Index>(r);
        if (!std::isfinite(refRes(ei))) { continue; }
        INFO("row " << r << ": ref=" << refRes(ei) << " jit=" << jitResiduals[r]);
        CHECK(jitResiduals[r] == Catch::Approx(refRes(ei)).epsilon(Eps));
    }
}

TEST_CASE("JitLMCostFunction TinySolver convergence", "[jit][lm]")
{
    // Run TinySolver + JitLMCostFunction on a problem with a known answer and
    // verify the optimized coefficients are correct.
    JIT::JitRuntimePool compilerPool;
    JIT::TreeCompiler compiler{&compilerPool};
    if (!compiler.HasAVX2()) {
        SKIP("AVX2 not available");
    }

    // Build a simple dataset: X1 in [0,1], Y = 2*X1 + 3
    constexpr int NRows = 64;
    std::vector<Operon::Scalar> x1(NRows), y(NRows);
    for (int i = 0; i < NRows; ++i) {
        x1[i] = static_cast<Operon::Scalar>(i) / (NRows - 1);
        y[i]  = 2.0F * x1[i] + 3.0F;
    }
    Dataset ds({"X1", "Y"}, {x1, y});

    Range const range{0, NRows};

    Problem problem{&ds};
    problem.SetTarget("Y");
    problem.SetTrainingRange(range);

    auto const target = problem.TargetValues(range);
    DTable dtable;

    // Tree: a * X1 + b  (a starts at 1, b starts at 0)
    auto tree = InfixParser::Parse("1.0 * X1 + 0.0", ds);
    for (auto& nd : tree.Nodes()) { nd.Optimize = nd.IsConstant(); }

    auto compiled    = compiler.CompileAVX2(tree);
    auto dag         = BuildJacobianDag(tree);
    auto compiledJac = compiler.CompileJacobian(dag);
    REQUIRE(compiled    != nullptr);
    REQUIRE(compiledJac != nullptr);

    auto const varOrder = JIT::VarOrder(tree);
    auto makeColPtrs = [&](std::vector<Operon::Hash> const& order) {
        std::vector<float const*> ptrs(order.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            ptrs[i] = ds.GetPaddedValues(order[i]) + range.Start();
        }
        return ptrs;
    };

    Interpreter<Operon::Scalar, DTable> interp{&dtable, &ds, &tree};

    JitLMCostFunction<> cf{
        gsl::not_null<InterpreterBase<Operon::Scalar> const*>{&interp},
        compiled->fn,
        makeColPtrs(varOrder),
        target, range,
        compiledJac->jacFn,
        makeColPtrs(varOrder)};

    REQUIRE(cf.NumParameters() == 2);
    REQUIRE(cf.NumResiduals() == NRows);

    // Initial params: a=1, b=0
    std::vector<Operon::Scalar> x0 = {1.0F, 0.0F};
    Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));

    ceres::TinySolver<JitLMCostFunction<>> solver;
    solver.options.max_num_iterations = 100;

    typename decltype(solver)::Parameters p = m0.cast<Operon::Scalar>();
    solver.Solve(cf, &p);
    m0 = p.template cast<Operon::Scalar>();

    INFO("initial_cost=" << solver.summary.initial_cost
         << " final_cost=" << solver.summary.final_cost
         << " iterations=" << solver.summary.iterations);

    // Should converge to zero cost (perfect fit Y = 2*X1 + 3).
    // Parameter ordering depends on the tree's postfix representation; we don't
    // check specific p(i) values, just that the expression evaluates correctly.
    CHECK(solver.summary.final_cost < 1e-6F);

    // Verify by evaluating the expression with the optimized coefficients.
    m0 = p.template cast<Operon::Scalar>();  // already done above, but be explicit
    auto predVec = interp.Evaluate(x0, range);
    for (int i = 0; i < NRows; ++i) {
        INFO("row " << i);
        CHECK(predVec[i] == Catch::Approx(y[i]).epsilon(1e-3F));
    }
}

} // namespace Operon::Test

#endif // HAVE_ASMJIT
