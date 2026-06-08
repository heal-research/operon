// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifdef HAVE_ASMJIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <vector>

#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {

namespace {

using DTable = DispatchTable<Operon::Scalar>;

// Evaluate tree via a compiled function; returns vector of nRows results.
auto EvalCompiled(JIT::CompiledTree const& compiled,
                  Operon::Tree const& tree,
                  Operon::Dataset const& ds, Operon::Range range) -> std::vector<float>
{
    std::vector<float const*> colPtrs(compiled.varOrder.size());
    for (std::size_t i = 0; i < compiled.varOrder.size(); ++i) {
        colPtrs[i] = ds.GetValues(compiled.varOrder[i]).data() + range.Start();
    }
    auto coeff = tree.GetCoefficients();
    std::vector<float> out(range.Size());
    compiled.fn(out.data(), colPtrs.data(), static_cast<int32_t>(range.Size()),
                coeff.empty() ? nullptr : coeff.data());
    return out;
}

// Evaluate tree via JIT; returns vector of nRows results.
auto EvalJIT(JIT::TreeCompiler& compiler, Operon::Tree const& tree,
             Operon::Dataset const& ds, Operon::Range range) -> std::vector<float>
{
    auto compiled = compiler.Compile(tree);
    REQUIRE(compiled != nullptr);
    REQUIRE(compiled->fn != nullptr);
    return EvalCompiled(*compiled, tree, ds, range);
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

TEST_CASE("JIT scalar correctness", "[jit]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, std::min(ds.Rows<std::size_t>(), std::size_t{200})};

    JIT::TreeCompiler compiler;

    auto check = [&](std::string_view expr) {
        INFO("expression: " << expr);
        auto tree = InfixParser::Parse(std::string(expr), ds);
        auto ref  = EvalRef(tree, ds, range);
        auto jit  = EvalJIT(compiler, tree, ds, range);

        REQUIRE(ref.size() == jit.size());
        for (std::size_t i = 0; i < ref.size(); ++i) {
            INFO("row " << i << ": ref=" << ref[i] << " jit=" << jit[i]);
            if (std::isnan(ref[i])) {
                CHECK(std::isnan(jit[i]));
            } else if (std::isinf(ref[i])) {
                CHECK(std::isinf(jit[i]));
                CHECK((ref[i] > 0) == (jit[i] > 0));
            } else {
                CHECK(jit[i] == Catch::Approx(ref[i]).epsilon(Tol));
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
    SECTION("Constant")    { check("X1 + X2"); }
    SECTION("Sin")         { check("sin(X1)"); }
    SECTION("Cos")         { check("cos(X1)"); }
    SECTION("Exp")         { check("exp(X1)"); }
    SECTION("Log")         { check("log(X1)"); }
    SECTION("Logabs")      { check("log(abs(X1))"); }
    SECTION("Sqrt")        { check("sqrt(X1 * X1)"); }
    SECTION("Sqrtabs")     { check("sqrt(abs(X1))"); }
    SECTION("Abs")         { check("abs(X1 - X2)"); }
    SECTION("Tanh")        { check("tanh(X1)"); }
    SECTION("Aq")          { check("X1 / sqrt(1 + X2 * X2)"); }
    SECTION("Composite")   { check("sin(X1) * cos(X2) + exp(0 - X3 * X3)"); }
}

TEST_CASE("JIT AVX2 correctness", "[jit][avx2]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    // Use a row count that is not a multiple of 8 to exercise the tail loop.
    auto range = Range{0, 201};

    JIT::TreeCompiler compiler;

    // Skip if AVX2 is not available on this CPU.
    if (!compiler.Runtime().cpu_features().x86().has(asmjit::CpuFeatures::X86::kAVX2)) {
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
    SECTION("Exp")         { check("exp(X1)"); }
    SECTION("Log")         { check("log(X1)"); }
    SECTION("Logabs")      { check("log(abs(X1))"); }
    SECTION("Tanh")        { check("tanh(X1)"); }
    SECTION("Aq")          { check("X1 / sqrt(1 + X2 * X2)"); }
    SECTION("Composite")   { check("sin(X1) * cos(X2) + exp(0 - X3 * X3)"); }
    // Rows=201: 25 full AVX2 iterations (200 rows) + 1 scalar tail row
    SECTION("Tail rows")   { check("X1 * X2 + X3"); }
}

} // namespace Operon::Test

#endif // HAVE_ASMJIT
