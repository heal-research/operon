// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <fmt/core.h>
#include <array>
#include <numbers>
#include <random>
#include <vector>

#include "../operon_test.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"

#ifdef HAVE_ASMJIT
#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#include "operon/optimizer/jit_lm_cost_function.hpp"
#endif

namespace nb = ankerl::nanobench;

namespace {
// Friedman-I synthetic dataset: 5 input variables + target
auto MakeFriedmanDataset(int rows, std::mt19937& rng) -> Operon::Dataset
{
    constexpr int NVARS = 5;
    std::uniform_real_distribution<Operon::Scalar> u(0.F, 1.F);
    std::vector<std::vector<Operon::Scalar>> cols(NVARS + 1, std::vector<Operon::Scalar>(rows));
    for (int r = 0; r < rows; ++r) {
        std::array<Operon::Scalar, NVARS> x{};
        for (int i = 0; i < NVARS; ++i) { x[i] = u(rng); cols[i][r] = x[i]; }  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        cols[NVARS][r] = (10.F * std::sin(std::numbers::pi_v<Operon::Scalar> * x[0] * x[1]))  // NOLINT
                       + (20.F * (x[2] - .5F) * (x[2] - .5F))
                       + (10.F * x[3]) + (5.F * x[4]);
    }
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(NVARS + 1));
    for (int i = 0; i < NVARS; ++i) { names.push_back(fmt::format("x{}", i + 1)); }
    names.push_back("Y");
    return Operon::Dataset(names, cols);
}
} // namespace

// Measures first-compilation cost per tree as a function of tree size.
// nanobench CSV: title="compile", name="size=NN", batch=1 → elapsed = seconds per compilation.
TEST_CASE("JIT compile time distribution", "[performance][jit][breakeven]")
{
#ifndef HAVE_ASMJIT
    SKIP("Built without HAVE_ASMJIT");
#endif

    constexpr int MAX_SIZE   = 50;
    constexpr int N_PER_SIZE = 200;

    std::mt19937 rng(42);  // NOLINT(cert-msc51-cpp)
    auto ds = MakeFriedmanDataset(1000, rng);

    auto inputs = ds.VariableHashes();
    inputs.pop_back();

    Operon::Problem problem{&ds};
    problem.SetTrainingRange({0, 1000});
    problem.SetTarget("Y");
    problem.ConfigurePrimitiveSet(Operon::PrimitiveSet::Arithmetic);

    Operon::RandomGenerator zrng(42);  // NOLINT(cert-msc51-cpp)
    Operon::JIT::JitZobrist zobrist(zrng, MAX_SIZE, inputs);

#ifdef HAVE_ASMJIT
    Operon::JIT::JitEvaluator jitEval(&problem, &zobrist, Operon::MSE{}, /*linearScaling=*/false);
    jitEval.SetBudget(std::numeric_limits<std::size_t>::max());

    Operon::BalancedTreeCreator creator(&problem.GetPrimitiveSet(), inputs, 0.0, MAX_SIZE);
    Operon::RandomGenerator treeRng(123);  // NOLINT(cert-msc51-cpp)

    nb::Bench bench;
    bench.title("compile").batch(1);

    for (int sz = 1; sz <= MAX_SIZE; ++sz) {
        std::vector<Operon::Tree> trees;
        trees.reserve(N_PER_SIZE);
        for (int k = 0; k < N_PER_SIZE; ++k) {
            trees.push_back(creator(treeRng, static_cast<std::size_t>(sz), 1, 100));
        }

        int idx = 0;
        bench.run(fmt::format("size={:02d}", sz), [&] {
            jitEval.ClearCache();
            nb::doNotOptimizeAway(jitEval.GetOrCompile(trees[idx++ % N_PER_SIZE]));
        });
    }

    bench.render(nb::templates::csv(), std::cout);
#endif
}

// For each tree size in SIZES, measures total time vs rows for:
//   - interpreter: one Evaluate call
//   - JIT cold:    ClearCache + GetOrCompile + one Evaluate call
// batch=1 → CSV elapsed = total seconds per operation, directly comparable.
// The crossover row count is the JIT break-even point for that tree size.
TEST_CASE("JIT vs interpreter break-even", "[performance][jit][breakeven]")
{
    constexpr int MAX_ROWS    = 100'000;
    constexpr int POOL_SIZE   = 20;   // trees per size, rotated to avoid compile-cache warmth
    constexpr std::array SIZES{ 10, 25, 50 };
    constexpr int MAX_SIZE    = SIZES.back();

    std::mt19937 rng(42);  // NOLINT(cert-msc51-cpp)
    auto ds = MakeFriedmanDataset(MAX_ROWS, rng);

    auto inputs = ds.VariableHashes();
    inputs.pop_back();

    Operon::Problem problem{&ds};
    problem.SetTrainingRange({0, static_cast<std::size_t>(MAX_ROWS)});
    problem.SetTarget("Y");
    problem.ConfigurePrimitiveSet(Operon::PrimitiveSet::Arithmetic);

    Operon::RandomGenerator zrng(42);   // NOLINT(cert-msc51-cpp)
    Operon::JIT::JitZobrist zobrist(zrng, MAX_SIZE, inputs);

    Operon::BalancedTreeCreator creator(&problem.GetPrimitiveSet(), inputs, 0.0, MAX_SIZE);
    Operon::RandomGenerator treeRng(456);  // NOLINT(cert-msc51-cpp)

    Operon::ScalarDispatch dtable;

    auto const nRowsPadMax = static_cast<int>((static_cast<unsigned>(MAX_ROWS) + 7U) & ~7U);
    std::vector<Operon::Scalar> scratch(static_cast<std::size_t>(nRowsPadMax));
    std::vector<Operon::Scalar> buf(static_cast<std::size_t>(MAX_ROWS));

#ifdef HAVE_ASMJIT
    Operon::JIT::JitEvaluator jitEval(&problem, &zobrist, Operon::MSE{}, /*linearScaling=*/false);
    jitEval.SetBudget(std::numeric_limits<std::size_t>::max());
#endif

    nb::Bench bench;
    bench.title("breakeven").batch(1);

    for (int sz : SIZES) {
        // Generate a pool of trees of this size
        std::vector<Operon::Tree> pool;
        pool.reserve(POOL_SIZE);
        for (int k = 0; k < POOL_SIZE; ++k) {
            pool.push_back(creator(treeRng, static_cast<std::size_t>(sz), 1, 100));
        }

        std::vector<std::vector<Operon::Scalar>> poolCoeffs(POOL_SIZE);
        for (int k = 0; k < POOL_SIZE; ++k) { pool[k].GetCoefficients(poolCoeffs[k]); }

        // Interpreter objects (one per tree in pool)
        std::vector<Operon::Interpreter<Operon::Scalar, Operon::ScalarDispatch>> interps;
        interps.reserve(POOL_SIZE);
        for (auto const& t : pool) { interps.emplace_back(&dtable, &ds, &t); }

#ifdef HAVE_ASMJIT
        // Pre-build column pointer arrays using VarOrder (no compilation needed).
        struct ColPtrs { std::vector<Operon::Scalar const*> ptrs; };
        std::vector<ColPtrs> colPtrPool(POOL_SIZE);
        for (int k = 0; k < POOL_SIZE; ++k) {
            auto const order = Operon::JIT::VarOrder(pool[k]);
            colPtrPool[k].ptrs.resize(order.size());
            for (std::size_t j = 0; j < order.size(); ++j) {
                colPtrPool[k].ptrs[j] = ds.GetPaddedValues(order[j]);
            }
        }
#endif

        int idx = 0;

        for (int rows = 1000; rows <= MAX_ROWS; rows += 1000) {
            Operon::Range const range{0, static_cast<std::size_t>(rows)};

            bench.run(fmt::format("interp sz={:02d} rows={:06d}", sz, rows), [&] {
                int const i = idx % POOL_SIZE;
                Operon::Span<Operon::Scalar const> cs{poolCoeffs[i].data(), poolCoeffs[i].size()};
                interps[i].Evaluate(cs, range, buf);
                nb::doNotOptimizeAway(buf.data());
            });

#ifdef HAVE_ASMJIT
            auto const nRowsPad = static_cast<int>((static_cast<unsigned>(rows) + 7U) & ~7U);
            bench.run(fmt::format("jit sz={:02d} rows={:06d}", sz, rows), [&] {
                int const i = idx % POOL_SIZE;
                jitEval.ClearCache();
                auto const* c = jitEval.GetOrCompile(pool[i]);
                if (c != nullptr) {
                    float const* cp = poolCoeffs[i].empty() ? nullptr : poolCoeffs[i].data();
                    c->fn(scratch.data(), colPtrPool[i].ptrs.data(), nRowsPad, cp);
                }
                nb::doNotOptimizeAway(scratch.data());
                ++idx;
            });
#endif
        }
    }

    bench.render(nb::templates::csv(), std::cout);
}

// Measures Jacobian break-even: compile_jac + jac_eval_JIT vs JacRev (interpreter).
// Each JIT iteration: ClearCache + GetOrCompileJacobian + one jacFn call.
// Each interp iteration: one JacRev call (reverse-mode AD, same coefficients count).
// batch=1 → elapsed = total seconds per operation.
TEST_CASE("JIT vs interpreter Jacobian break-even", "[performance][jit][breakeven]")
{
#ifndef HAVE_ASMJIT
    SKIP("Built without HAVE_ASMJIT");
#endif

    constexpr int MAX_ROWS  = 100'000;
    constexpr int POOL_SIZE = 20;
    constexpr std::array SIZES{ 10, 25, 50 };
    constexpr int MAX_SIZE  = SIZES.back();

    std::mt19937 rng(42);  // NOLINT(cert-msc51-cpp)
    auto ds = MakeFriedmanDataset(MAX_ROWS, rng);

    auto inputs = ds.VariableHashes();
    inputs.pop_back();

    Operon::Problem problem{&ds};
    problem.SetTrainingRange({0, static_cast<std::size_t>(MAX_ROWS)});
    problem.SetTarget("Y");
    problem.ConfigurePrimitiveSet(Operon::PrimitiveSet::Arithmetic);

    Operon::RandomGenerator zrng(42);  // NOLINT(cert-msc51-cpp)
    Operon::JIT::JitZobrist zobrist(zrng, MAX_SIZE, inputs);

    Operon::BalancedTreeCreator creator(&problem.GetPrimitiveSet(), inputs, 0.0, MAX_SIZE);
    Operon::RandomGenerator treeRng(789);  // NOLINT(cert-msc51-cpp)

    Operon::ScalarDispatch dtable;

    auto const nRowsPadMax = static_cast<std::size_t>((static_cast<unsigned>(MAX_ROWS) + 7U) & ~7U);

    Operon::JIT::JitEvaluator jitEval(&problem, &zobrist, Operon::MSE{}, /*linearScaling=*/false);
    jitEval.SetBudget(std::numeric_limits<std::size_t>::max());

    nb::Bench bench;
    bench.title("jac-breakeven").batch(1);

    for (int sz : SIZES) {
        std::vector<Operon::Tree> pool;
        pool.reserve(POOL_SIZE);
        for (int k = 0; k < POOL_SIZE; ++k) {
            pool.push_back(creator(treeRng, static_cast<std::size_t>(sz), 1, 100));
        }

        std::vector<std::vector<Operon::Scalar>> poolCoeffs(POOL_SIZE);
        for (int k = 0; k < POOL_SIZE; ++k) { pool[k].GetCoefficients(poolCoeffs[k]); }

        // Build interpreters and pre-discover jacColPtrs for JIT path.
        std::vector<Operon::Interpreter<Operon::Scalar, Operon::ScalarDispatch>> interps;
        interps.reserve(POOL_SIZE);
        for (auto const& t : pool) { interps.emplace_back(&dtable, &ds, &t); }

        struct JacColPtrs { std::vector<Operon::Scalar const*> ptrs; };
        std::vector<JacColPtrs> jacColPtrPool(POOL_SIZE);
        for (int k = 0; k < POOL_SIZE; ++k) {
            auto const order = Operon::JIT::VarOrder(pool[k]);
            jacColPtrPool[k].ptrs.resize(order.size());
            for (std::size_t j = 0; j < order.size(); ++j) {
                jacColPtrPool[k].ptrs[j] = ds.GetPaddedValues(order[j]);
            }
        }

        int idx = 0;

        for (int rows = 1000; rows <= MAX_ROWS; rows += 1000) {
            Operon::Range const range{0, static_cast<std::size_t>(rows)};
            auto const nRowsPad = static_cast<int>((static_cast<unsigned>(rows) + 7U) & ~7U);
            auto const nCoeffs  = static_cast<std::size_t>(pool[0].CoefficientsCount());

            // Allocate Jacobian output buffers (nRows × nCoeffs, col-major).
            std::vector<Operon::Scalar>              jacBufInterp(static_cast<std::size_t>(rows) * nCoeffs);
            std::vector<std::vector<Operon::Scalar>> jacColBufsJit(nCoeffs,
                std::vector<Operon::Scalar>(nRowsPadMax));
            std::vector<float*> jacOutPtrs(nCoeffs);
            for (std::size_t k = 0; k < nCoeffs; ++k) { jacOutPtrs[k] = jacColBufsJit[k].data(); }

            bench.run(fmt::format("jacrev sz={:02d} rows={:06d}", sz, rows), [&] {
                int const i = idx % POOL_SIZE;
                Operon::Span<Operon::Scalar const> cs{ poolCoeffs[i].data(), poolCoeffs[i].size() };
                Operon::Span<Operon::Scalar> jac{ jacBufInterp.data(), jacBufInterp.size() };
                interps[i].JacRev(cs, range, jac);
                nb::doNotOptimizeAway(jacBufInterp.data());
            });

            bench.run(fmt::format("jit_jac sz={:02d} rows={:06d}", sz, rows), [&] {
                int const i = idx % POOL_SIZE;
                jitEval.ClearCache();
                auto const* jac = jitEval.GetOrCompileJacobian(pool[i]);
                if (jac != nullptr && jac->jacFn != nullptr) {
                    float const* cp = poolCoeffs[i].empty() ? nullptr : poolCoeffs[i].data();
                    jac->jacFn(jacOutPtrs.data(), jacColPtrPool[i].ptrs.data(), nRowsPad, cp);
                }
                nb::doNotOptimizeAway(jacOutPtrs[0]);
                ++idx;
            });
        }
    }

    bench.render(nb::templates::csv(), std::cout);
}
