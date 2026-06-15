// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifdef HAVE_ASMJIT

#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/core/node.hpp"

#include <asmjit/asmjit.h>
#include <asmjit/x86.h>

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

// Fast polynomial variants matching the interpreter's dispatch table.
// Must be included AFTER eve headers so the templates can instantiate.
#include "operon/interpreter/backend/eve/functions.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace Operon::JIT {

namespace {

using namespace asmjit;
using namespace asmjit::x86;

// Scalar wrappers using EVE functions — match the AVX2 path exactly.
// EVE functions are CPOs, not plain function pointers, so each needs a
// thin wrapper to produce a concrete address for invokeF1's call instruction.
static float scalar_abs  (float x) noexcept { return eve::abs  (x); }
static float scalar_acos (float x) noexcept { return eve::acos (x); }
static float scalar_asin (float x) noexcept { return eve::asin (x); }
static float scalar_atan (float x) noexcept { return eve::atan (x); }
static float scalar_cbrt (float x) noexcept { return eve::cbrt (x); }
static float scalar_cos  (float x) noexcept { return eve::cos  (x); }
static float scalar_cosh (float x) noexcept { return eve::cosh (x); }
static float scalar_exp  (float x) noexcept {
    constexpr float MaxExp = 88.72283172607421875F;
    return eve::exp(eve::clamp(x, -MaxExp, MaxExp));  // matches FastExp
}
static float scalar_log  (float x) noexcept { return eve::log(x); }
static float scalar_logabs(float x) noexcept { return eve::log(eve::abs(x)); }
static float scalar_log1p(float x) noexcept { return eve::log1p(x); }
static float scalar_sin  (float x) noexcept { return eve::sin  (x); }
static float scalar_sinh (float x) noexcept { return eve::sinh (x); }
static float scalar_tan  (float x) noexcept { return eve::tan  (x); }
static float scalar_tanh (float x) noexcept {
    constexpr float MaxTanh = 7.99881172180175781F;
    return eve::tanh(eve::clamp(x, -MaxTanh, MaxTanh));  // matches FastTanh
}

// Match FastPow semantics: NaN for negative base + non-integer exponent,
// sign flip for negative base + odd integer exponent, 0^pos=0, x^0=1.
static float scalar_pow(float x, float y) noexcept {
    if (y == 0.0F)              return 1.0F;
    if (x == 0.0F && y > 0.0F) return 0.0F;
    constexpr float MaxExp = 88.72283172607421875F;
    float z = eve::exp(eve::clamp(y * eve::log(eve::abs(x)), -MaxExp, MaxExp));
    if (x < 0.0F) {
        if (!eve::is_flint(y)) return std::numeric_limits<float>::quiet_NaN();
        if (eve::is_odd(y))    z = -z;
    }
    return z;
}

// Invoke a scalar float→float C function.
auto invokeF1(Compiler& cc, void* fn_ptr, Vec arg) -> Vec {
    Vec result = cc.new_xmm_ss();
    InvokeNode* inv{};
    cc.invoke(Out(inv),
              reinterpret_cast<uint64_t>(fn_ptr),
              FuncSignature::build<float, float>());
    inv->set_arg(0, arg);
    inv->set_ret(0, result);
    return result;
}

// Invoke scalar_pow(a, b) — matches interpreter's FastPow semantics.
auto invokePowf(Compiler& cc, Vec a, Vec b) -> Vec {
    Vec result = cc.new_xmm_ss();
    InvokeNode* inv{};
    cc.invoke(Out(inv),
              reinterpret_cast<uint64_t>(reinterpret_cast<void*>(scalar_pow)),
              FuncSignature::build<float, float, float>());
    inv->set_arg(0, a);
    inv->set_arg(1, b);
    inv->set_ret(0, result);
    return result;
}

// Load a compile-time float constant into a fresh xmm_ss register.
auto loadFloat(Compiler& cc, float val) -> Vec {
    uint32_t bits{};
    std::memcpy(&bits, &val, sizeof bits);
    Gp tmp = cc.new_gp32();
    cc.mov(tmp, bits);
    Vec reg = cc.new_xmm_ss();
    cc.vmovd(reg, tmp);
    return reg;
}

// Emit scalar (xmm_ss) evaluation of `nodes[start..end)` into `stack`.
// `nodeVecs[i]` is filled with the result Vec for each processed node i.
// `constIdx` tracks the next coefficient index across calls.
// All indices into nodeVecs use the global dag index (same as the position in nodes[]).
void emitNodesScalar(
    Compiler& cc,
    std::vector<Node> const& nodes,
    std::size_t start,
    std::size_t end,
    std::vector<Gp> const& colPtrs,
    std::vector<Vec> const& coeffRegs,
    std::vector<Operon::Hash> const& varOrder,
    Gp row,
    std::vector<Vec>& stack,
    std::vector<Vec>& nodeVecs,
    int& constIdx)
{
    for (std::size_t ii = start; ii < end; ++ii) {
        auto const& n = nodes[ii];

        if (n.IsRef()) {
            Vec v = nodeVecs[n.RefTo];
            nodeVecs[ii] = v;
            stack.push_back(v);
            continue;
        }

        if (n.IsVariable()) {
            auto it = std::find(varOrder.begin(), varOrder.end(), n.HashValue);
            auto colIdx = static_cast<std::size_t>(std::distance(varOrder.begin(), it));
            Vec val = cc.new_xmm_ss();
            cc.vmovss(val, x86::ptr(colPtrs[colIdx], row, 2));
            Vec weighted;
            if (n.Optimize) {
                weighted = cc.new_xmm_ss();
                cc.vmulss(weighted, val, coeffRegs[static_cast<std::size_t>(constIdx++)]);
            } else if (n.Value != 1.0f) {
                Vec w = loadFloat(cc, n.Value);
                weighted = cc.new_xmm_ss();
                cc.vmulss(weighted, val, w);
            } else {
                weighted = val;
            }
            nodeVecs[ii] = weighted;
            stack.push_back(weighted);
            continue;
        }

        if (n.IsConstant()) {
            Vec val;
            if (n.Optimize) {
                val = coeffRegs[static_cast<std::size_t>(constIdx++)];
            } else {
                val = loadFloat(cc, n.Value);
            }
            nodeVecs[ii] = val;
            stack.push_back(val);
            continue;
        }

        // Operator node.
        auto arity = n.Arity;
        std::vector<Vec> args(static_cast<std::size_t>(arity));
        for (int k = 0; k < arity; ++k) {
            args[static_cast<std::size_t>(k)] = stack.back();
            stack.pop_back();
        }

        Vec res;
        switch (n.Type) {
        case NodeType::Add: {
            res = args[0];
            for (int k = 1; k < arity; ++k) {
                Vec tmp = cc.new_xmm_ss();
                cc.vaddss(tmp, res, args[static_cast<std::size_t>(k)]);
                res = tmp;
            }
            break;
        }
        case NodeType::Mul: {
            res = args[0];
            for (int k = 1; k < arity; ++k) {
                Vec tmp = cc.new_xmm_ss();
                cc.vmulss(tmp, res, args[static_cast<std::size_t>(k)]);
                res = tmp;
            }
            break;
        }
        case NodeType::Sub: {
            if (arity == 1) {
                Vec zero = loadFloat(cc, 0.0f);
                res = cc.new_xmm_ss();
                cc.vsubss(res, zero, args[0]);
            } else {
                res = cc.new_xmm_ss();
                cc.vsubss(res, args[0], args[1]);
                for (int k = 2; k < arity; ++k) {
                    Vec tmp = cc.new_xmm_ss();
                    cc.vsubss(tmp, res, args[static_cast<std::size_t>(k)]);
                    res = tmp;
                }
            }
            break;
        }
        case NodeType::Div: {
            if (arity == 1) {
                Vec one = loadFloat(cc, 1.0f);
                res = cc.new_xmm_ss();
                cc.vdivss(res, one, args[0]);
            } else {
                res = cc.new_xmm_ss();
                cc.vdivss(res, args[0], args[1]);
                for (int k = 2; k < arity; ++k) {
                    Vec tmp = cc.new_xmm_ss();
                    cc.vdivss(tmp, res, args[static_cast<std::size_t>(k)]);
                    res = tmp;
                }
            }
            break;
        }
        case NodeType::Fmin: {
            res = args[0];
            for (int k = 1; k < arity; ++k) {
                Vec tmp = cc.new_xmm_ss();
                cc.vminss(tmp, res, args[static_cast<std::size_t>(k)]);
                res = tmp;
            }
            break;
        }
        case NodeType::Fmax: {
            res = args[0];
            for (int k = 1; k < arity; ++k) {
                Vec tmp = cc.new_xmm_ss();
                cc.vmaxss(tmp, res, args[static_cast<std::size_t>(k)]);
                res = tmp;
            }
            break;
        }
        case NodeType::Aq: {
            Vec b2  = cc.new_xmm_ss(); cc.vmulss(b2, args[1], args[1]);
            Vec one = loadFloat(cc, 1.0f);
            Vec sum = cc.new_xmm_ss(); cc.vaddss(sum, one, b2);
            Vec sq  = cc.new_xmm_ss(); cc.vsqrtss(sq, sq, sum);
            res = cc.new_xmm_ss(); cc.vdivss(res, args[0], sq);
            break;
        }
        case NodeType::Pow: {
            res = invokePowf(cc, args[0], args[1]);
            break;
        }
        case NodeType::Powabs: {
            Vec absA = invokeF1(cc, reinterpret_cast<void*>(scalar_abs), args[0]);
            res = invokePowf(cc, absA, args[1]);
            break;
        }
        case NodeType::Abs:    { res = invokeF1(cc, reinterpret_cast<void*>(scalar_abs),  args[0]); break; }
        case NodeType::Acos:   { res = invokeF1(cc, reinterpret_cast<void*>(scalar_acos), args[0]); break; }
        case NodeType::Asin:   { res = invokeF1(cc, reinterpret_cast<void*>(scalar_asin), args[0]); break; }
        case NodeType::Atan:   { res = invokeF1(cc, reinterpret_cast<void*>(scalar_atan), args[0]); break; }
        case NodeType::Cbrt:   { res = invokeF1(cc, reinterpret_cast<void*>(scalar_cbrt), args[0]); break; }
        case NodeType::Ceil: {
            res = cc.new_xmm_ss();
            cc.vroundss(res, args[0], args[0],
                        Imm(static_cast<uint8_t>(RoundImm::kUp) |
                            static_cast<uint8_t>(RoundImm::kSuppress)));
            break;
        }
        case NodeType::Cos:    { res = invokeF1(cc, reinterpret_cast<void*>(scalar_cos),  args[0]); break; }
        case NodeType::Cosh:   { res = invokeF1(cc, reinterpret_cast<void*>(scalar_cosh), args[0]); break; }
        case NodeType::Exp:    { res = invokeF1(cc, reinterpret_cast<void*>(scalar_exp),  args[0]); break; }
        case NodeType::Floor: {
            res = cc.new_xmm_ss();
            cc.vroundss(res, args[0], args[0],
                        Imm(static_cast<uint8_t>(RoundImm::kDown) |
                            static_cast<uint8_t>(RoundImm::kSuppress)));
            break;
        }
        case NodeType::Log:    { res = invokeF1(cc, reinterpret_cast<void*>(scalar_log),    args[0]); break; }
        case NodeType::Logabs: { res = invokeF1(cc, reinterpret_cast<void*>(scalar_logabs), args[0]); break; }
        case NodeType::Log1p:  { res = invokeF1(cc, reinterpret_cast<void*>(scalar_log1p), args[0]); break; }
        case NodeType::Sin:    { res = invokeF1(cc, reinterpret_cast<void*>(scalar_sin),   args[0]); break; }
        case NodeType::Sinh:   { res = invokeF1(cc, reinterpret_cast<void*>(scalar_sinh),  args[0]); break; }
        case NodeType::Sqrt: {
            res = cc.new_xmm_ss();
            cc.vsqrtss(res, args[0], args[0]);
            break;
        }
        case NodeType::Sqrtabs: {
            Vec absVal = invokeF1(cc, reinterpret_cast<void*>(scalar_abs), args[0]);
            res = cc.new_xmm_ss();
            cc.vsqrtss(res, absVal, absVal);
            break;
        }
        case NodeType::Tan:    { res = invokeF1(cc, reinterpret_cast<void*>(scalar_tan),  args[0]); break; }
        case NodeType::Tanh:   { res = invokeF1(cc, reinterpret_cast<void*>(scalar_tanh), args[0]); break; }
        case NodeType::Square: {
            res = cc.new_xmm_ss();
            cc.vmulss(res, args[0], args[0]);
            break;
        }
        default: {
            res = loadFloat(cc, 0.0f);
            break;
        }
        }

        nodeVecs[ii] = res;
        stack.push_back(res);
    }
}

} // namespace

auto TreeCompiler::Compile(Operon::Tree const& tree) -> std::unique_ptr<CompiledTree> {
    auto result = std::make_unique<CompiledTree>(rt_);

    auto const& nodes = tree.Nodes();
    for (auto const& n : nodes) {
        if (!n.IsVariable()) { continue; }
        auto h = n.HashValue;
        if (std::find(result->varOrder.begin(), result->varOrder.end(), h) == result->varOrder.end()) {
            result->varOrder.push_back(h);
        }
    }

    CodeHolder code;
    code.init(rt_.environment(), rt_.cpu_features());
    Compiler cc(&code);

    // void fn(float* out, float const* const* cols, int32_t nRows, float const* consts)
    FuncNode* fnNode = cc.add_func(
        FuncSignature::build<void, float*, float const* const*, int32_t, float const*>());

    Gp outPtr    = cc.new_gp_ptr("out");
    Gp colsPtr   = cc.new_gp_ptr("cols");
    Gp nRowsArg  = cc.new_gp32("nRows");
    Gp constsPtr = cc.new_gp_ptr("consts");

    fnNode->set_arg(0, outPtr);
    fnNode->set_arg(1, colsPtr);
    fnNode->set_arg(2, nRowsArg);
    fnNode->set_arg(3, constsPtr);

    std::vector<Gp> colPtrs(result->varOrder.size());
    for (std::size_t i = 0; i < result->varOrder.size(); ++i) {
        colPtrs[i] = cc.new_gp_ptr();
        cc.mov(colPtrs[i], x86::ptr(colsPtr, static_cast<int32_t>(i * sizeof(void*))));
    }

    int nConsts = 0;
    for (auto const& n : nodes) {
        if (n.Optimize) { ++nConsts; }
    }
    std::vector<Vec> coeffRegs(static_cast<std::size_t>(nConsts));
    for (int j = 0; j < nConsts; ++j) {
        coeffRegs[static_cast<std::size_t>(j)] = cc.new_xmm_ss();
        cc.vmovss(coeffRegs[static_cast<std::size_t>(j)],
                  x86::ptr(constsPtr, static_cast<int32_t>(j * static_cast<int>(sizeof(float)))));
    }

    Gp row = cc.new_gp64("row");
    cc.xor_(row.r32(), row.r32());

    Label loopBegin = cc.new_label();
    Label loopEnd   = cc.new_label();

    cc.bind(loopBegin);
    cc.cmp(row.r32(), nRowsArg);
    cc.jge(loopEnd);

    {
        std::vector<Vec> stack;
        std::vector<Vec> nodeVecs(nodes.size());
        int constIdx = 0;
        stack.reserve(32);
        emitNodesScalar(cc, nodes, 0, nodes.size(), colPtrs, coeffRegs, result->varOrder, row, stack, nodeVecs, constIdx);
        cc.vmovss(x86::ptr(outPtr, row, 2), stack.back());
    }

    cc.inc(row);
    cc.jmp(loopBegin);
    cc.bind(loopEnd);

    cc.ret();
    cc.end_func();

    if (auto err = cc.finalize(); err != Error::kOk) {
        return nullptr;
    }

    EvalFn fn_ptr = nullptr;
    if (auto err = rt_.add(&fn_ptr, &code); err != Error::kOk) {
        return nullptr;
    }

    result->fn = fn_ptr;
    return result;
}

// =============================================================================
// Phase 2 — AVX2 vectorized compiler
// =============================================================================

namespace {

// In-place float[8] transcendental helpers — eve-backed AVX2 SIMD.
// Signature: void fn(float* p) — operates on exactly 8 floats in-place.
// Called from CompileAVX2 JIT code via a 32-byte aligned stack buffer.
// Compiled with -mavx2 so eve::wide<float, eve::fixed<8>> maps to a single ymm register.
using W8 = eve::wide<float, eve::fixed<8>>;
static void vec_fabsf (float* p) { eve::store(eve::abs  (W8{p}), p); }
static void vec_acosf (float* p) { eve::store(eve::acos (W8{p}), p); }
static void vec_asinf (float* p) { eve::store(eve::asin (W8{p}), p); }
static void vec_atanf (float* p) { eve::store(eve::atan (W8{p}), p); }
static void vec_cbrtf (float* p) { eve::store(eve::cbrt (W8{p}), p); }
static void vec_cosf  (float* p) { eve::store(Operon::Backend::detail::FastSinCos<false, float>(W8{p}), p); }
static void vec_coshf (float* p) { eve::store(eve::cosh (W8{p}), p); }
static void vec_expf  (float* p) { eve::store(Operon::Backend::detail::FastExp<float>  (W8{p}), p); }
static void vec_logf  (float* p) { eve::store(Operon::Backend::detail::FastLog<float>  (W8{p}), p); }
static void vec_log1pf(float* p) { eve::store(eve::log1p(W8{p}), p); }
static void vec_sinf  (float* p) { eve::store(Operon::Backend::detail::FastSinCos<true,  float>(W8{p}), p); }
static void vec_sinhf (float* p) { eve::store(eve::sinh (W8{p}), p); }
static void vec_tanf  (float* p) { eve::store(eve::tan  (W8{p}), p); }
static void vec_tanhf (float* p) { eve::store(Operon::Backend::detail::FastTanh<float>(W8{p}), p); }
// Match interpreter's FastPow exactly: NaN for negative base with non-integer exponent,
// sign correction for negative base with odd integer exponent, 0^0=1, x^0=1.
static void vec_powf  (float* p, float const* q) { eve::store(Operon::Backend::detail::FastPow<float>(W8{p}, W8{q}), p); }

// Apply a void(float*) helper to a ymm register via a 32-byte aligned stack buffer.
// Returns the new ymm register containing the results.
auto invokeF1_ps(Compiler& cc, void* fn_ptr, Vec arg) -> Vec {
    Mem scratch = cc.new_stack(32, 32);
    cc.vmovaps(scratch, arg);
    Gp ptr = cc.new_gp_ptr();
    cc.lea(ptr, scratch);
    InvokeNode* inv{};
    cc.invoke(Out(inv), reinterpret_cast<uint64_t>(fn_ptr),
              FuncSignature::build<void, float*>());
    inv->set_arg(0, ptr);
    Vec result = cc.new_ymm_ps();
    cc.vmovaps(result, scratch);
    return result;
}

// Apply vec_powf(p, q): store both ymm args to aligned scratch, call, load result.
auto invokePowf_ps(Compiler& cc, Vec a, Vec b) -> Vec {
    Mem scratchA = cc.new_stack(32, 32);
    Mem scratchB = cc.new_stack(32, 32);
    cc.vmovaps(scratchA, a);
    cc.vmovaps(scratchB, b);
    Gp ptrA = cc.new_gp_ptr();
    Gp ptrB = cc.new_gp_ptr();
    cc.lea(ptrA, scratchA);
    cc.lea(ptrB, scratchB);
    InvokeNode* inv{};
    cc.invoke(Out(inv), reinterpret_cast<uint64_t>(reinterpret_cast<void*>(vec_powf)),
              FuncSignature::build<void, float*, float const*>());
    inv->set_arg(0, ptrA);
    inv->set_arg(1, ptrB);
    Vec result = cc.new_ymm_ps();
    cc.vmovaps(result, scratchA);
    return result;
}

// Broadcast a compile-time float constant into all 8 lanes of a ymm register.
auto broadcastFloat(Compiler& cc, float val) -> Vec {
    uint32_t bits{};
    std::memcpy(&bits, &val, sizeof bits);
    Gp tmp = cc.new_gp32();
    cc.mov(tmp, bits);
    Vec xmm = cc.new_xmm_ss();
    cc.vmovd(xmm, tmp);
    Vec ymm = cc.new_ymm_ps();
    cc.vbroadcastss(ymm, xmm);
    return ymm;
}

// Emit AVX2 (ymm_ps) evaluation of `nodes[start..end)` into `stack`.
// `nodeVecs[i]` is filled with the result Vec for each processed node i.
// `constIdx` tracks the next coefficient index across calls.
// All indices into nodeVecs use the global dag index (same as the position in nodes[]).
void emitNodesAVX2(
    Compiler& cc,
    std::vector<Node> const& nodes,
    std::size_t start,
    std::size_t end,
    std::vector<Gp> const& colPtrs,
    std::vector<Vec> const& ymmCoeffs,
    std::vector<Operon::Hash> const& varOrder,
    Gp row,
    std::vector<Vec>& stack,
    std::vector<Vec>& nodeVecs,
    int& constIdx)
{
    for (std::size_t ii = start; ii < end; ++ii) {
        auto const& n = nodes[ii];

        if (n.IsRef()) {
            Vec v = nodeVecs[n.RefTo];
            nodeVecs[ii] = v;
            stack.push_back(v);
            continue;
        }

        if (n.IsVariable()) {
            auto it = std::find(varOrder.begin(), varOrder.end(), n.HashValue);
            auto colIdx = static_cast<std::size_t>(std::distance(varOrder.begin(), it));
            Vec val = cc.new_ymm_ps();
            cc.vmovups(val, x86::ptr(colPtrs[colIdx], row, 2));
            Vec weighted;
            if (n.Optimize) {
                weighted = cc.new_ymm_ps();
                cc.vmulps(weighted, val, ymmCoeffs[static_cast<std::size_t>(constIdx++)]);
            } else if (n.Value != 1.0f) {
                Vec w = broadcastFloat(cc, n.Value);
                weighted = cc.new_ymm_ps();
                cc.vmulps(weighted, val, w);
            } else {
                weighted = val;
            }
            nodeVecs[ii] = weighted;
            stack.push_back(weighted);
            continue;
        }

        if (n.IsConstant()) {
            Vec val;
            if (n.Optimize) {
                val = ymmCoeffs[static_cast<std::size_t>(constIdx++)];
            } else {
                val = broadcastFloat(cc, n.Value);
            }
            nodeVecs[ii] = val;
            stack.push_back(val);
            continue;
        }

        auto arity = n.Arity;
        std::vector<Vec> args(static_cast<std::size_t>(arity));
        for (int k = 0; k < arity; ++k) {
            args[static_cast<std::size_t>(k)] = stack.back();
            stack.pop_back();
        }

        Vec res;
        switch (n.Type) {
        case NodeType::Add: {
            res = args[0];
            for (int k = 1; k < arity; ++k) {
                Vec tmp = cc.new_ymm_ps();
                cc.vaddps(tmp, res, args[static_cast<std::size_t>(k)]);
                res = tmp;
            }
            break;
        }
        case NodeType::Mul: {
            res = args[0];
            for (int k = 1; k < arity; ++k) {
                Vec tmp = cc.new_ymm_ps();
                cc.vmulps(tmp, res, args[static_cast<std::size_t>(k)]);
                res = tmp;
            }
            break;
        }
        case NodeType::Sub: {
            if (arity == 1) {
                Vec zero = broadcastFloat(cc, 0.0f);
                res = cc.new_ymm_ps();
                cc.vsubps(res, zero, args[0]);
            } else {
                res = cc.new_ymm_ps();
                cc.vsubps(res, args[0], args[1]);
                for (int k = 2; k < arity; ++k) {
                    Vec tmp = cc.new_ymm_ps();
                    cc.vsubps(tmp, res, args[static_cast<std::size_t>(k)]);
                    res = tmp;
                }
            }
            break;
        }
        case NodeType::Div: {
            if (arity == 1) {
                Vec one = broadcastFloat(cc, 1.0f);
                res = cc.new_ymm_ps();
                cc.vdivps(res, one, args[0]);
            } else {
                res = cc.new_ymm_ps();
                cc.vdivps(res, args[0], args[1]);
                for (int k = 2; k < arity; ++k) {
                    Vec tmp = cc.new_ymm_ps();
                    cc.vdivps(tmp, res, args[static_cast<std::size_t>(k)]);
                    res = tmp;
                }
            }
            break;
        }
        case NodeType::Fmin: {
            res = args[0];
            for (int k = 1; k < arity; ++k) {
                Vec tmp = cc.new_ymm_ps();
                cc.vminps(tmp, res, args[static_cast<std::size_t>(k)]);
                res = tmp;
            }
            break;
        }
        case NodeType::Fmax: {
            res = args[0];
            for (int k = 1; k < arity; ++k) {
                Vec tmp = cc.new_ymm_ps();
                cc.vmaxps(tmp, res, args[static_cast<std::size_t>(k)]);
                res = tmp;
            }
            break;
        }
        case NodeType::Aq: {
            Vec b2  = cc.new_ymm_ps(); cc.vmulps(b2, args[1], args[1]);
            Vec one = broadcastFloat(cc, 1.0f);
            Vec sum = cc.new_ymm_ps(); cc.vaddps(sum, one, b2);
            Vec sq  = cc.new_ymm_ps(); cc.vsqrtps(sq, sum);
            res = cc.new_ymm_ps(); cc.vdivps(res, args[0], sq);
            break;
        }
        case NodeType::Pow:    { res = invokePowf_ps(cc, args[0], args[1]); break; }
        case NodeType::Powabs: {
            Vec absA = invokeF1_ps(cc, reinterpret_cast<void*>(vec_fabsf), args[0]);
            res = invokePowf_ps(cc, absA, args[1]);
            break;
        }
        case NodeType::Abs:    { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_fabsf), args[0]); break; }
        case NodeType::Acos:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_acosf), args[0]); break; }
        case NodeType::Asin:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_asinf), args[0]); break; }
        case NodeType::Atan:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_atanf), args[0]); break; }
        case NodeType::Cbrt:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_cbrtf), args[0]); break; }
        case NodeType::Ceil: {
            res = cc.new_ymm_ps();
            cc.vroundps(res, args[0],
                        Imm(static_cast<uint8_t>(RoundImm::kUp) |
                            static_cast<uint8_t>(RoundImm::kSuppress)));
            break;
        }
        case NodeType::Cos:    { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_cosf),   args[0]); break; }
        case NodeType::Cosh:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_coshf),  args[0]); break; }
        case NodeType::Exp:    { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_expf),   args[0]); break; }
        case NodeType::Floor: {
            res = cc.new_ymm_ps();
            cc.vroundps(res, args[0],
                        Imm(static_cast<uint8_t>(RoundImm::kDown) |
                            static_cast<uint8_t>(RoundImm::kSuppress)));
            break;
        }
        case NodeType::Log:    { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_logf),   args[0]); break; }
        case NodeType::Logabs: {
            Vec absVal = invokeF1_ps(cc, reinterpret_cast<void*>(vec_fabsf), args[0]);
            res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_logf), absVal);
            break;
        }
        case NodeType::Log1p:  { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_log1pf), args[0]); break; }
        case NodeType::Sin:    { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_sinf),   args[0]); break; }
        case NodeType::Sinh:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_sinhf),  args[0]); break; }
        case NodeType::Sqrt: {
            res = cc.new_ymm_ps();
            cc.vsqrtps(res, args[0]);
            break;
        }
        case NodeType::Sqrtabs: {
            Vec absVal = invokeF1_ps(cc, reinterpret_cast<void*>(vec_fabsf), args[0]);
            res = cc.new_ymm_ps();
            cc.vsqrtps(res, absVal);
            break;
        }
        case NodeType::Tan:    { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_tanf),   args[0]); break; }
        case NodeType::Tanh:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_tanhf),  args[0]); break; }
        case NodeType::Square: {
            res = cc.new_ymm_ps();
            cc.vmulps(res, args[0], args[0]);
            break;
        }
        default: {
            res = broadcastFloat(cc, 0.0f);
            break;
        }
        }

        nodeVecs[ii] = res;
        stack.push_back(res);
    }
}

} // namespace (avx2 helpers)

auto TreeCompiler::CompileAVX2(Operon::Tree const& tree) -> std::unique_ptr<CompiledTree> {
    using namespace asmjit;
    using namespace asmjit::x86;

    if (!rt_.cpu_features().x86().has(CpuFeatures::X86::kAVX2)) {
        return nullptr;
    }

    auto result = std::make_unique<CompiledTree>(rt_);

    auto const& nodes = tree.Nodes();
    for (auto const& n : nodes) {
        if (!n.IsVariable()) { continue; }
        auto h = n.HashValue;
        if (std::find(result->varOrder.begin(), result->varOrder.end(), h) == result->varOrder.end()) {
            result->varOrder.push_back(h);
        }
    }

    int nConsts = 0;
    for (auto const& n : nodes) { if (n.Optimize) { ++nConsts; } }

    CodeHolder code;
    code.init(rt_.environment(), rt_.cpu_features());
    Compiler cc(&code);

    FuncNode* fnNode = cc.add_func(
        FuncSignature::build<void, float*, float const* const*, int32_t, float const*>());
    fnNode->frame().set_avx_enabled();

    Gp outPtr    = cc.new_gp_ptr("out");
    Gp colsPtr   = cc.new_gp_ptr("cols");
    Gp nRowsArg  = cc.new_gp32("nRows");
    Gp constsPtr = cc.new_gp_ptr("consts");

    fnNode->set_arg(0, outPtr);
    fnNode->set_arg(1, colsPtr);
    fnNode->set_arg(2, nRowsArg);
    fnNode->set_arg(3, constsPtr);

    std::vector<Gp> colPtrs(result->varOrder.size());
    for (std::size_t i = 0; i < result->varOrder.size(); ++i) {
        colPtrs[i] = cc.new_gp_ptr();
        cc.mov(colPtrs[i], x86::ptr(colsPtr, static_cast<int32_t>(i * sizeof(void*))));
    }

    std::vector<Vec> ymmCoeffs(static_cast<std::size_t>(nConsts));
    for (int j = 0; j < nConsts; ++j) {
        Vec xmm_tmp = cc.new_xmm_ss();
        cc.vmovss(xmm_tmp, x86::ptr(constsPtr, static_cast<int32_t>(j * static_cast<int>(sizeof(float)))));
        ymmCoeffs[static_cast<std::size_t>(j)] = cc.new_ymm_ps();
        cc.vbroadcastss(ymmCoeffs[static_cast<std::size_t>(j)], xmm_tmp);
    }

    Gp mainEnd = cc.new_gp32("mainEnd");
    cc.mov(mainEnd, nRowsArg);
    cc.and_(mainEnd, Imm(-8));

    Gp row = cc.new_gp64("row");
    cc.xor_(row.r32(), row.r32());

    Label mainBegin = cc.new_label();
    Label mainEnd_lbl = cc.new_label();

    cc.bind(mainBegin);
    cc.cmp(row.r32(), mainEnd);
    cc.jge(mainEnd_lbl);

    {
        std::vector<Vec> stack;
        std::vector<Vec> nodeVecs(nodes.size());
        int constIdx = 0;
        stack.reserve(32);
        emitNodesAVX2(cc, nodes, 0, nodes.size(), colPtrs, ymmCoeffs, result->varOrder, row, stack, nodeVecs, constIdx);
        cc.vmovups(x86::ptr(outPtr, row, 2), stack.back());
    }

    cc.add(row.r32(), Imm(8));
    cc.jmp(mainBegin);
    cc.bind(mainEnd_lbl);

    // No scalar tail: callers must use Dataset::GetPaddedValues which returns
    // columns backed by paddedRows = (nRows+7)&~7 zero-padded floats.

    cc.ret();
    cc.end_func();

    if (auto err = cc.finalize(); err != Error::kOk) { return nullptr; }

    EvalFn fn_ptr = nullptr;
    if (auto err = rt_.add(&fn_ptr, &code); err != Error::kOk) { return nullptr; }

    result->fn = fn_ptr;
    return result;
}

auto TreeCompiler::CompileJacobian(JacobianDag const& dag) -> std::unique_ptr<CompiledJacobian> {
    using namespace asmjit;
    using namespace asmjit::x86;

    if (!rt_.cpu_features().x86().has(CpuFeatures::X86::kAVX2)) {
        return nullptr;
    }

    auto const& nodes = dag.Nodes;
    auto const nRoots = static_cast<int>(dag.Roots.size());

    if (nRoots == 0) { return nullptr; }

    auto result = std::make_unique<CompiledJacobian>(rt_);
    result->nConsts = nRoots;

    // Build var order from all dag nodes (original + derivative).
    for (auto const& n : nodes) {
        if (!n.IsVariable()) { continue; }
        auto h = n.HashValue;
        if (std::find(result->varOrder.begin(), result->varOrder.end(), h) == result->varOrder.end()) {
            result->varOrder.push_back(h);
        }
    }

    // Count optimizable constants in the original portion only.
    int nConsts = 0;
    for (std::size_t i = 0; i < dag.OriginalSize; ++i) {
        if (nodes[i].Optimize) { ++nConsts; }
    }

    CodeHolder code;
    code.init(rt_.environment(), rt_.cpu_features());
    Compiler cc(&code);

    // void fn(float* const* outs, float const* const* cols, int32_t nRows, float const* consts)
    FuncNode* fnNode = cc.add_func(
        FuncSignature::build<void, float* const*, float const* const*, int32_t, float const*>());
    fnNode->frame().set_avx_enabled();

    Gp outsPtr   = cc.new_gp_ptr("outs");
    Gp colsPtr   = cc.new_gp_ptr("cols");
    Gp nRowsArg  = cc.new_gp32("nRows");
    Gp constsPtr = cc.new_gp_ptr("consts");

    fnNode->set_arg(0, outsPtr);
    fnNode->set_arg(1, colsPtr);
    fnNode->set_arg(2, nRowsArg);
    fnNode->set_arg(3, constsPtr);

    // Pre-loop: load input column pointers.
    std::vector<Gp> colPtrs(result->varOrder.size());
    for (std::size_t i = 0; i < result->varOrder.size(); ++i) {
        colPtrs[i] = cc.new_gp_ptr();
        cc.mov(colPtrs[i], x86::ptr(colsPtr, static_cast<int32_t>(i * sizeof(void*))));
    }

    // Pre-loop: load derivative output column pointers.
    std::vector<Gp> outPtrs(static_cast<std::size_t>(nRoots));
    for (int k = 0; k < nRoots; ++k) {
        outPtrs[static_cast<std::size_t>(k)] = cc.new_gp_ptr();
        cc.mov(outPtrs[static_cast<std::size_t>(k)],
               x86::ptr(outsPtr, static_cast<int32_t>(k * static_cast<int>(sizeof(void*)))));
    }

    // Pre-loop: broadcast coefficients into ymm (AVX2 main loop).
    std::vector<Vec> ymmCoeffs(static_cast<std::size_t>(nConsts));
    for (int j = 0; j < nConsts; ++j) {
        Vec xmm_tmp = cc.new_xmm_ss();
        cc.vmovss(xmm_tmp, x86::ptr(constsPtr, static_cast<int32_t>(j * static_cast<int>(sizeof(float)))));
        ymmCoeffs[static_cast<std::size_t>(j)] = cc.new_ymm_ps();
        cc.vbroadcastss(ymmCoeffs[static_cast<std::size_t>(j)], xmm_tmp);
    }

    Gp mainEnd = cc.new_gp32("mainEnd");
    cc.mov(mainEnd, nRowsArg);
    cc.and_(mainEnd, Imm(-8));

    Gp row = cc.new_gp64("row");
    cc.xor_(row.r32(), row.r32());

    Label mainBegin = cc.new_label();
    Label mainEnd_lbl = cc.new_label();

    cc.bind(mainBegin);
    cc.cmp(row.r32(), mainEnd);
    cc.jge(mainEnd_lbl);

    {
        // Phased emission: process original nodes once, then each derivative
        // column immediately followed by its output store.  This keeps virtual
        // register liveness short and avoids the RA needing to keep hundreds of
        // ymm registers simultaneously live until after a single giant emitNodes call.
        std::vector<Vec> nodeVecs(nodes.size());
        int constIdx = 0;

        // Phase 1: original primal nodes.
        {
            std::vector<Vec> stack;
            stack.reserve(32);
            emitNodesAVX2(cc, nodes, 0, dag.OriginalSize, colPtrs, ymmCoeffs, result->varOrder, row, stack, nodeVecs, constIdx);
        }

        // Phase 2: emit each derivative column's nodes then store immediately.
        std::size_t colStart = dag.OriginalSize;
        for (int k = 0; k < nRoots; ++k) {
            auto const r = dag.Roots[static_cast<std::size_t>(k)];
            if (r == std::numeric_limits<std::size_t>::max()) {
                Vec zero = broadcastFloat(cc, 0.0f);
                cc.vmovups(x86::ptr(outPtrs[static_cast<std::size_t>(k)], row, 2), zero);
            } else {
                std::vector<Vec> stack;
                stack.reserve(32);
                emitNodesAVX2(cc, nodes, colStart, r + 1, colPtrs, ymmCoeffs, result->varOrder, row, stack, nodeVecs, constIdx);
                cc.vmovups(x86::ptr(outPtrs[static_cast<std::size_t>(k)], row, 2), nodeVecs[r]);
                colStart = r + 1;
            }
        }
    }

    cc.add(row.r32(), Imm(8));
    cc.jmp(mainBegin);
    cc.bind(mainEnd_lbl);

    // No scalar tail: callers must round nRows up to the next multiple of 8
    // and provide column/output buffers padded to that size.

    cc.ret();
    cc.end_func();

    if (auto err = cc.finalize(); err != Error::kOk) { return nullptr; }

    EvalJacFn fn_ptr = nullptr;
    if (auto err = rt_.add(&fn_ptr, &code); err != Error::kOk) { return nullptr; }

    result->fn = fn_ptr;
    return result;
}

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
