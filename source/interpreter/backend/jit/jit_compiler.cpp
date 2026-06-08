// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifdef HAVE_ASMJIT

#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/core/node.hpp"

#include <asmjit/asmjit.h>
#include <asmjit/x86.h>

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace Operon::JIT {

namespace {

using namespace asmjit;
using namespace asmjit::x86;

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

// Invoke powf(a, b).
auto invokePowf(Compiler& cc, Vec a, Vec b) -> Vec {
    Vec result = cc.new_xmm_ss();
    InvokeNode* inv{};
    cc.invoke(Out(inv),
              reinterpret_cast<uint64_t>(reinterpret_cast<void*>(powf)),
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

} // namespace

auto TreeCompiler::Compile(Operon::Tree const& tree) -> std::unique_ptr<CompiledTree> {
    auto result = std::make_unique<CompiledTree>(rt_);

    // Build unique variable order (first-seen, postfix order).
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

    // --- Pre-loop: load loop-invariant values ---

    // Column pointers: colPtrs[i] = cols[i]
    std::vector<Gp> colPtrs(result->varOrder.size());
    for (std::size_t i = 0; i < result->varOrder.size(); ++i) {
        colPtrs[i] = cc.new_gp_ptr();
        cc.mov(colPtrs[i], x86::ptr(colsPtr, static_cast<int32_t>(i * sizeof(void*))));
    }

    // Coefficients: coeffRegs[j] = consts[j] for each Optimize node
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

    // --- Row loop ---
    Gp row = cc.new_gp64("row");
    cc.xor_(row.r32(), row.r32());

    Label loopBegin = cc.new_label();
    Label loopEnd   = cc.new_label();

    cc.bind(loopBegin);
    cc.cmp(row.r32(), nRowsArg);
    cc.jge(loopEnd);

    // --- Tree evaluation via virtual register stack ---
    std::vector<Vec> stack;
    stack.reserve(32);

    int constIdx = 0; // next coefficient index

    for (auto const& n : nodes) {
        if (n.IsVariable()) {
            // Find column index for this variable.
            auto it = std::find(result->varOrder.begin(), result->varOrder.end(), n.HashValue);
            auto colIdx = static_cast<std::size_t>(std::distance(result->varOrder.begin(), it));

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
            stack.push_back(weighted);

        } else if (n.IsConstant()) {
            Vec val;
            if (n.Optimize) {
                val = coeffRegs[static_cast<std::size_t>(constIdx++)];
            } else {
                val = loadFloat(cc, n.Value);
            }
            stack.push_back(val);

        } else {
            // Operator node.
            // Children are stored right-to-left in postfix: the first pop is the
            // first (left) operand (mirrors NaryOp/BinaryOp in dispatch).
            auto arity = n.Arity;
            std::vector<Vec> args(static_cast<std::size_t>(arity));
            for (int k = 0; k < arity; ++k) {
                args[static_cast<std::size_t>(k)] = stack.back();
                stack.pop_back();
            }

            Vec res;
            switch (n.Type) {
            // --- N-ary ---
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
            // --- Binary ---
            case NodeType::Aq: {
                // args[0] / sqrt(1 + args[1]^2)
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
                Vec absA = invokeF1(cc, reinterpret_cast<void*>(fabsf), args[0]);
                res = invokePowf(cc, absA, args[1]);
                break;
            }
            // --- Unary ---
            case NodeType::Abs: {
                res = invokeF1(cc, reinterpret_cast<void*>(fabsf), args[0]);
                break;
            }
            case NodeType::Acos: {
                res = invokeF1(cc, reinterpret_cast<void*>(acosf), args[0]);
                break;
            }
            case NodeType::Asin: {
                res = invokeF1(cc, reinterpret_cast<void*>(asinf), args[0]);
                break;
            }
            case NodeType::Atan: {
                res = invokeF1(cc, reinterpret_cast<void*>(atanf), args[0]);
                break;
            }
            case NodeType::Cbrt: {
                res = invokeF1(cc, reinterpret_cast<void*>(cbrtf), args[0]);
                break;
            }
            case NodeType::Ceil: {
                res = cc.new_xmm_ss();
                cc.vroundss(res, args[0], args[0],
                            Imm(static_cast<uint8_t>(RoundImm::kUp) |
                                static_cast<uint8_t>(RoundImm::kSuppress)));
                break;
            }
            case NodeType::Cos: {
                res = invokeF1(cc, reinterpret_cast<void*>(cosf), args[0]);
                break;
            }
            case NodeType::Cosh: {
                res = invokeF1(cc, reinterpret_cast<void*>(coshf), args[0]);
                break;
            }
            case NodeType::Exp: {
                res = invokeF1(cc, reinterpret_cast<void*>(expf), args[0]);
                break;
            }
            case NodeType::Floor: {
                res = cc.new_xmm_ss();
                cc.vroundss(res, args[0], args[0],
                            Imm(static_cast<uint8_t>(RoundImm::kDown) |
                                static_cast<uint8_t>(RoundImm::kSuppress)));
                break;
            }
            case NodeType::Log: {
                res = invokeF1(cc, reinterpret_cast<void*>(logf), args[0]);
                break;
            }
            case NodeType::Logabs: {
                Vec absVal = invokeF1(cc, reinterpret_cast<void*>(fabsf), args[0]);
                res = invokeF1(cc, reinterpret_cast<void*>(logf), absVal);
                break;
            }
            case NodeType::Log1p: {
                res = invokeF1(cc, reinterpret_cast<void*>(log1pf), args[0]);
                break;
            }
            case NodeType::Sin: {
                res = invokeF1(cc, reinterpret_cast<void*>(sinf), args[0]);
                break;
            }
            case NodeType::Sinh: {
                res = invokeF1(cc, reinterpret_cast<void*>(sinhf), args[0]);
                break;
            }
            case NodeType::Sqrt: {
                res = cc.new_xmm_ss();
                cc.vsqrtss(res, args[0], args[0]);
                break;
            }
            case NodeType::Sqrtabs: {
                Vec absVal = invokeF1(cc, reinterpret_cast<void*>(fabsf), args[0]);
                res = cc.new_xmm_ss();
                cc.vsqrtss(res, absVal, absVal);
                break;
            }
            case NodeType::Tan: {
                res = invokeF1(cc, reinterpret_cast<void*>(tanf), args[0]);
                break;
            }
            case NodeType::Tanh: {
                res = invokeF1(cc, reinterpret_cast<void*>(tanhf), args[0]);
                break;
            }
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

            stack.push_back(res);
        }
    }

    // Store result for this row.
    cc.vmovss(x86::ptr(outPtr, row, 2), stack.back());

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
static void vec_cosf  (float* p) { eve::store(eve::cos  (W8{p}), p); }
static void vec_coshf (float* p) { eve::store(eve::cosh (W8{p}), p); }
static void vec_expf  (float* p) { eve::store(eve::exp  (W8{p}), p); }
static void vec_logf  (float* p) { eve::store(eve::log  (W8{p}), p); }
static void vec_log1pf(float* p) { eve::store(eve::log1p(W8{p}), p); }
static void vec_sinf  (float* p) { eve::store(eve::sin  (W8{p}), p); }
static void vec_sinhf (float* p) { eve::store(eve::sinh (W8{p}), p); }
static void vec_tanf  (float* p) { eve::store(eve::tan  (W8{p}), p); }
static void vec_tanhf (float* p) { eve::store(eve::tanh (W8{p}), p); }
static void vec_powf  (float* p, float const* q) { eve::store(eve::pow(W8{p}, W8{q}), p); }

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

// Emit AVX2 tree evaluation onto `stack` at the current row offset (in `row`).
// Uses ymm_ps registers for arithmetic; calls scalarized helpers for transcendentals.
void emitTreeAVX2(Compiler& cc,
                  std::vector<Node> const& nodes,
                  std::vector<Gp> const& colPtrs,
                  std::vector<Vec> const& ymmCoeffs,
                  std::vector<Operon::Hash> const& varOrder,
                  Gp row,
                  std::vector<Vec>& stack)
{
    int constIdx = 0;

    for (auto const& n : nodes) {
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
            stack.push_back(weighted);

        } else if (n.IsConstant()) {
            Vec val;
            if (n.Optimize) {
                val = ymmCoeffs[static_cast<std::size_t>(constIdx++)];
            } else {
                val = broadcastFloat(cc, n.Value);
            }
            stack.push_back(val);

        } else {
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
            case NodeType::Pow: {
                res = invokePowf_ps(cc, args[0], args[1]);
                break;
            }
            case NodeType::Powabs: {
                Vec absA = invokeF1_ps(cc, reinterpret_cast<void*>(vec_fabsf), args[0]);
                res = invokePowf_ps(cc, absA, args[1]);
                break;
            }
            case NodeType::Abs: {
                res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_fabsf), args[0]);
                break;
            }
            case NodeType::Acos:  { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_acosf),  args[0]); break; }
            case NodeType::Asin:  { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_asinf),  args[0]); break; }
            case NodeType::Atan:  { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_atanf),  args[0]); break; }
            case NodeType::Cbrt:  { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_cbrtf),  args[0]); break; }
            case NodeType::Ceil: {
                res = cc.new_ymm_ps();
                cc.vroundps(res, args[0],
                            Imm(static_cast<uint8_t>(RoundImm::kUp) |
                                static_cast<uint8_t>(RoundImm::kSuppress)));
                break;
            }
            case NodeType::Cos:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_cosf),   args[0]); break; }
            case NodeType::Cosh:  { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_coshf),  args[0]); break; }
            case NodeType::Exp:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_expf),   args[0]); break; }
            case NodeType::Floor: {
                res = cc.new_ymm_ps();
                cc.vroundps(res, args[0],
                            Imm(static_cast<uint8_t>(RoundImm::kDown) |
                                static_cast<uint8_t>(RoundImm::kSuppress)));
                break;
            }
            case NodeType::Log:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_logf),   args[0]); break; }
            case NodeType::Logabs: {
                Vec absVal = invokeF1_ps(cc, reinterpret_cast<void*>(vec_fabsf), args[0]);
                res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_logf), absVal);
                break;
            }
            case NodeType::Log1p: { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_log1pf), args[0]); break; }
            case NodeType::Sin:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_sinf),   args[0]); break; }
            case NodeType::Sinh:  { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_sinhf),  args[0]); break; }
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
            case NodeType::Tan:   { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_tanf),   args[0]); break; }
            case NodeType::Tanh:  { res = invokeF1_ps(cc, reinterpret_cast<void*>(vec_tanhf),  args[0]); break; }
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
            stack.push_back(res);
        }
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
    // Tell asmjit's RA to use VEX-encoded spills (vmovaps) instead of
    // legacy SSE movaps.  Without this, spilling ymm registers only saves
    // the lower 128 bits, corrupting the upper half on restore.
    fnNode->frame().set_avx_enabled();

    Gp outPtr    = cc.new_gp_ptr("out");
    Gp colsPtr   = cc.new_gp_ptr("cols");
    Gp nRowsArg  = cc.new_gp32("nRows");
    Gp constsPtr = cc.new_gp_ptr("consts");

    fnNode->set_arg(0, outPtr);
    fnNode->set_arg(1, colsPtr);
    fnNode->set_arg(2, nRowsArg);
    fnNode->set_arg(3, constsPtr);

    // Pre-loop: column pointers
    std::vector<Gp> colPtrs(result->varOrder.size());
    for (std::size_t i = 0; i < result->varOrder.size(); ++i) {
        colPtrs[i] = cc.new_gp_ptr();
        cc.mov(colPtrs[i], x86::ptr(colsPtr, static_cast<int32_t>(i * sizeof(void*))));
    }

    // Pre-loop: broadcast coefficients into ymm (for main loop) and also
    // load scalar xmm (for tail loop). We store ymm versions; scalar tail
    // will reload from constsPtr directly.
    std::vector<Vec> ymmCoeffs(static_cast<std::size_t>(nConsts));
    for (int j = 0; j < nConsts; ++j) {
        Vec xmm_tmp = cc.new_xmm_ss();
        cc.vmovss(xmm_tmp, x86::ptr(constsPtr, static_cast<int32_t>(j * static_cast<int>(sizeof(float)))));
        ymmCoeffs[static_cast<std::size_t>(j)] = cc.new_ymm_ps();
        cc.vbroadcastss(ymmCoeffs[static_cast<std::size_t>(j)], xmm_tmp);
    }

    // mainEnd = nRows & ~7  (largest multiple of 8 <= nRows)
    Gp mainEnd = cc.new_gp32("mainEnd");
    cc.mov(mainEnd, nRowsArg);
    cc.and_(mainEnd, Imm(-8));  // equivalent to & ~7 for positive integers

    Gp row = cc.new_gp64("row");
    cc.xor_(row.r32(), row.r32());

    // --- AVX2 main loop (8 rows/iter) ---
    Label mainBegin = cc.new_label();
    Label mainEnd_lbl = cc.new_label();

    cc.bind(mainBegin);
    cc.cmp(row.r32(), mainEnd);
    cc.jge(mainEnd_lbl);

    {
        std::vector<Vec> stack;
        stack.reserve(32);
        emitTreeAVX2(cc, nodes, colPtrs, ymmCoeffs, result->varOrder, row, stack);
        cc.vmovups(x86::ptr(outPtr, row, 2), stack.back());
    }

    cc.add(row.r32(), Imm(8));
    cc.jmp(mainBegin);
    cc.bind(mainEnd_lbl);

    // --- Scalar tail loop (1 row/iter) ---
    // Re-load scalar xmm coefficients from constsPtr for the tail.
    std::vector<Vec> xmmCoeffs(static_cast<std::size_t>(nConsts));
    for (int j = 0; j < nConsts; ++j) {
        xmmCoeffs[static_cast<std::size_t>(j)] = cc.new_xmm_ss();
        cc.vmovss(xmmCoeffs[static_cast<std::size_t>(j)],
                  x86::ptr(constsPtr, static_cast<int32_t>(j * static_cast<int>(sizeof(float)))));
    }

    Label tailBegin = cc.new_label();
    Label tailEnd   = cc.new_label();

    cc.bind(tailBegin);
    cc.cmp(row.r32(), nRowsArg);
    cc.jge(tailEnd);

    {
        // Scalar tree evaluation (same logic as Compile(), inline).
        std::vector<Vec> stack;
        stack.reserve(32);
        int constIdx = 0;

        for (auto const& n : nodes) {
            if (n.IsVariable()) {
                auto it = std::find(result->varOrder.begin(), result->varOrder.end(), n.HashValue);
                auto colIdx = static_cast<std::size_t>(std::distance(result->varOrder.begin(), it));

                Vec val = cc.new_xmm_ss();
                cc.vmovss(val, x86::ptr(colPtrs[colIdx], row, 2));

                Vec weighted;
                if (n.Optimize) {
                    weighted = cc.new_xmm_ss();
                    cc.vmulss(weighted, val, xmmCoeffs[static_cast<std::size_t>(constIdx++)]);
                } else if (n.Value != 1.0f) {
                    weighted = cc.new_xmm_ss();
                    cc.vmulss(weighted, val, loadFloat(cc, n.Value));
                } else {
                    weighted = val;
                }
                stack.push_back(weighted);

            } else if (n.IsConstant()) {
                Vec val;
                if (n.Optimize) {
                    val = xmmCoeffs[static_cast<std::size_t>(constIdx++)];
                } else {
                    val = loadFloat(cc, n.Value);
                }
                stack.push_back(val);

            } else {
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
                    for (int k = 1; k < arity; ++k) { Vec t = cc.new_xmm_ss(); cc.vaddss(t, res, args[static_cast<std::size_t>(k)]); res = t; }
                    break;
                }
                case NodeType::Mul: {
                    res = args[0];
                    for (int k = 1; k < arity; ++k) { Vec t = cc.new_xmm_ss(); cc.vmulss(t, res, args[static_cast<std::size_t>(k)]); res = t; }
                    break;
                }
                case NodeType::Sub: {
                    if (arity == 1) { Vec z = loadFloat(cc, 0.0f); res = cc.new_xmm_ss(); cc.vsubss(res, z, args[0]); }
                    else { res = cc.new_xmm_ss(); cc.vsubss(res, args[0], args[1]); for (int k=2;k<arity;++k) { Vec t=cc.new_xmm_ss(); cc.vsubss(t,res,args[static_cast<std::size_t>(k)]); res=t; } }
                    break;
                }
                case NodeType::Div: {
                    if (arity == 1) { Vec o = loadFloat(cc, 1.0f); res = cc.new_xmm_ss(); cc.vdivss(res, o, args[0]); }
                    else { res = cc.new_xmm_ss(); cc.vdivss(res, args[0], args[1]); for (int k=2;k<arity;++k) { Vec t=cc.new_xmm_ss(); cc.vdivss(t,res,args[static_cast<std::size_t>(k)]); res=t; } }
                    break;
                }
                case NodeType::Fmin: { res=args[0]; for(int k=1;k<arity;++k){Vec t=cc.new_xmm_ss();cc.vminss(t,res,args[static_cast<std::size_t>(k)]);res=t;} break; }
                case NodeType::Fmax: { res=args[0]; for(int k=1;k<arity;++k){Vec t=cc.new_xmm_ss();cc.vmaxss(t,res,args[static_cast<std::size_t>(k)]);res=t;} break; }
                case NodeType::Aq: {
                    Vec b2=cc.new_xmm_ss();cc.vmulss(b2,args[1],args[1]);
                    Vec one=loadFloat(cc,1.0f);Vec sum=cc.new_xmm_ss();cc.vaddss(sum,one,b2);
                    Vec sq=cc.new_xmm_ss();cc.vsqrtss(sq,sq,sum);
                    res=cc.new_xmm_ss();cc.vdivss(res,args[0],sq);
                    break;
                }
                case NodeType::Pow:    { res = invokePowf(cc, args[0], args[1]); break; }
                case NodeType::Powabs: { Vec a=invokeF1(cc,reinterpret_cast<void*>(fabsf),args[0]); res=invokePowf(cc,a,args[1]); break; }
                case NodeType::Abs:    { res=invokeF1(cc,reinterpret_cast<void*>(fabsf),args[0]); break; }
                case NodeType::Acos:   { res=invokeF1(cc,reinterpret_cast<void*>(acosf),args[0]); break; }
                case NodeType::Asin:   { res=invokeF1(cc,reinterpret_cast<void*>(asinf),args[0]); break; }
                case NodeType::Atan:   { res=invokeF1(cc,reinterpret_cast<void*>(atanf),args[0]); break; }
                case NodeType::Cbrt:   { res=invokeF1(cc,reinterpret_cast<void*>(cbrtf),args[0]); break; }
                case NodeType::Ceil: {
                    res=cc.new_xmm_ss();
                    cc.vroundss(res,args[0],args[0],Imm(static_cast<uint8_t>(RoundImm::kUp)|static_cast<uint8_t>(RoundImm::kSuppress)));
                    break;
                }
                case NodeType::Cos:    { res=invokeF1(cc,reinterpret_cast<void*>(cosf),args[0]); break; }
                case NodeType::Cosh:   { res=invokeF1(cc,reinterpret_cast<void*>(coshf),args[0]); break; }
                case NodeType::Exp:    { res=invokeF1(cc,reinterpret_cast<void*>(expf),args[0]); break; }
                case NodeType::Floor: {
                    res=cc.new_xmm_ss();
                    cc.vroundss(res,args[0],args[0],Imm(static_cast<uint8_t>(RoundImm::kDown)|static_cast<uint8_t>(RoundImm::kSuppress)));
                    break;
                }
                case NodeType::Log:    { res=invokeF1(cc,reinterpret_cast<void*>(logf),args[0]); break; }
                case NodeType::Logabs: { Vec a=invokeF1(cc,reinterpret_cast<void*>(fabsf),args[0]); res=invokeF1(cc,reinterpret_cast<void*>(logf),a); break; }
                case NodeType::Log1p:  { res=invokeF1(cc,reinterpret_cast<void*>(log1pf),args[0]); break; }
                case NodeType::Sin:    { res=invokeF1(cc,reinterpret_cast<void*>(sinf),args[0]); break; }
                case NodeType::Sinh:   { res=invokeF1(cc,reinterpret_cast<void*>(sinhf),args[0]); break; }
                case NodeType::Sqrt:   { res=cc.new_xmm_ss();cc.vsqrtss(res,args[0],args[0]); break; }
                case NodeType::Sqrtabs:{ Vec a=invokeF1(cc,reinterpret_cast<void*>(fabsf),args[0]); res=cc.new_xmm_ss();cc.vsqrtss(res,a,a); break; }
                case NodeType::Tan:    { res=invokeF1(cc,reinterpret_cast<void*>(tanf),args[0]); break; }
                case NodeType::Tanh:   { res=invokeF1(cc,reinterpret_cast<void*>(tanhf),args[0]); break; }
                case NodeType::Square: { res=cc.new_xmm_ss();cc.vmulss(res,args[0],args[0]); break; }
                default:               { res=loadFloat(cc,0.0f); break; }
                }
                stack.push_back(res);
            }
        }

        cc.vmovss(x86::ptr(outPtr, row, 2), stack.back());
    }

    cc.inc(row);
    cc.jmp(tailBegin);
    cc.bind(tailEnd);

    cc.ret();
    cc.end_func();

    if (auto err = cc.finalize(); err != Error::kOk) { return nullptr; }

    EvalFn fn_ptr = nullptr;
    if (auto err = rt_.add(&fn_ptr, &code); err != Error::kOk) { return nullptr; }

    result->fn = fn_ptr;
    return result;
}

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
