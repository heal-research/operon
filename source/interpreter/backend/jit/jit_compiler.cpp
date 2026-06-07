// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifdef HAVE_ASMJIT

#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/core/node.hpp"

#include <asmjit/asmjit.h>
#include <asmjit/x86.h>

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

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
