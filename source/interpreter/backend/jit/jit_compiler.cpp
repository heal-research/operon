// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifdef HAVE_ASMJIT

#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/core/node.hpp"

// Fast polynomial variants matching the interpreter's dispatch table.
// Must be included AFTER eve headers so the templates can instantiate.
#include "operon/interpreter/backend/eve/functions.hpp"

#include <fmt/format.h>
#include <immintrin.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Operon::JIT {

using namespace asmjit; // NOLINT(google-build-using-namespace)
using namespace asmjit::x86; // NOLINT(google-build-using-namespace)

// Registry accessors defined here (before EmitNodesAvx2 below, which reads
// from them) since a plain function-local static needs its declaration
// visible at first use — unlike Register{Unary,Binary}JitCodegen further
// down, which are only called by external registration code, not by
// EmitNodesAvx2 itself.
auto JitUnaryCodegenRules() -> JitUnaryCodegenRegistry&
{
    static JitUnaryCodegenRegistry registry;
    return registry;
}

auto JitBinaryCodegenRules() -> JitBinaryCodegenRegistry&
{
    static JitBinaryCodegenRegistry registry;
    return registry;
}

// =============================================================================
// Phase 2 — AVX2 vectorized compiler
// =============================================================================

namespace {

    // AVX2 transcendental helpers — take and return __m256 directly via ymm registers.
    // EVE wide<float,fixed<8>> has an implicit operator storage_type() and a storage_type
    // constructor, so returning/constructing W8(v) is zero-cost: just a ymm register.
    using W8 = eve::wide<float, eve::fixed<8>>;
    auto VecFabsf(__m256 v) noexcept -> __m256 { return eve::abs(W8(v)); }
    auto VecAcosf(__m256 v) noexcept -> __m256 { return eve::acos(W8(v)); }
    auto VecAsinf(__m256 v) noexcept -> __m256 { return eve::asin(W8(v)); }
    auto VecAtanf(__m256 v) noexcept -> __m256 { return eve::atan(W8(v)); }
    auto VecCbrtf(__m256 v) noexcept -> __m256 { return eve::cbrt(W8(v)); }
    auto VecCosf(__m256 v) noexcept -> __m256 { return Backend::detail::FastSinCos<false, float>(W8(v)); }
    auto VecCoshf(__m256 v) noexcept -> __m256 { return eve::cosh(W8(v)); }
    auto VecExpf(__m256 v) noexcept -> __m256 { return Backend::detail::FastExp<float>(W8(v)); }
    auto VecLogf(__m256 v) noexcept -> __m256 { return Backend::detail::FastLog<float>(W8(v)); }
    auto VecLogabsf(__m256 v) noexcept -> __m256 { return Backend::detail::FastLog<float>(W8(eve::abs(W8(v)))); }
    auto VecLog1pf(__m256 v) noexcept -> __m256 { return eve::log1p(W8(v)); }
    auto VecSinf(__m256 v) noexcept -> __m256 { return Backend::detail::FastSinCos<true, float>(W8(v)); }
    auto VecSinhf(__m256 v) noexcept -> __m256 { return eve::sinh(W8(v)); }
    auto VecTanf(__m256 v) noexcept -> __m256 { return eve::tan(W8(v)); }
    auto VecTanhf(__m256 v) noexcept -> __m256 { return Backend::detail::FastTanh<float>(W8(v)); }
    // Match interpreter's FastPow exactly.
    auto VecPowf(__m256 a, __m256 b) noexcept -> __m256 { return Backend::detail::FastPow<float>(W8(a), W8(b)); }

    // __m256 TypeId (89 = TypeId::kFloat32x8): passed/returned in ymm registers on x86-64 SysV ABI.
    constexpr FuncSignature ymm_f32x8_f32x8 { CallConvId::kCDecl, FuncSignature::kNoVarArgs, TypeId::kFloat32x8, TypeId::kFloat32x8 };
    constexpr FuncSignature ymm_f32x8_f32x8x2 { CallConvId::kCDecl, FuncSignature::kNoVarArgs, TypeId::kFloat32x8, TypeId::kFloat32x8, TypeId::kFloat32x8 };

    // Invoke fn(__m256) -> __m256 directly in ymm registers: no stack spill needed.
    auto InvokeF1Ps(Compiler& cc, __m256 (*fn)(__m256) noexcept, const Vec& arg) -> Vec
    {
        Vec const result = cc.new_ymm_ps();
        InvokeNode* inv {};
        cc.invoke(Out(inv), reinterpret_cast<uint64_t>(fn), ymm_f32x8_f32x8);
        inv->set_arg(0, arg);
        inv->set_ret(0, result);
        return result;
    }

    // Invoke vec_powf(__m256, __m256) -> __m256 directly in ymm registers.
    auto InvokePowfPs(Compiler& cc, const Vec& a, const Vec& b) -> Vec
    {
        Vec const result = cc.new_ymm_ps();
        InvokeNode* inv {};
        cc.invoke(Out(inv), reinterpret_cast<uint64_t>(VecPowf), ymm_f32x8_f32x8x2);
        inv->set_arg(0, a);
        inv->set_arg(1, b);
        inv->set_ret(0, result);
        return result;
    }

    // Broadcast a compile-time float constant into all 8 lanes of a ymm register.
    auto BroadcastFloat(Compiler& cc, float val) -> Vec
    {
        uint32_t bits {};
        std::memcpy(&bits, &val, sizeof bits);
        Gp const tmp = cc.new_gp32();
        cc.mov(tmp, bits);
        Vec const xmm = cc.new_xmm_ss();
        cc.vmovd(xmm, tmp);
        Vec const ymm = cc.new_ymm_ps();
        cc.vbroadcastss(ymm, xmm);
        return ymm;
    }

    // Emit AVX2 (ymm_ps) evaluation of `nodes[start..end)` into `stack`.
    // `nodeVecs[i]` is filled with the result Vec for each processed node i.
    // `constIdx` tracks the next coefficient index across calls.
    // All indices into nodeVecs use the global dag index (same as the position in nodes[]).
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    void EmitNodesAvx2(
        Compiler& cc,
        std::vector<Node> const& nodes,
        std::size_t start,
        std::size_t end,
        std::vector<Gp> const& colPtrs,
        std::vector<Vec> const& ymmCoeffs,
        std::vector<Operon::Hash> const& varOrder,
        const Gp& row,
        std::vector<Vec>& stack,
        std::vector<Vec>& nodeVecs,
        int& constIdx)
    {
        for (std::size_t ii = start; ii < end; ++ii) {
            auto const& n = nodes[ii];

            if (n.IsRef()) {
                ENSURE(n.RefTo < ii);
                Vec const copy = cc.new_ymm_ps();
                cc.vmovups(copy, nodeVecs[n.RefTo]);
                nodeVecs[ii] = copy;
                stack.push_back(copy);
                continue;
            }

            if (n.IsVariable()) {
                auto it = std::find(varOrder.begin(), varOrder.end(), n.HashValue);
                auto colIdx = static_cast<std::size_t>(std::distance(varOrder.begin(), it));
                Vec const val = cc.new_ymm_ps();
                cc.vmovups(val, x86::ptr(colPtrs[colIdx], row, 2));
                Vec weighted;
                if (n.Optimize) {
                    weighted = cc.new_ymm_ps();
                    cc.vmulps(weighted, val, ymmCoeffs[static_cast<std::size_t>(constIdx++)]);
                } else if (n.Value != 1.0F) {
                    Vec const w = BroadcastFloat(cc, n.Value);
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
                    val = BroadcastFloat(cc, n.Value);
                }
                nodeVecs[ii] = val;
                stack.push_back(val);
                continue;
            }

            auto arity = n.Arity;
            std::vector<Vec> args(static_cast<std::size_t>(arity));
            for (int k = 0; std::cmp_less(k, arity); ++k) {
                args[static_cast<std::size_t>(k)] = stack.back();
                stack.pop_back();
            }

            Vec res;
            // Add/Mul/Sub/Div/Fmin/Fmax stay hardcoded: each is an n-ary
            // reduction over however many args were popped above, not a
            // single-hash unary/binary registry entry (same scope boundary
            // as PR 8/PR 9). Every other op — unary and binary alike — goes
            // through the registry, built-in or user-defined.
            switch (n.HashValue) {
            case Operon::Hash(Operon::BuiltinOp::Add): {
                res = args[0];
                for (int k = 1; std::cmp_less(k, arity); ++k) {
                    Vec const tmp = cc.new_ymm_ps();
                    cc.vaddps(tmp, res, args[static_cast<std::size_t>(k)]);
                    res = tmp;
                }
                break;
            }
            case Operon::Hash(Operon::BuiltinOp::Mul): {
                res = args[0];
                for (int k = 1; std::cmp_less(k, arity); ++k) {
                    Vec const tmp = cc.new_ymm_ps();
                    cc.vmulps(tmp, res, args[static_cast<std::size_t>(k)]);
                    res = tmp;
                }
                break;
            }
            case Operon::Hash(Operon::BuiltinOp::Sub): {
                if (arity == 1) {
                    Vec const zero = BroadcastFloat(cc, 0.0F);
                    res = cc.new_ymm_ps();
                    cc.vsubps(res, zero, args[0]);
                } else {
                    res = cc.new_ymm_ps();
                    cc.vsubps(res, args[0], args[1]);
                    for (int k = 2; std::cmp_less(k, arity); ++k) {
                        Vec const tmp = cc.new_ymm_ps();
                        cc.vsubps(tmp, res, args[static_cast<std::size_t>(k)]);
                        res = tmp;
                    }
                }
                break;
            }
            case Operon::Hash(Operon::BuiltinOp::Div): {
                if (arity == 1) {
                    Vec const one = BroadcastFloat(cc, 1.0F);
                    res = cc.new_ymm_ps();
                    cc.vdivps(res, one, args[0]);
                } else {
                    res = cc.new_ymm_ps();
                    cc.vdivps(res, args[0], args[1]);
                    for (int k = 2; std::cmp_less(k, arity); ++k) {
                        Vec const tmp = cc.new_ymm_ps();
                        cc.vdivps(tmp, res, args[static_cast<std::size_t>(k)]);
                        res = tmp;
                    }
                }
                break;
            }
            case Operon::Hash(Operon::BuiltinOp::Fmin): {
                res = args[0];
                for (int k = 1; std::cmp_less(k, arity); ++k) {
                    Vec const tmp = cc.new_ymm_ps();
                    cc.vminps(tmp, res, args[static_cast<std::size_t>(k)]);
                    res = tmp;
                }
                break;
            }
            case Operon::Hash(Operon::BuiltinOp::Fmax): {
                res = args[0];
                for (int k = 1; std::cmp_less(k, arity); ++k) {
                    Vec const tmp = cc.new_ymm_ps();
                    cc.vmaxps(tmp, res, args[static_cast<std::size_t>(k)]);
                    res = tmp;
                }
                break;
            }
            default:
                if (auto const* unary = JitUnaryCodegenRules().TryGet(n.HashValue)) {
                    res = (*unary)(cc, args[0]);
                } else if (auto const* binary = JitBinaryCodegenRules().TryGet(n.HashValue)) {
                    res = (*binary)(cc, args[0], args[1]);
                } else {
                    throw std::runtime_error(fmt::format("JIT: no codegen for hash {} (not a built-in op)\n", n.HashValue));
                }
            }

            nodeVecs[ii] = res;
            stack.push_back(res);
        }
    }

} // anonymous namespace

void RegisterUnaryJitCodegen(Operon::Hash hash, JitUnaryCodegenFn fn)
{
    JitUnaryCodegenRules().Register(hash, std::move(fn));
}

void RegisterBinaryJitCodegen(Operon::Hash hash, JitBinaryCodegenFn fn)
{
    JitBinaryCodegenRules().Register(hash, std::move(fn));
}

namespace {

// Registers the built-in unary/binary AVX2 codegen rules exactly once,
// mirroring StandardLibrary::RegisterNames()'s lazy-static-lambda-once
// pattern. Every rule is lifted verbatim out of the old switch above.
void RegisterBuiltinJitCodegens()
{
    static auto const registered = [] {
        auto& unary  = JitUnaryCodegenRules();
        auto& binary = JitBinaryCodegenRules();

        unary.Register(Operon::Hash(Operon::BuiltinOp::Abs),     [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecFabsf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Acos),    [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecAcosf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Asin),    [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecAsinf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Atan),    [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecAtanf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Cbrt),    [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecCbrtf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Ceil),    [](Compiler& cc, Vec const& a) {
            Vec res = cc.new_ymm_ps();
            cc.vroundps(res, a, Imm(static_cast<uint8_t>(RoundImm::kUp) | static_cast<uint8_t>(RoundImm::kSuppress)));
            return res;
        });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Cos),     [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecCosf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Cosh),    [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecCoshf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Exp),     [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecExpf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Floor),   [](Compiler& cc, Vec const& a) {
            Vec res = cc.new_ymm_ps();
            cc.vroundps(res, a, Imm(static_cast<uint8_t>(RoundImm::kDown) | static_cast<uint8_t>(RoundImm::kSuppress)));
            return res;
        });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Log),     [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecLogf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Logabs),  [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecLogabsf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Log1p),   [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecLog1pf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Sin),     [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecSinf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Sinh),    [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecSinhf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Sqrt),    [](Compiler& cc, Vec const& a) {
            Vec res = cc.new_ymm_ps();
            cc.vsqrtps(res, a);
            return res;
        });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Sqrtabs), [](Compiler& cc, Vec const& a) {
            Vec const absVal = InvokeF1Ps(cc, VecFabsf, a);
            Vec res = cc.new_ymm_ps();
            cc.vsqrtps(res, absVal);
            return res;
        });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Tan),     [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecTanf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Tanh),    [](Compiler& cc, Vec const& a) { return InvokeF1Ps(cc, VecTanhf, a); });
        unary.Register(Operon::Hash(Operon::BuiltinOp::Square),  [](Compiler& cc, Vec const& a) {
            Vec res = cc.new_ymm_ps();
            cc.vmulps(res, a, a);
            return res;
        });

        binary.Register(Operon::Hash(Operon::BuiltinOp::Pow), [](Compiler& cc, Vec const& a, Vec const& b) {
            return InvokePowfPs(cc, a, b);
        });
        binary.Register(Operon::Hash(Operon::BuiltinOp::Powabs), [](Compiler& cc, Vec const& a, Vec const& b) {
            Vec const absA = InvokeF1Ps(cc, VecFabsf, a);
            return InvokePowfPs(cc, absA, b);
        });
        binary.Register(Operon::Hash(Operon::BuiltinOp::Aq), [](Compiler& cc, Vec const& a, Vec const& b) {
            Vec const b2 = cc.new_ymm_ps();
            cc.vmulps(b2, b, b);
            Vec const one = BroadcastFloat(cc, 1.0F);
            Vec const sum = cc.new_ymm_ps();
            cc.vaddps(sum, one, b2);
            Vec const sq = cc.new_ymm_ps();
            cc.vsqrtps(sq, sum);
            Vec res = cc.new_ymm_ps();
            cc.vdivps(res, a, sq);
            return res;
        });

        return true;
    }();
    static_cast<void>(registered);
}

} // anonymous namespace

auto HasUnaryJitCodegen(Operon::Hash hash) -> bool
{
    RegisterBuiltinJitCodegens();
    return JitUnaryCodegenRules().Contains(hash);
}

auto HasBinaryJitCodegen(Operon::Hash hash) -> bool
{
    RegisterBuiltinJitCodegens();
    return JitBinaryCodegenRules().Contains(hash);
}

auto TreeCompiler::CompileAVX2(Operon::Tree const& tree) -> std::unique_ptr<CompileMeta>
{
    using namespace asmjit; // NOLINT(google-build-using-namespace)
    using namespace asmjit::x86; // NOLINT(google-build-using-namespace)

    auto& rt = pick();

    if (!rt.cpu_features().x86().has(CpuFeatures::X86::kAVX2)) {
        return nullptr;
    }

    RegisterBuiltinJitCodegens();

    auto const& nodes = tree.Nodes();
    auto varOrder = VarOrder(tree);

    int nConsts = 0;
    for (auto const& n : nodes) {
        if (n.Optimize) {
            ++nConsts;
        }
    }

    CodeHolder code;
    code.init(rt.environment(), rt.cpu_features());
    Compiler cc(&code);

    FuncNode* fnNode = cc.add_func(
        FuncSignature::build<void, float*, float const* const*, int32_t, float const*>());
    fnNode->frame().set_avx_enabled();

    // A registry miss inside EmitNodesAvx2 (an op with no codegen callback,
    // built-in or user-defined) throws; caught here and converted into the
    // same "return nullptr" every other compile failure below already uses
    // — JitEvaluator's caller already treats a nullptr CompileMeta as "fall
    // back to interpreting this tree," so an unmapped op degrades exactly
    // like an asmjit finalize/runtime-add failure would.
    try {

    Gp const outPtr = cc.new_gp_ptr("out");
    Gp const colsPtr = cc.new_gp_ptr("cols");
    Gp const nRowsArg = cc.new_gp32("nRows");
    Gp const constsPtr = cc.new_gp_ptr("consts");

    fnNode->set_arg(0, outPtr);
    fnNode->set_arg(1, colsPtr);
    fnNode->set_arg(2, nRowsArg);
    fnNode->set_arg(3, constsPtr);

    std::vector<Gp> colPtrs(varOrder.size());
    for (std::size_t i = 0; i < varOrder.size(); ++i) {
        colPtrs[i] = cc.new_gp_ptr();
        cc.mov(colPtrs[i], x86::ptr(colsPtr, static_cast<int32_t>(i * sizeof(void*))));
    }

    std::vector<Vec> ymmCoeffs(static_cast<std::size_t>(nConsts));
    for (int j = 0; j < nConsts; ++j) {
        Vec const xmmTmp = cc.new_xmm_ss();
        cc.vmovss(xmmTmp, x86::ptr(constsPtr, static_cast<int32_t>(j * static_cast<int>(sizeof(float)))));
        ymmCoeffs[static_cast<std::size_t>(j)] = cc.new_ymm_ps();
        cc.vbroadcastss(ymmCoeffs[static_cast<std::size_t>(j)], xmmTmp);
    }

    Gp const mainEnd = cc.new_gp32("mainEnd");
    cc.mov(mainEnd, nRowsArg);
    cc.and_(mainEnd, Imm(-8));

    Gp const row = cc.new_gp64("row");
    cc.xor_(row.r32(), row.r32());

    Label const mainBegin = cc.new_label();
    Label const mainEndLbl = cc.new_label();

    cc.bind(mainBegin);
    cc.cmp(row.r32(), mainEnd);
    cc.jge(mainEndLbl);

    {
        std::vector<Vec> stack;
        std::vector<Vec> nodeVecs(nodes.size());
        int constIdx = 0;
        stack.reserve(32);
        EmitNodesAvx2(cc, nodes, 0, nodes.size(), colPtrs, ymmCoeffs, varOrder, row, stack, nodeVecs, constIdx);
        cc.vmovups(x86::ptr(outPtr, row, 2), stack.back());
    }

    cc.add(row.r32(), Imm(8));
    cc.jmp(mainBegin);
    cc.bind(mainEndLbl);

    cc.ret();
    cc.end_func();

    if (auto err = cc.finalize(); err != Error::kOk) {
        return nullptr;
    }

    EvalFn fnPtr = nullptr;
    if (auto err = rt.add(&fnPtr, &code); err != Error::kOk) {
        return nullptr;
    }

    auto result = std::make_unique<CompileMeta>();
    result->rtTree = &rt;
    result->fn = fnPtr;
    result->nVars = static_cast<int>(varOrder.size());
    result->nConsts = nConsts;
    return result;
    } catch (std::exception const&) {
        return nullptr;
    }
}

auto TreeCompiler::CompileJacobian(JacobianDag const& dag) -> std::unique_ptr<CompileMeta>
{
    using namespace asmjit; // NOLINT(google-build-using-namespace)
    using namespace asmjit::x86; // NOLINT(google-build-using-namespace)

    auto& rt = pick();

    if (!rt.cpu_features().x86().has(CpuFeatures::X86::kAVX2)) {
        return nullptr;
    }

    auto const& nodes = dag.Nodes;
    auto const nRoots = static_cast<int>(dag.Roots.size());

    if (nRoots == 0) {
        return nullptr;
    }

    RegisterBuiltinJitCodegens();

    try {

    // Build var order from all dag nodes (original + derivative).
    std::vector<Operon::Hash> varOrder;
    for (auto const& n : nodes) {
        if (!n.IsVariable()) {
            continue;
        }
        if (std::ranges::find(varOrder, n.HashValue) == varOrder.end()) {
            varOrder.push_back(n.HashValue);
        }
    }

    // Count optimizable constants in the original portion only.
    int nConsts = 0;
    for (std::size_t i = 0; i < dag.OriginalSize; ++i) {
        if (nodes[i].Optimize) {
            ++nConsts;
        }
    }

    CodeHolder code;
    code.init(rt.environment(), rt.cpu_features());
    Compiler cc(&code);

    // void fn(float* const* outs, float const* const* cols, int32_t nRows, float const* consts)
    FuncNode* fnNode = cc.add_func(
        FuncSignature::build<void, float* const*, float const* const*, int32_t, float const*>());
    fnNode->frame().set_avx_enabled();

    Gp const outsPtr = cc.new_gp_ptr("outs");
    Gp const colsPtr = cc.new_gp_ptr("cols");
    Gp const nRowsArg = cc.new_gp32("nRows");
    Gp const constsPtr = cc.new_gp_ptr("consts");

    fnNode->set_arg(0, outsPtr);
    fnNode->set_arg(1, colsPtr);
    fnNode->set_arg(2, nRowsArg);
    fnNode->set_arg(3, constsPtr);

    // Pre-loop: load input column pointers.
    std::vector<Gp> colPtrs(varOrder.size());
    for (std::size_t i = 0; i < varOrder.size(); ++i) {
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
        Vec const xmmTmp = cc.new_xmm_ss();
        cc.vmovss(xmmTmp, x86::ptr(constsPtr, static_cast<int32_t>(j * static_cast<int>(sizeof(float)))));
        ymmCoeffs[static_cast<std::size_t>(j)] = cc.new_ymm_ps();
        cc.vbroadcastss(ymmCoeffs[static_cast<std::size_t>(j)], xmmTmp);
    }

    Gp const mainEnd = cc.new_gp32("mainEnd");
    cc.mov(mainEnd, nRowsArg);
    cc.and_(mainEnd, Imm(-8));

    Gp const row = cc.new_gp64("row");
    cc.xor_(row.r32(), row.r32());

    Label const mainBegin = cc.new_label();
    Label const mainEndLbl = cc.new_label();

    cc.bind(mainBegin);
    cc.cmp(row.r32(), mainEnd);
    cc.jge(mainEndLbl);

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
            EmitNodesAvx2(cc, nodes, 0, dag.OriginalSize, colPtrs, ymmCoeffs, varOrder, row, stack, nodeVecs, constIdx);
        }

        // Phase 2: emit each derivative column's nodes then store immediately.
        std::size_t colStart = dag.OriginalSize;
        for (int k = 0; k < nRoots; ++k) {
            auto const r = dag.Roots[static_cast<std::size_t>(k)];
            if (r == std::numeric_limits<std::size_t>::max()) {
                Vec const zero = BroadcastFloat(cc, 0.0F);
                cc.vmovups(x86::ptr(outPtrs[static_cast<std::size_t>(k)], row, 2), zero);
            } else {
                std::vector<Vec> stack;
                stack.reserve(32);
                EmitNodesAvx2(cc, nodes, colStart, r + 1, colPtrs, ymmCoeffs, varOrder, row, stack, nodeVecs, constIdx);
                cc.vmovups(x86::ptr(outPtrs[static_cast<std::size_t>(k)], row, 2), nodeVecs[r]);
                colStart = r + 1;
            }
        }
    }

    cc.add(row.r32(), Imm(8));
    cc.jmp(mainBegin);
    cc.bind(mainEndLbl);

    // No scalar tail: callers must round nRows up to the next multiple of 8
    // and provide column/output buffers padded to that size.

    cc.ret();
    cc.end_func();

    if (auto err = cc.finalize(); err != Error::kOk) {
        return nullptr;
    }

    EvalJacFn fnPtr = nullptr;
    if (auto err = rt.add(&fnPtr, &code); err != Error::kOk) {
        return nullptr;
    }

    auto result = std::make_unique<CompileMeta>();
    result->rtJac = &rt;
    result->jacFn = fnPtr;
    return result;
    } catch (std::exception const&) {
        return nullptr;
    }
}

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
