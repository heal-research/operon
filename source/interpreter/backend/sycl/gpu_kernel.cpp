// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

// Only translation unit compiled with add_sycl_to_target.
// Intentionally avoids heavy headers (Eigen, fmt, unordered_dense) so that
// the acpp/clang-18 pipeline does not encounter clang-21 resource intrinsics.

#include <cfloat>
#include <sycl/sycl.hpp>

#include "operon/interpreter/backend/sycl/gpu_kernel.hpp"
#include "operon/interpreter/backend/sycl/gpu_node_types.hpp"

namespace Operon::Sycl {

// ---- GpuContext ------------------------------------------------------------

struct GpuContext {
    sycl::queue q;

    float*    d_data{nullptr};    // [nVars × nRows] dataset, column-major
    float*    d_target{nullptr};  // [nRows] target column for fitness reduction
    GpuOp*    d_ops{nullptr};     // [popSize × maxLen] ops, individual-major
    uint32_t* d_lengths{nullptr}; // [popSize] tree lengths
    float*    d_fitness{nullptr}; // [popSize] per-individual fitness

    uint32_t nVars{0}, nRows{0};
    uint32_t opsCapacity{0};     // allocated GpuOp elements
    uint32_t fitnessCapacity{0}; // allocated float elements (fitness)
    uint32_t targetCapacity{0};  // allocated float elements (target)

    GpuContext()
        : q(sycl::default_selector_v, sycl::property::queue::in_order{})
    {}

    ~GpuContext() {
        if (d_data)    { sycl::free(d_data,    q); }
        if (d_target)  { sycl::free(d_target,  q); }
        if (d_ops)     { sycl::free(d_ops,     q); }
        if (d_lengths) { sycl::free(d_lengths, q); }
        if (d_fitness) { sycl::free(d_fitness, q); }
    }

    GpuContext(GpuContext const&)            = delete;
    GpuContext& operator=(GpuContext const&) = delete;
};

// ---- EvalOp helper ---------------------------------------------------------

namespace {

// Maximum stack depth for the postfix evaluator. Tree depth is ≤ 7 in
// typical GP runs (depth-7 balanced binary tree = 127 nodes). 8 entries
// exactly fills the RDNA2 per-thread VGPR budget with VW=4 (32 VGPRs/thread).
static constexpr int StackDepth = 8;

// Vector width for row-wise vectorization. Each thread evaluates VW rows
// simultaneously through the tree (same node path, N-wide data ops),
// amortizing interpreter dispatch overhead by VW.
// VW=4 wins on RDNA2 gfx1030 at typical GP population sizes (≤ 5000);
// larger values increase register spill pressure and yield diminishing returns.
static constexpr uint32_t VW = 4U;
using vecW = sycl::vec<float, VW>;

// Load VW consecutive floats from ptr into a vecW (loop unrolled at compile time).
template<uint32_t N = VW>
SYCL_EXTERNAL inline auto load_vecW(float const* ptr) -> sycl::vec<float, N>
{
    sycl::vec<float, N> v;
    for (uint32_t i = 0U; i < N; ++i) { v[i] = ptr[i]; }
    return v;
}

// Templatized over T = float (scalar) or vecW (vectorized).
// All SYCL math functions are gentype and accept vec<float,N> directly.
// Operon trees store nodes in left-to-right depth-first order with the
// parent LAST.  arg1 sits immediately before the parent (pushed last →
// stack top = base[arity-1]); arg2 is further left (pushed first →
// base[0]).  Commutative ops are order-independent; non-commutative ops
// (Sub, Div, Aq, Pow, Powabs) must use base[arity-1] as their first operand.
template<typename T>
SYCL_EXTERNAL inline auto EvalOp(
    uint8_t type, uint8_t arity,
    T* stack, int top) -> T
{
    using NT = Operon::NodeType;
    auto const t = static_cast<NT>(type);
    T* base = stack + top - arity + 1;

    switch (t) {
    case NT::Add: { T s = base[0]; for (int k = 1; k < arity; ++k) { s += base[k]; } return s; }
    case NT::Mul: { T p = base[0]; for (int k = 1; k < arity; ++k) { p *= base[k]; } return p; }
    case NT::Sub: { T s = base[arity-1]; for (int k = 0; k < arity-1; ++k) { s -= base[k]; } return s; }
    case NT::Div: { T s = base[arity-1]; for (int k = 0; k < arity-1; ++k) { s /= base[k]; } return s; }
    case NT::Fmin: { T s = base[0]; for (int k = 1; k < arity; ++k) { s = sycl::fmin(s, base[k]); } return s; }
    case NT::Fmax: { T s = base[0]; for (int k = 1; k < arity; ++k) { s = sycl::fmax(s, base[k]); } return s; }
    case NT::Aq:      return base[1] / sycl::sqrt(T{1.0F} + base[0] * base[0]);
    case NT::Pow:     return sycl::pow(base[1], base[0]);
    case NT::Powabs:  return sycl::pow(sycl::fabs(base[1]), base[0]);
    case NT::Abs:     return sycl::fabs(base[0]);
    case NT::Acos:    return sycl::acos(base[0]);
    case NT::Asin:    return sycl::asin(base[0]);
    case NT::Atan:    return sycl::atan(base[0]);
    case NT::Cbrt:    return sycl::cbrt(base[0]);
    case NT::Ceil:    return sycl::ceil(base[0]);
    case NT::Cos:     return sycl::cos(base[0]);
    case NT::Cosh:    return sycl::cosh(base[0]);
    case NT::Exp:     return sycl::exp(base[0]);
    case NT::Floor:   return sycl::floor(base[0]);
    case NT::Log:     return sycl::log(base[0]);
    case NT::Logabs:  return sycl::log(sycl::fabs(base[0]));
    case NT::Log1p:   return sycl::log1p(base[0]);
    case NT::Sin:     return sycl::sin(base[0]);
    case NT::Sinh:    return sycl::sinh(base[0]);
    case NT::Sqrt:    return sycl::sqrt(base[0]);
    case NT::Sqrtabs: return sycl::sqrt(sycl::fabs(base[0]));
    case NT::Square:  return base[0] * base[0];
    case NT::Tan:     return sycl::tan(base[0]);
    case NT::Tanh:    return sycl::tanh(base[0]);
    default:          return T{0.0F};
    }
}

} // namespace

// ---- Public API ------------------------------------------------------------

auto GpuContextCreate() -> GpuContext*
{
    return new GpuContext{};
}

void GpuContextDestroy(GpuContext* ctx) noexcept
{
    delete ctx;
}

void GpuContextUploadDataset(GpuContext*  ctx,
                              float const* colMajor,
                              uint32_t     nVars,
                              uint32_t     nRows)
{
    if (ctx->d_data) {
        sycl::free(ctx->d_data, ctx->q);
    }
    ctx->nVars  = nVars;
    ctx->nRows  = nRows;
    ctx->d_data = sycl::malloc_device<float>(
        static_cast<std::size_t>(nVars) * nRows, ctx->q);
    // In-order queue: upload completes before the next kernel submission.
    ctx->q.memcpy(ctx->d_data, colMajor,
        static_cast<std::size_t>(nVars) * nRows * sizeof(float));
}

void GpuContextUploadTarget(GpuContext*  ctx,
                             float const* target,
                             uint32_t     nRows)
{
    auto const count = static_cast<std::size_t>(nRows);
    if (count > ctx->targetCapacity) {
        if (ctx->d_target) { sycl::free(ctx->d_target, ctx->q); }
        ctx->d_target       = sycl::malloc_device<float>(count, ctx->q);
        ctx->targetCapacity = nRows;
    }
    ctx->q.memcpy(ctx->d_target, target, count * sizeof(float));
}

void GpuContextEvaluate(GpuContext*      ctx,
                         GpuOp const*    ops,
                         uint32_t const* lengths,
                         uint32_t        popSize,
                         uint32_t        maxLen,
                         GpuFitType      fitType,
                         bool            doScale,
                         float*          fitness)
{
    auto const nRows = ctx->nRows;

    // Grow ops + lengths buffers if needed
    auto const opsCount = static_cast<std::size_t>(popSize) * maxLen;
    if (opsCount > ctx->opsCapacity) {
        if (ctx->d_ops)     { sycl::free(ctx->d_ops,     ctx->q); }
        if (ctx->d_lengths) { sycl::free(ctx->d_lengths, ctx->q); }
        ctx->d_ops      = sycl::malloc_device<GpuOp>    (opsCount, ctx->q);
        ctx->d_lengths  = sycl::malloc_device<uint32_t> (popSize,  ctx->q);
        ctx->opsCapacity = static_cast<uint32_t>(opsCount);
    }

    // Grow per-individual fitness buffer if needed
    if (popSize > ctx->fitnessCapacity) {
        if (ctx->d_fitness) { sycl::free(ctx->d_fitness, ctx->q); }
        ctx->d_fitness       = sycl::malloc_device<float>(popSize, ctx->q);
        ctx->fitnessCapacity = popSize;
    }

    // Upload ops and lengths (in-order queue: completes before kernel starts)
    ctx->q.memcpy(ctx->d_ops,     ops,     opsCount * sizeof(GpuOp));
    ctx->q.memcpy(ctx->d_lengths, lengths, popSize  * sizeof(uint32_t));

    // Capture raw pointers for the kernel lambda
    GpuOp const* d_ops     = ctx->d_ops;
    uint32_t*    d_lengths = ctx->d_lengths;
    float const* d_data    = ctx->d_data;
    float*       d_fitness = ctx->d_fitness;
    float const* d_target  = ctx->d_target;

    // Fused eval + fitness reduction kernel.
    // One work-group per individual (si = group id). Each thread processes
    // rows lid, lid+WGS, lid+2*WGS, ..., evaluating the tree and accumulating
    // 6 partial sums. A tree reduction over local memory yields the scalar fitness.
    //
    // Key advantage: d_ops[si * maxLen..] (≤ 400 B) stays L1-cached across all
    // row batches; the 10 kB d_target stays L2-cached across all work-groups.
    // No intermediate results buffer: eliminates the O(pop × nRows) HBM round-trip.
    static constexpr uint32_t WGS      = 64U;
    static constexpr uint32_t NumAccum = 6U;  // sx,sy,sxx,sxy,syy,sae

    ctx->q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lmem(WGS * NumAccum, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>{static_cast<std::size_t>(popSize) * WGS, WGS},
            [=](sycl::nd_item<1> item) {
                auto const si  = static_cast<uint32_t>(item.get_group(0));
                auto const lid = static_cast<uint32_t>(item.get_local_id(0));

                auto const len    = d_lengths[si];
                auto const offset = static_cast<std::size_t>(si) * maxLen;

                using NT = Operon::NodeType;

                // Vectorized main loop: each iteration processes VW=4 rows simultaneously.
                // The tree switch fires once per node instead of VW times, amortizing
                // interpreter dispatch overhead. All SYCL math functions accept vecW.
                uint32_t const nRowsAligned = (nRows / VW) * VW;

                vecW vsx{0.0F}, vsy{0.0F}, vsxx{0.0F}, vsxy{0.0F}, vsyy{0.0F}, vsae{0.0F};

                for (uint32_t ri = lid * VW; ri < nRowsAligned; ri += WGS * VW) {
                    vecW vstack[StackDepth]; // NOLINT(cppcoreguidelines-avoid-c-arrays)
                    int  top = -1;

                    for (uint32_t k = 0; k < len; ++k) {
                        GpuOp const op = d_ops[offset + k];
                        if (op.Type == GpuNoop) { continue; }

                        auto const t = static_cast<NT>(op.Type);
                        if (t == NT::Variable) {
                            auto const col = static_cast<std::size_t>(op.VarIdx) * nRows + ri;
                            vstack[++top] = load_vecW(d_data + col) * op.Value;
                        } else if (t == NT::Constant) {
                            vstack[++top] = vecW{op.Value};
                        } else {
                            vecW res = EvalOp(op.Type, op.Arity, vstack, top);
                            top -= static_cast<int>(op.Arity) - 1;
                            vstack[top] = res;
                        }
                    }

                    vecW const x = (top == 0) ? vstack[0] : vecW{0.0F};
                    vecW const y = load_vecW(d_target + ri);
                    vsx  += x;
                    vsy  += y;
                    vsxx += x * x;
                    vsxy += x * y;
                    vsyy += y * y;
                    vsae += sycl::fabs(x - y);
                }

                // Reduce vecW accumulators to scalar
                auto hsum = [](vecW v) -> float {
                    float s = 0.0F;
                    for (uint32_t i = 0U; i < VW; ++i) { s += v[i]; }
                    return s;
                };
                float sx  = hsum(vsx);
                float sy  = hsum(vsy);
                float sxx = hsum(vsxx);
                float sxy = hsum(vsxy);
                float syy = hsum(vsyy);
                float sae = hsum(vsae);

                // Scalar tail: at most VW-1 rows not covered by the vectorized loop
                for (uint32_t ri = nRowsAligned + lid; ri < nRows; ri += WGS) {
                    float fstack[StackDepth]; // NOLINT(cppcoreguidelines-avoid-c-arrays)
                    int   top = -1;

                    for (uint32_t k = 0; k < len; ++k) {
                        GpuOp const op = d_ops[offset + k];
                        if (op.Type == GpuNoop) { continue; }

                        auto const t = static_cast<NT>(op.Type);
                        if (t == NT::Variable) {
                            fstack[++top] = d_data[static_cast<std::size_t>(op.VarIdx) * nRows + ri] * op.Value;
                        } else if (t == NT::Constant) {
                            fstack[++top] = op.Value;
                        } else {
                            float res = EvalOp(op.Type, op.Arity, fstack, top);
                            top -= static_cast<int>(op.Arity) - 1;
                            fstack[top] = res;
                        }
                    }

                    float const x = (top == 0) ? fstack[0] : 0.0F;
                    float const y = d_target[ri];
                    sx  += x; sy  += y;
                    sxx += x * x; sxy += x * y;
                    syy += y * y; sae += sycl::fabs(x - y);
                }

                // Store partial sums in local memory
                lmem[lid * NumAccum + 0] = sx;
                lmem[lid * NumAccum + 1] = sy;
                lmem[lid * NumAccum + 2] = sxx;
                lmem[lid * NumAccum + 3] = sxy;
                lmem[lid * NumAccum + 4] = syy;
                lmem[lid * NumAccum + 5] = sae;

                item.barrier(sycl::access::fence_space::local_space);

                // Tree reduction
                for (uint32_t stride = WGS / 2U; stride > 0U; stride >>= 1U) {
                    if (lid < stride) {
                        for (uint32_t k = 0; k < NumAccum; ++k) {
                            lmem[lid * NumAccum + k] += lmem[(lid + stride) * NumAccum + k];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lid == 0U) {
                    float const n    = static_cast<float>(nRows);
                    float const mX   = lmem[0] / n;
                    float const mY   = lmem[1] / n;
                    float const varX = sycl::fmax(lmem[2] / n - mX * mX, 0.0F);
                    float const varY = sycl::fmax(lmem[4] / n - mY * mY, 0.0F);
                    float const covXY= lmem[3] / n - mX * mY;
                    float const mae  = lmem[5] / n;

                    float fit = FLT_MAX;

                    if (doScale) {
                        float const denom = varX * varY;
                        float const cor2  = (denom > 0.0F) ? (covXY * covXY) / denom : 0.0F;
                        float const mseS  = (varX > 0.0F) ? sycl::fmax(varY - covXY * covXY / varX, 0.0F) : varY;

                        switch (fitType) {
                        case GpuFitType::R2:
                        case GpuFitType::C2:  fit = -cor2; break;
                        case GpuFitType::MSE:
                        case GpuFitType::SSE: fit = mseS;  break;
                        case GpuFitType::NMSE:fit = 1.0F - cor2; break;
                        case GpuFitType::RMSE:fit = sycl::sqrt(mseS); break;
                        case GpuFitType::MAE: fit = mae; break;
                        }
                    } else {
                        float const dm  = mX - mY;
                        float const mse = sycl::fmax(varX - 2.0F * covXY + varY + dm * dm, 0.0F);

                        switch (fitType) {
                        case GpuFitType::MSE:
                        case GpuFitType::SSE: fit = mse; break;
                        case GpuFitType::NMSE:fit = (varY > 0.0F) ? mse / varY : 0.0F; break;
                        case GpuFitType::RMSE:fit = sycl::sqrt(mse); break;
                        case GpuFitType::R2: {
                            float const r2 = (varY > 0.0F) ? 1.0F - mse / varY : 0.0F;
                            fit = -r2;
                            break;
                        }
                        case GpuFitType::C2: {
                            float const denom2 = varX * varY;
                            float const cor2   = (denom2 > 0.0F) ? (covXY * covXY) / denom2 : 0.0F;
                            fit = -cor2;
                            break;
                        }
                        case GpuFitType::MAE: fit = mae; break;
                        }
                    }

                    d_fitness[si] = sycl::isfinite(fit) ? fit : FLT_MAX;
                }
            });
    });

    // Download only the fitness vector — O(popSize) floats
    ctx->q.memcpy(fitness, d_fitness, static_cast<std::size_t>(popSize) * sizeof(float));
    ctx->q.wait();
}

} // namespace Operon::Sycl
