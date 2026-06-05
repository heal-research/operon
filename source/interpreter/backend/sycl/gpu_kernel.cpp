// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

// This file is the only translation unit compiled with add_sycl_to_target.
// It intentionally avoids heavy headers (Eigen, fmt, etc.) so that the
// acpp/clang-18 pipeline does not encounter clang-21 resource intrinsics.

#include <sycl/sycl.hpp>

#include "operon/interpreter/backend/sycl/gpu_kernel.hpp"
#include "operon/interpreter/backend/sycl/gpu_node_types.hpp"

namespace Operon::Sycl {

namespace {

static constexpr int StackDepth = 64;

SYCL_EXTERNAL inline auto EvalOp(
    uint8_t type, uint8_t arity,
    float* stack, int top) -> float
{
    using NT = Operon::NodeType;
    auto const t = static_cast<NT>(type);
    float* base  = stack + top - arity + 1;

    switch (t) {
    case NT::Add: {
        float s = base[0];
        for (int k = 1; k < arity; ++k) { s += base[k]; }
        return s;
    }
    case NT::Mul: {
        float p = base[0];
        for (int k = 1; k < arity; ++k) { p *= base[k]; }
        return p;
    }
    case NT::Sub: {
        float s = base[0];
        for (int k = 1; k < arity; ++k) { s -= base[k]; }
        return s;
    }
    case NT::Div: {
        float s = base[0];
        for (int k = 1; k < arity; ++k) { s /= base[k]; }
        return s;
    }
    case NT::Fmin: {
        float s = base[0];
        for (int k = 1; k < arity; ++k) { s = sycl::fmin(s, base[k]); }
        return s;
    }
    case NT::Fmax: {
        float s = base[0];
        for (int k = 1; k < arity; ++k) { s = sycl::fmax(s, base[k]); }
        return s;
    }
    case NT::Aq: {
        float a = base[0], b = base[1];
        return a / sycl::sqrt(1.0F + b * b);
    }
    case NT::Pow:     return sycl::pow(base[0], base[1]);
    case NT::Powabs:  return sycl::pow(sycl::fabs(base[0]), base[1]);
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
    default:          return 0.0F;
    }
}

} // namespace

auto RunKernel(EncodedPopulation const& enc) -> std::vector<float>
{
    auto const popSize = enc.PopSize;
    auto const maxLen  = enc.MaxLen;
    auto const nRows   = enc.NRows;

    std::vector<float> results(static_cast<std::size_t>(popSize) * nRows, 0.0F);

    {
        sycl::queue q{ sycl::default_selector_v };

        sycl::buffer<GpuOp,    1> opsBuf   (enc.Ops.data(),        sycl::range<1>{enc.Ops.size()});
        sycl::buffer<uint32_t, 1> lenBuf   (enc.Lengths.data(),    sycl::range<1>{popSize});
        sycl::buffer<float,    1> dataBuf  (enc.DataBuffer.data(), sycl::range<1>{enc.DataBuffer.size()});
        sycl::buffer<float,    1> resultBuf(results.data(),        sycl::range<1>{results.size()});

        q.submit([&](sycl::handler& cgh) {
            auto opsAcc    = opsBuf   .get_access<sycl::access::mode::read>(cgh);
            auto lenAcc    = lenBuf   .get_access<sycl::access::mode::read>(cgh);
            auto dataAcc   = dataBuf  .get_access<sycl::access::mode::read>(cgh);
            auto resultAcc = resultBuf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<2>{popSize, nRows},
                [=](sycl::item<2> item) {
                    auto const si = static_cast<uint32_t>(item.get_id(0));
                    auto const ri = static_cast<uint32_t>(item.get_id(1));

                    float stack[StackDepth]; // NOLINT(cppcoreguidelines-avoid-c-arrays)
                    int   top = -1;

                    auto const len    = lenAcc[si];
                    auto const offset = static_cast<std::size_t>(si) * maxLen;

                    using NT = Operon::NodeType;

                    for (uint32_t k = 0; k < len; ++k) {
                        GpuOp const op = opsAcc[offset + k];

                        if (op.Type == GpuNoop) { continue; }

                        auto const t = static_cast<NT>(op.Type);

                        if (t == NT::Variable) {
                            float colVal = dataAcc[static_cast<std::size_t>(op.VarIdx) * nRows + ri];
                            stack[++top] = colVal * op.Value;
                        } else if (t == NT::Constant) {
                            stack[++top] = op.Value;
                        } else {
                            float res = EvalOp(op.Type, op.Arity, stack, top);
                            top -= (static_cast<int>(op.Arity) - 1);
                            stack[top] = res;
                        }
                    }

                    resultAcc[static_cast<std::size_t>(si) * nRows + ri] =
                        (top == 0) ? stack[0] : 0.0F;
                });
        });

        q.wait();
    }

    return results;
}

} // namespace Operon::Sycl
