// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_JIT_LEAST_SQUARES_HPP
#define OPERON_JIT_LEAST_SQUARES_HPP

#ifdef HAVE_ASMJIT

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gsl/pointers>

#include "operon/core/contracts.hpp"
#include "operon/core/range.hpp"
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/least_squares.hpp"

namespace Operon {

// JIT kernels and their column/Jacobian buffers are float-typed
// (jit_compiler.hpp); a double Operon::Scalar build cannot use them.
static_assert(std::same_as<Scalar, float>, "JIT least-squares costs require Operon::Scalar == float (USE_SINGLE_PRECISION=ON)");

/**
 * Backend-neutral LeastSquaresCostFunction over compiled JIT kernels: raw
 * residual = compiled/interpreter prediction - target, raw Jacobian is the
 * compiled derivative DAG (EvalJacFn) when available, falling back to
 * interpreter JacRev copied into the caller's arbitrary-stride view. Writes
 * raw unweighted residuals/Jacobians -- LeastSquaresLMAdapter or
 * LeastSquaresGradientAdapter applies numerical WLS weights. colPtrs[i] and
 * jacColPtrs[i] must follow the ordering returned by JIT::VarOrder(tree),
 * each already offset to range.Start(). Compile/interpreter failures are
 * reported as LeastSquaresErrorCode::EvaluationFailure with the original
 * InterpreterError preserved as Cause.
 */
class JitLeastSquaresCostFunction final : public LeastSquaresCostFunction {
public:
    JitLeastSquaresCostFunction(
        gsl::not_null<InterpreterBase<Scalar> const*> interpreter,
        JIT::EvalFn fn,
        std::vector<float const*> colPtrs,
        ConstScalarSpan target,
        Range range,
        JIT::EvalJacFn jacFn = nullptr,
        std::vector<float const*> jacColPtrs = {},
        int nVars = -1,
        int nConsts = -1)
        : interpreter_(interpreter)
        , fn_(fn)
        , colPtrs_(std::move(colPtrs))
        , jacFn_(jacFn)
        , jacColPtrs_(std::move(jacColPtrs))
        , target_(target.subspan(range.Start(), range.Size()))
        , range_(range)
        , numParameters_(static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount()))
        , nRowsPad_(CheckedPaddedRows(range.Size()))
        , scratchResiduals_(nRowsPad_)
        , scratchJac_(nRowsPad_ * numParameters_)
        , nVars_(nVars)
        , nConsts_(nConsts)
    {
        // Precomputed once: these point into scratchJac_'s fixed layout, which
        // never changes across Evaluate() calls, so recomputing them per call
        // (as a freshly heap-allocated std::vector, no less) was pure waste on
        // what can be a very hot path. Skipped for residual-only objects
        // (jacFn_ == nullptr), which never dereference jacOutPtrs_.
        if (jacFn_ != nullptr) {
            jacOutPtrs_.resize(numParameters_);
            for (std::size_t k = 0; k < numParameters_; ++k) {
                jacOutPtrs_[k] = scratchJac_.data() + (k * nRowsPad_);
            }
        }
    }

    // jacOutPtrs_ points into this object's own scratchJac_; a copy would leave
    // the copy's pointers aimed at the source's buffer, silently returning
    // stale Jacobians.
    JitLeastSquaresCostFunction(JitLeastSquaresCostFunction const&) = delete;
    auto operator=(JitLeastSquaresCostFunction const&) -> JitLeastSquaresCostFunction& = delete;
    JitLeastSquaresCostFunction(JitLeastSquaresCostFunction&&) = delete;
    auto operator=(JitLeastSquaresCostFunction&&) -> JitLeastSquaresCostFunction& = delete;
    ~JitLeastSquaresCostFunction() final = default;

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return numParameters_; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return range_.Size(); }

    [[nodiscard]] auto Evaluate(
        ConstScalarSpan parameters,
        ScalarSpan residuals,
        std::optional<ScalarMatrixView> jacobian) const
        -> tl::expected<void, LeastSquaresError> override
    {
        auto const n = range_.Size();
        if (parameters.size() != numParameters_) {
            return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape, .Expected = numParameters_, .Actual = parameters.size() });
        }
        if (residuals.size() != n) {
            return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape, .Expected = n, .Actual = residuals.size() });
        }
        if (jacobian && (jacobian->extent(0) != n || jacobian->extent(1) != numParameters_)) {
            return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape, .Expected = n, .Actual = jacobian->extent(0), .Row = n, .Column = numParameters_ });
        }

        auto const nRowsPad = static_cast<int32_t>(nRowsPad_);

        if (fn_ != nullptr) {
            ENSURE(nVars_ < 0 || static_cast<int>(colPtrs_.size()) == nVars_);
            ENSURE(nConsts_ < 0 || static_cast<int>(numParameters_) == nConsts_);
            fn_(scratchResiduals_.data(), colPtrs_.data(), nRowsPad, parameters.data());
            std::copy_n(scratchResiduals_.data(), n, residuals.data());
        } else {
            auto predicted = interpreter_->Evaluate(parameters, range_, residuals);
            if (!predicted) {
                return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::EvaluationFailure, .Cause = predicted.error() });
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            residuals[i] -= target_[i];
        }

        if (jacobian) {
            if (jacFn_ != nullptr) {
                ENSURE(nVars_ < 0 || static_cast<int>(jacColPtrs_.size()) == nVars_);
                ENSURE(nConsts_ < 0 || static_cast<int>(numParameters_) == nConsts_);
                jacFn_(jacOutPtrs_.data(), jacColPtrs_.data(), nRowsPad, parameters.data());
                for (std::size_t j = 0; j < numParameters_; ++j) {
                    for (std::size_t i = 0; i < n; ++i) {
                        At(*jacobian, i, j) = scratchJac_[(j * nRowsPad_) + i];
                    }
                }
            } else {
                jacobianScratch_.resize(n * numParameters_);
                auto jacResult = interpreter_->JacRev(parameters, range_, jacobianScratch_);
                if (!jacResult) {
                    return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::EvaluationFailure, .Cause = jacResult.error() });
                }
                for (std::size_t j = 0; j < numParameters_; ++j) {
                    for (std::size_t i = 0; i < n; ++i) {
                        At(*jacobian, i, j) = jacobianScratch_[(j * n) + i];
                    }
                }
            }
        }

        return {};
    }

private:
    gsl::not_null<InterpreterBase<Scalar> const*> interpreter_;
    JIT::EvalFn fn_;
    std::vector<float const*> colPtrs_;
    JIT::EvalJacFn jacFn_ = nullptr;
    std::vector<float const*> jacColPtrs_;
    ConstScalarSpan target_;
    Range range_; // NOLINT(readability-identifier-naming)
    std::size_t numParameters_;
    std::size_t nRowsPad_;
    mutable std::vector<Scalar> scratchResiduals_;
    mutable std::vector<Scalar> scratchJac_;
    std::vector<float*> jacOutPtrs_; // precomputed pointers into scratchJac_, see ctor
    mutable std::vector<Scalar> jacobianScratch_; // interpreter JacRev fallback, column-major
    int nVars_ = -1;
    int nConsts_ = -1;
    static auto CheckedPaddedRows(std::size_t rows) -> std::size_t
    {
        constexpr auto maxRows = static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (rows > maxRows - 7) {
            throw std::invalid_argument("JIT least-squares range exceeds the supported row count");
        }
        return (rows + 7U) & ~std::size_t { 7U };
    }

};

} // namespace Operon

#endif // HAVE_ASMJIT
#endif
