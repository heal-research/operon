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
#include <optional>
#include <utility>
#include <vector>

#include <gsl/pointers>

#include "operon/core/range.hpp"
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/least_squares.hpp"

namespace Operon {

// JIT kernels and their column/Jacobian buffers are float-typed
// (jit_compiler.hpp); a double Operon::Scalar build cannot use them.
static_assert(
    std::same_as<Scalar, float>, "JIT least-squares costs require Operon::Scalar == float (USE_SINGLE_PRECISION=ON)");

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
 *
 * Construction never throws for invalid user data. The constructor validates
 * once, without allocating on the evaluation path, and stores the first
 * violation as a typed LeastSquaresError; every later Evaluate() then returns
 * that same error before touching any buffer, kernel, or the interpreter, so
 * an invalid object is deterministic and cannot reach undefined behavior.
 * NumParameters()/NumResiduals() still report the values derived from the tree
 * and the range, so wrapping adapters keep a consistent shape. Checks, in
 * order:
 *  - Range: range.Start() + range.Size() must fit target.size(), else
 *    InvalidShape (Expected = required target size, Actual = target.size()).
 *  - Row count: range.Size() must be at most INT32_MAX - 7 (the kernels take
 *    an int32 padded row count), else InvalidShape (Expected = INT32_MAX - 7,
 *    Actual = range.Size()).
 *  - Kernel metadata (only when fn or jacFn is supplied): a non-negative
 *    nConsts must equal NumParameters(), else InvalidShape (Expected =
 *    nConsts, Actual = NumParameters()).
 *  - Column arrays (colPtrs when fn is supplied, jacColPtrs when jacFn is
 *    supplied; Row = 0 and 1 respectively): the size must equal nVars, or
 *    JIT::VarOrder(tree).size() when nVars is negative, else InvalidShape
 *    (Expected = variable count, Actual = array size); every pointer must be
 *    non-null, else InvalidView (Column = index of the first null pointer).
 * What cannot be checked cheaply stays a caller precondition: that fn/jacFn
 * were compiled for this tree, that each non-null column pointer addresses at
 * least nRowsPad readable floats, and that colPtrs follow VarOrder(tree).
 * A null fn (or jacFn) is legitimate and selects the interpreter (JacRev)
 * path; the column array for an absent kernel is ignored.
 *
 * Oversized ranges. An invalid range can make NumResiduals() larger than any
 * solver backend can represent (> INT_MAX). The constructor still allocates
 * nothing in that case (it returns before sizing any scratch), and
 * FitLeastSquares rejects the shape before allocating or evaluating the cost.
 *
 * Through FitLeastSquares, a stored configuration error that is not an
 * oversized range is observed by evaluating the cost once, so the adapter
 * counts that failed call (FunctionEvaluations == 1; JacobianEvaluations == 1
 * on Tiny, whose first call also requests the Jacobian, 0 on Eigen). An
 * oversized range is rejected before any evaluation and counts nothing.
 *
 * Thread safety. Evaluate() is const but writes mutable scratch (the padded
 * residual and Jacobian buffers and the interpreter-Jacobian fallback
 * buffer), and there is no locking. One instance must therefore be used by
 * one thread at a time; concurrent Evaluate() calls on the same instance are
 * a data race. Use one instance per thread (the construction cost is a
 * single validation pass plus the scratch allocation).
 *
 * LeastSquaresError::Row/Column. Their meaning depends on Code: for the
 * column-array errors above Row selects the kernel (0 residual, 1
 * Jacobian) and Column indexes the first null pointer; for a Jacobian view
 * shape error in Evaluate(), Row/Column carry the expected extents; for
 * non-finite outputs reported by an adapter they are matrix coordinates.
 */
class JitLeastSquaresCostFunction final : public LeastSquaresCostFunction {
public:
    JitLeastSquaresCostFunction(gsl::not_null<InterpreterBase<Scalar> const*> interpreter, JIT::EvalFn fn,
        std::vector<float const*> colPtrs, ConstScalarSpan target, Range range, JIT::EvalJacFn jacFn = nullptr,
        std::vector<float const*> jacColPtrs = {}, int nVars = -1, int nConsts = -1)
        : interpreter_(interpreter)
        , fn_(fn)
        , colPtrs_(std::move(colPtrs))
        , jacFn_(jacFn)
        , jacColPtrs_(std::move(jacColPtrs))
        , range_(range)
        , numParameters_(static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount()))
        , nVars_(nVars)
        , nConsts_(nConsts)
    {
        configurationError_ = Validate(target);
        if (configurationError_) {
            return;
        }
        target_ = target.subspan(range.Start(), range.Size());
        nRowsPad_ = (range.Size() + 7U) & ~std::size_t { 7U };
        // Scratch is only ever written by the corresponding kernel, so a
        // residual-only or interpreter-only object allocates nothing it
        // would never read.
        if (fn_ != nullptr) {
            scratchResiduals_.resize(nRowsPad_);
        }
        // Precomputed once: these point into scratchJac_'s fixed layout, which
        // never changes across Evaluate() calls, so recomputing them per call
        // (as a freshly heap-allocated std::vector, no less) was pure waste on
        // what can be a very hot path. Skipped for residual-only objects
        // (jacFn_ == nullptr), which never dereference jacOutPtrs_.
        if (jacFn_ != nullptr) {
            scratchJac_.resize(nRowsPad_ * numParameters_);
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

    /** The construction-time validation failure every Evaluate() returns, if any. */
    [[nodiscard]] auto ConfigurationError() const noexcept -> std::optional<LeastSquaresError> const&
    {
        return configurationError_;
    }

    [[nodiscard]] auto Evaluate(ConstScalarSpan parameters, ScalarSpan residuals,
        std::optional<ScalarMatrixView> jacobian) const -> tl::expected<void, LeastSquaresError> override
    {
        if (configurationError_) {
            return tl::unexpected(*configurationError_);
        }
        auto const n = range_.Size();
        if (parameters.size() != numParameters_) {
            return tl::unexpected(LeastSquaresError {
                .Code = LeastSquaresErrorCode::InvalidShape, .Expected = numParameters_, .Actual = parameters.size() });
        }
        if (residuals.size() != n) {
            return tl::unexpected(LeastSquaresError {
                .Code = LeastSquaresErrorCode::InvalidShape, .Expected = n, .Actual = residuals.size() });
        }
        if (jacobian && (jacobian->extent(0) != n || jacobian->extent(1) != numParameters_)) {
            return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape,
                .Expected = n,
                .Actual = jacobian->extent(0),
                .Row = n,
                .Column = numParameters_ });
        }

        auto const nRowsPad = static_cast<int32_t>(nRowsPad_);

        if (fn_ != nullptr) {
            fn_(scratchResiduals_.data(), colPtrs_.data(), nRowsPad, parameters.data());
            std::copy_n(scratchResiduals_.data(), n, residuals.data());
        } else {
            auto predicted = interpreter_->Evaluate(parameters, range_, residuals);
            if (!predicted) {
                return tl::unexpected(
                    LeastSquaresError { .Code = LeastSquaresErrorCode::EvaluationFailure, .Cause = predicted.error() });
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            residuals[i] -= target_[i];
        }

        if (jacobian) {
            if (jacFn_ != nullptr) {
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
                    return tl::unexpected(LeastSquaresError {
                        .Code = LeastSquaresErrorCode::EvaluationFailure, .Cause = jacResult.error() });
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
    ConstScalarSpan target_; // empty while configurationError_ is set
    Range range_; // NOLINT(readability-identifier-naming)
    std::size_t numParameters_;
    std::size_t nRowsPad_ = 0;
    mutable std::vector<Scalar> scratchResiduals_;
    mutable std::vector<Scalar> scratchJac_;
    std::vector<float*> jacOutPtrs_; // precomputed pointers into scratchJac_, see ctor
    mutable std::vector<Scalar> jacobianScratch_; // interpreter JacRev fallback, column-major
    int nVars_ = -1;
    int nConsts_ = -1;
    std::optional<LeastSquaresError> configurationError_;

    // Returns the first construction-time violation (see the class comment
    // for the check list). Runs once; allocates only for the nVars < 0
    // VarOrder cross-check.
    [[nodiscard]] auto Validate(ConstScalarSpan target) const -> std::optional<LeastSquaresError>
    {
        constexpr auto maxSize = std::numeric_limits<std::size_t>::max();
        constexpr auto maxRows = static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) - 7U;

        auto const start = range_.Start();
        auto const rows = range_.Size();
        if (start > target.size() || rows > target.size() - start) {
            auto const required = rows > maxSize - start ? maxSize : start + rows;
            return LeastSquaresError {
                .Code = LeastSquaresErrorCode::InvalidShape, .Expected = required, .Actual = target.size()
            };
        }
        if (rows > maxRows) {
            return LeastSquaresError {
                .Code = LeastSquaresErrorCode::InvalidShape, .Expected = maxRows, .Actual = rows
            };
        }
        if (fn_ == nullptr && jacFn_ == nullptr) {
            return std::nullopt;
        }
        if (nConsts_ >= 0 && static_cast<std::size_t>(nConsts_) != numParameters_) {
            return LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape,
                .Expected = static_cast<std::size_t>(nConsts_),
                .Actual = numParameters_ };
        }
        auto const expectedVars
            = nVars_ >= 0 ? static_cast<std::size_t>(nVars_) : JIT::VarOrder(*interpreter_->GetTree()).size();
        if (fn_ != nullptr) {
            if (auto error = ValidateColumns(colPtrs_, expectedVars, 0)) {
                return error;
            }
        }
        if (jacFn_ != nullptr) {
            if (auto error = ValidateColumns(jacColPtrs_, expectedVars, 1)) {
                return error;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] static auto ValidateColumns(std::vector<float const*> const& columns, std::size_t expected,
        std::size_t kernel) -> std::optional<LeastSquaresError>
    {
        if (columns.size() != expected) {
            return LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape,
                .Expected = expected,
                .Actual = columns.size(),
                .Row = kernel };
        }
        auto const nullIt = std::ranges::find(columns, static_cast<float const*>(nullptr));
        if (nullIt != columns.end()) {
            return LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidView,
                .Expected = expected,
                .Actual = columns.size(),
                .Row = kernel,
                .Column = static_cast<std::size_t>(nullIt - columns.begin()) };
        }
        return std::nullopt;
    }
};

} // namespace Operon

#endif // HAVE_ASMJIT
#endif
