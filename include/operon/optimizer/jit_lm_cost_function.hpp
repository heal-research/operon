// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#pragma once

#ifdef HAVE_ASMJIT

#include <atomic>
#include <vector>

#include <Eigen/Core>
#include <gsl/pointers>

#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/backend/jit/jit_compiler.hpp"

namespace Operon {

// LM cost function that replaces the interpreter residual path with a JIT-compiled
// forward evaluation while keeping interpreter JacRev for the Jacobian.
//
// Residual evaluation is 1.5–5× faster than the interpreter depending on
// tree complexity, so each LM iteration saves proportionally on forward evals.
// JacRev stays interpreter-backed (AD) because reverse-mode AD cannot yet be
// emitted by the JIT compiler.
//
// colPtrs[i] must point to the dataset column for compiled->varOrder[i],
// already offset to range.Start() so the function only needs [0, nRows).
template<typename T = Operon::Scalar, int StorageOrder = Eigen::ColMajor>
struct JitLMCostFunction {
    static auto constexpr Storage{ StorageOrder };
    using Scalar = Operon::Scalar;

    static_assert(std::is_same_v<T, float>,
        "JitLMCostFunction requires float precision (EvalFn operates on float arrays)");

    enum {
        NUM_RESIDUALS  = Eigen::Dynamic, // NOLINT
        NUM_PARAMETERS = Eigen::Dynamic, // NOLINT
    };

    JitLMCostFunction(gsl::not_null<InterpreterBase<T> const*> interpreter,
                      JIT::EvalFn                              fn,
                      std::vector<float const*>                colPtrs,
                      Operon::Span<Operon::Scalar const>       target,
                      Operon::Range                            range)
        : interpreter_(interpreter)
        , fn_(fn)
        , colPtrs_(std::move(colPtrs))
        , target_(target)
        , range_(range)
        , numResiduals_(range.Size())
        , numParameters_(static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount()))
    {}

    inline auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        EXPECT(target_.size() == numResiduals_);
        EXPECT(parameters != nullptr);
        Operon::Span<Operon::Scalar const> params{ parameters, numParameters_ };

        if (jacobian != nullptr) {
            ++jacobianCallCount_;
            Operon::Span<Operon::Scalar> jac{jacobian, numResiduals_ * numParameters_};
            interpreter_->JacRev(params, range_, jac);
        }

        if (residuals != nullptr) {
            ++residualCallCount_;
            auto const nRows = static_cast<int32_t>(numResiduals_);
            fn_(residuals, colPtrs_.data(), nRows, parameters);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> x(residuals, static_cast<Eigen::Index>(numResiduals_));
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> y(target_.data(), static_cast<Eigen::Index>(numResiduals_));
            x -= y;
        }
        return true;
    }

    auto operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool
    {
        return Evaluate(parameters, residuals, jacobian);
    }

    [[nodiscard]] auto NumResiduals()  const -> int { return static_cast<int>(numResiduals_); }
    [[nodiscard]] auto NumParameters() const -> int { return static_cast<int>(numParameters_); }

    using JacobianType = Eigen::Matrix<Operon::Scalar, -1, -1>;
    using QRSolver     = Eigen::ColPivHouseholderQR<JacobianType>;

    auto operator()(Eigen::Matrix<Scalar, -1, 1> const& input,
                    Eigen::Matrix<Scalar, -1, 1>&        residual) const -> int
    {
        Evaluate(input.data(), residual.data(), nullptr);
        return 0;
    }

    auto df(Eigen::Matrix<Scalar, -1, 1> const& input, // NOLINT
            Eigen::Matrix<Scalar, -1, -1>&       jacobian) const -> int
    {
        static_assert(StorageOrder == Eigen::ColMajor,
            "Eigen::LevenbergMarquardt requires column-major Jacobian.");
        Evaluate(input.data(), nullptr, jacobian.data());
        return 0;
    }

    [[nodiscard]] auto values() const -> int { return NumResiduals(); }  // NOLINT
    [[nodiscard]] auto inputs() const -> int { return NumParameters(); } // NOLINT

    [[nodiscard]] auto ResidualCalls()  const -> std::size_t { return residualCallCount_.load(); }
    [[nodiscard]] auto JacobianCalls()  const -> std::size_t { return jacobianCallCount_.load(); }

private:
    gsl::not_null<InterpreterBase<T> const*> interpreter_;
    JIT::EvalFn                              fn_;
    std::vector<float const*>                colPtrs_;
    Operon::Span<Operon::Scalar const>       target_;
    Operon::Range const                      range_;   // NOLINT
    std::size_t                              numResiduals_;
    std::size_t                              numParameters_;

    mutable std::atomic_size_t jacobianCallCount_{0};
    mutable std::atomic_size_t residualCallCount_{0};
};

} // namespace Operon

#endif // HAVE_ASMJIT
