// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LM_WEIGHTS_HPP
#define OPERON_LM_WEIGHTS_HPP

#include <Eigen/Core>
#include <algorithm>
#include <tl/expected.hpp>

#include "operon/core/contracts.hpp"
#include "operon/core/types.hpp"

namespace Operon {

struct LMWeightError {
    enum class Code {
        SizeMismatch,
        NegativeValue,
    };

    Code Kind;
    std::size_t Index {};
};

// Standard WLS-via-LM trick: scaling both the residual and its Jacobian row by
// sqrt(w_i) makes the unweighted LM/GN normal equations solve the weighted
// problem (sum(w_i * r_i^2)) instead.
//
// Called with `weights` already sliced down to the range-local (numResiduals-sized)
// span: an interpreter cost's caller takes the whole-column target/weights and
// slices once there before calling this, since the cost never mini-batches.
[[nodiscard]] inline auto TryValidateLMWeights(Operon::Span<Operon::Scalar const> weights, std::size_t numResiduals)
    -> tl::expected<void, LMWeightError>
{
    if (!weights.empty() && weights.size() != numResiduals) {
        return tl::unexpected(LMWeightError { LMWeightError::Code::SizeMismatch });
    }
    auto const it = std::ranges::find_if(weights, [](auto weight) { return weight < Operon::Scalar { 0 }; });
    if (it != weights.end()) {
        return tl::unexpected(LMWeightError {
            LMWeightError::Code::NegativeValue,
            static_cast<std::size_t>(std::distance(weights.begin(), it)),
        });
    }
    return {};
}

inline void ValidateLMWeights(Operon::Span<Operon::Scalar const> weights, std::size_t numResiduals)
{
    EXPECT(TryValidateLMWeights(weights, numResiduals).has_value());
}

inline void ApplyLMResidualWeights(Operon::Span<Operon::Scalar const> weights, Operon::Scalar* residuals, std::size_t numResiduals)
{
    if (weights.empty()) {
        return;
    }
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> x(residuals, static_cast<Eigen::Index>(numResiduals));
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> w(weights.data(), static_cast<Eigen::Index>(numResiduals));
    x *= w.sqrt();
}

inline void ApplyLMJacobianWeights(Operon::Span<Operon::Scalar const> weights, Operon::Scalar* jacobian, std::size_t numResiduals, std::size_t numParameters)
{
    if (weights.empty()) {
        return;
    }
    Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, -1>> j(jacobian, static_cast<Eigen::Index>(numResiduals), static_cast<Eigen::Index>(numParameters));
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> w(weights.data(), static_cast<Eigen::Index>(numResiduals));
    j.array().colwise() *= w.sqrt();
}

} // namespace Operon

#endif
