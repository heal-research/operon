// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_FISHER_INFORMATION_HPP
#define OPERON_FISHER_INFORMATION_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <tl/expected.hpp>

#include "operon/core/memory_view.hpp"

namespace Operon {

enum class FisherErrorCode : std::uint8_t {
    InvalidShape,
    InvalidSigma,
    NonFiniteResult,
};

struct FisherError {
    FisherErrorCode Code {FisherErrorCode::InvalidShape};
    std::size_t Expected {};
    std::size_t Actual {};
};

/** F = J^T diag(1/sigma_i^2) J, written into caller-owned fisher (NumParameters x NumParameters). */
[[nodiscard]] inline auto ComputeFisherMatrix(
    ConstScalarMatrixView jacobian,
    ConstScalarSpan sigma,
    ScalarMatrixView fisher)
    -> tl::expected<void, FisherError>
{
    auto const n = jacobian.extent(0); // observations
    auto const p = jacobian.extent(1); // parameters

    if (fisher.extent(0) != p || fisher.extent(1) != p) {
        return tl::unexpected(FisherError { .Code = FisherErrorCode::InvalidShape, .Expected = p, .Actual = fisher.extent(0) });
    }
    if (!sigma.empty() && sigma.size() != 1 && sigma.size() != n) {
        return tl::unexpected(FisherError { .Code = FisherErrorCode::InvalidShape, .Expected = n, .Actual = sigma.size() });
    }
    for (auto const s : sigma) {
        if (!std::isfinite(static_cast<double>(s)) || s <= Scalar { 0 }) {
            return tl::unexpected(FisherError { .Code = FisherErrorCode::InvalidSigma });
        }
    }

    auto const invVarAt = [&sigma](std::size_t i) -> AccumulationScalar {
        AccumulationScalar s { 1 };
        if (!sigma.empty()) {
            s = static_cast<AccumulationScalar>(sigma.size() == 1 ? sigma[0] : sigma[i]);
        }
        return AccumulationScalar { 1 } / (s * s);
    };

    for (std::size_t a = 0; a < p; ++a) {
        for (std::size_t b = a; b < p; ++b) {
            AccumulationScalar sum {0};
            for (std::size_t i = 0; i < n; ++i) {
                sum += invVarAt(i) * static_cast<AccumulationScalar>(At(jacobian, i, a)) * static_cast<AccumulationScalar>(At(jacobian, i, b));
            }
            if (!std::isfinite(sum)) {
                return tl::unexpected(FisherError { .Code = FisherErrorCode::NonFiniteResult });
            }
            auto const value = static_cast<Scalar>(sum);
            At(fisher, a, b) = value;
            At(fisher, b, a) = value;
        }
    }

    return {};
}

} // namespace Operon

#endif
