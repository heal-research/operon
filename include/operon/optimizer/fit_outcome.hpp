// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_FIT_OUTCOME_HPP
#define OPERON_FIT_OUTCOME_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <utility>
#include <variant>
#include <vector>

#include <tl/expected.hpp>

#include "operon/core/comparison.hpp"
#include "operon/core/interpreter_error.hpp"
#include "operon/core/types.hpp"
#include "operon/optimizer/gradient_cost.hpp"
#include "operon/optimizer/least_squares.hpp"
#include "operon/optimizer/least_squares_gradient_adapter.hpp"

namespace Operon {

// Fields every fit always produces. FitResult, FitFailure,
// FitEvaluationError, and FitConfigurationError all carry them, so callers
// that need e.g. FunctionEvaluations regardless of outcome use Diagnostics().
struct FitDiagnostics {
    std::vector<Operon::Scalar> InitialParameters;
    std::vector<Operon::Scalar> FinalParameters;
    Operon::Scalar InitialCost {};
    Operon::Scalar FinalCost {};
    int Iterations {};
    int FunctionEvaluations {};
    int JacobianEvaluations {};
};

struct FitResult : FitDiagnostics {}; // FinalCost improved on InitialCost
struct FitFailure : FitDiagnostics {}; // valid fit that did not improve (incl. non-finite cost)
struct FitEvaluationError : FitDiagnostics {
    GradientError Error;
};
struct FitConfigurationError : FitDiagnostics {
    WeightError Error;
};

using FitError = std::variant<FitFailure, FitEvaluationError, FitConfigurationError>;
using FitOutcome = tl::expected<FitResult, FitError>;

[[nodiscard]] inline auto Diagnostics(FitOutcome const& outcome) -> FitDiagnostics const&
{
    if (outcome) {
        return *outcome;
    }
    return std::visit([](auto const& error) -> FitDiagnostics const& { return error; }, outcome.error());
}

[[nodiscard]] inline auto EvaluationError(FitOutcome const& outcome) -> FitEvaluationError const*
{
    return outcome ? nullptr : std::get_if<FitEvaluationError>(&outcome.error());
}

[[nodiscard]] inline auto ConfigurationError(FitOutcome const& outcome) -> FitConfigurationError const*
{
    return outcome ? nullptr : std::get_if<FitConfigurationError>(&outcome.error());
}

namespace detail {
    inline auto CheckSuccess(double initialCost, double finalCost)
    {
        constexpr auto CHECK_NAN { true };
        return Operon::Less<CHECK_NAN> {}(finalCost, initialCost);
    }

    // Replaces the near-identical summary-assembly tail block that used to
    // be repeated at the end of every Optimize() override: each override
    // builds one FitDiagnostics via aggregate init, then returns
    // MakeFitOutcome(std::move(diag)) as its last line.
    inline auto MakeFitOutcome(FitDiagnostics diag) -> FitOutcome
    {
        if (CheckSuccess(diag.InitialCost, diag.FinalCost)) {
            return FitResult { std::move(diag) };
        }
        return tl::unexpected(FitFailure { std::move(diag) });
    }
    inline auto MakeFitEvaluationError(GradientError error, FitDiagnostics diag) -> FitOutcome
    {
        FitEvaluationError failure;
        static_cast<FitDiagnostics&>(failure) = std::move(diag);
        failure.Error = std::move(error);
        return tl::unexpected(FitError { std::move(failure) });
    }

    // Convenience for the (still common) case of a raw interpreter failure
    // with no separate numerical cost wrapping it.
    inline auto MakeFitEvaluationError(InterpreterError error, FitDiagnostics diag) -> FitOutcome
    {
        return MakeFitEvaluationError(GradientError { .Code = GradientErrorCode::EvaluationFailure, .Cause = std::move(error) }, std::move(diag));
    }

    inline auto MakeFitConfigurationError(WeightError error, FitDiagnostics diag) -> FitOutcome
    {
        FitConfigurationError failure;
        static_cast<FitDiagnostics&>(failure) = std::move(diag);
        failure.Error = error;
        return tl::unexpected(FitError { std::move(failure) });
    }

    // An evaluation error raised before any cost was evaluated (for example a
    // shape that is rejected up front): there is no valid initial cost, so both
    // costs are NaN, FinalParameters equals InitialParameters, and every
    // counter stays zero.
    inline auto MakeUnevaluatedFitEvaluationError(GradientError error, FitDiagnostics diag) -> FitOutcome
    {
        diag.FinalParameters = diag.InitialParameters;
        diag.InitialCost = diag.FinalCost = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        return MakeFitEvaluationError(std::move(error), std::move(diag));
    }

    // Narrows a size_t iteration option to T, saturating at T's maximum instead
    // of wrapping (a wrapped budget could be negative or silently zero).
    template <std::integral T>
    [[nodiscard]] constexpr auto SaturatingCast(std::size_t value) -> T
    {
        constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<T>::max());
        return static_cast<T>(std::min(value, limit));
    }

    // a * b, saturating at the largest size_t instead of wrapping.
    [[nodiscard]] constexpr auto SaturatingMultiply(std::size_t a, std::size_t b) -> std::size_t
    {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
            return std::numeric_limits<std::size_t>::max();
        }
        return a * b;
    }

    // A tree without coefficients has nothing to solve. The solvers are not
    // run (Tiny's summary would otherwise leak its -1 sentinels), the endpoint
    // cost is evaluated once, and every backend reports
    // InitialCost == FinalCost, Iterations == 0, and the unchanged (empty)
    // parameter vector. The outcome is therefore a FitFailure (no improvement)
    // unless the single evaluation itself failed.
    inline auto ZeroParameterDiagnostics(FitDiagnostics diag, Operon::Scalar cost, int functionEvaluations) -> FitDiagnostics
    {
        diag.FinalParameters = diag.InitialParameters;
        diag.InitialCost = diag.FinalCost = cost;
        diag.Iterations = 0;
        diag.FunctionEvaluations = functionEvaluations;
        diag.JacobianEvaluations = 0;
        return diag;
    }
} // namespace detail

} // namespace Operon

#endif
