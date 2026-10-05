// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "pareto_front.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <fmt/os.h>

#include "operon/core/types.hpp"
#include "operon/error_metrics/error_metrics.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/information_criteria/information_criteria.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/linear_scaling.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"

namespace Operon {

namespace {

    auto EscapeJson(std::string const& s) -> std::string
    {
        std::string result;
        result.reserve(s.size());
        for (char const c : s) {
            if (c == '"') {
                result += "\\\"";
            } else if (c == '\\') {
                result += "\\\\";
            } else {
                result += c;
            }
        }
        return result;
    }

} // namespace

auto WriteParetoFront(std::string const& path, Operon::Span<Individual const> population, ScalarDispatch const& dtable,
    Problem const& problem) -> void
{
    auto const* ds = problem.GetDataset();
    auto const trainRange = problem.TrainingRange();
    auto const testRange = problem.TestRange();
    auto const targetTrain = problem.TargetValues(trainRange);
    auto const targetTest = problem.TargetValues(testRange);

    std::vector<Individual const*> front;
    for (auto const& ind : population) {
        if (ind.Rank == 0) {
            front.push_back(&ind);
        }
    }
    std::ranges::sort(front, [](auto const* a, auto const* b) -> bool { return (*a)[0] < (*b)[0]; });

    auto jsonNum = [](double v) -> std::string {
        if (!std::isfinite(v)) {
            return "null";
        }
        return fmt::format("{:.17g}", v);
    };

    auto out = fmt::output_file(path);
    out.print("[\n");
    for (auto i = 0UL; i < front.size(); ++i) {
        auto const* ind = front[i];
        Interpreter<Scalar, ScalarDispatch> const interp { &dtable, ds, &ind->Genotype };
        auto estimTrainResult = interp.Evaluate(ind->Genotype.GetCoefficients(), trainRange);
        if (!estimTrainResult) {
            throw std::runtime_error(FormatInterpreterError(estimTrainResult.error()));
        }
        auto estimTrain = std::move(*estimTrainResult);
        auto estimTestResult = interp.Evaluate(ind->Genotype.GetCoefficients(), testRange);
        if (!estimTestResult) {
            throw std::runtime_error(FormatInterpreterError(estimTestResult.error()));
        }
        auto estimTest = std::move(*estimTestResult);

        // Scale factor for the raw tree's Jacobian: the reported model is
        // y = a * tree(x; coeffs) + b when linear scaling is on, so
        // d(y)/d(coeffs) = a * d(tree)/d(coeffs) — this must multiply the
        // Jacobian used for the MDL Fisher-information term below, or the
        // parameter-cost term is biased by a missing a^2 factor whenever
        // the fitted slope isn't ~1.
        auto scale = Scalar { 1 };
        auto const scaling = Operon::FitLinearScaling(ind->Genotype, problem, dtable, trainRange);
        if (scaling) {
            scale = static_cast<Scalar>(scaling->Scale);
            scaling->ApplyInPlace(Operon::Span<Operon::Scalar> { estimTrain });
            scaling->ApplyInPlace(Operon::Span<Operon::Scalar> { estimTest });
        }

        auto const r2Train = -R2 {}(estimTrain, targetTrain);
        auto const r2Test = -R2 {}(estimTest, targetTest);
        auto const mseTrain = MSE {}(estimTrain, targetTrain);
        auto const mseTest = MSE {}(estimTest, targetTest);
        auto const nmseTrain = NMSE {}(estimTrain, targetTrain);
        auto const nmseTest = NMSE {}(estimTest, targetTest);
        auto const maeTrain = MAE {}(estimTrain, targetTrain);
        auto const maeTest = MAE {}(estimTest, targetTest);

        auto const k = WeightedComplexity(ind->Genotype).first; // fComplexity: computed internally by MDL/FBF below

        auto const n = static_cast<double>(trainRange.Size());
        auto const profiledSigma
            = std::max(static_cast<Scalar>(std::sqrt(mseTrain)), std::numeric_limits<Scalar>::epsilon());
        auto const sigmaArr = std::array<Scalar, 1> { profiledSigma };
        auto const nll = static_cast<double>(GaussianLikelihood<Scalar>::ComputeLikelihood(
            { estimTrain.data(), estimTrain.size() }, targetTrain, { sigmaArr.data(), sigmaArr.size() }));

        auto const fbf = FractionalBayesFactor(ind->Genotype, n, nll);

        auto const coeffs = ind->Genotype.GetCoefficients();
        auto const columns = coeffs.size();
        auto jacobianStorage = std::vector<Scalar>(trainRange.Size() * columns);
        using Extents = std::dextents<MemoryIndex, 2>;
        using Mapping = std::layout_stride::mapping<Extents>;
        // Column stride must stay nonzero for an empty training range (extent 0
        // addresses no element, but a zero stride violates layout_stride preconditions).
        auto const columnStride = std::max<MemoryIndex>(trainRange.Size(), MemoryIndex { 1 });
        auto jacobian = ScalarMatrixView { jacobianStorage.data(),
            Mapping { Extents { trainRange.Size(), columns }, std::array<MemoryIndex, 2> { 1, columnStride } } };
        if (auto result = interp.JacRev(coeffs, trainRange, jacobianStorage); !result) {
            throw std::runtime_error(FormatInterpreterError(result.error()));
        }
        for (std::size_t row = 0; row < trainRange.Size(); ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                At(jacobian, row, column) *= scale;
            }
        }
        auto fisherDiagonal = std::vector<Scalar>(columns);
        auto mdl = std::numeric_limits<double>::quiet_NaN(); // exported as null for a degenerate member
        if (auto result
            = GaussianLikelihood<Scalar>::ComputeFisherDiagonal(estimTrain, jacobian, sigmaArr, fisherDiagonal);
            result) {
            mdl = MinimumDescriptionLength(ind->Genotype, coeffs, fisherDiagonal, nll);
        } else if (result.error().Code != FisherErrorCode::NonFiniteResult
            && result.error().Code != FisherErrorCode::InvalidSigma) {
            throw std::runtime_error("failed to compute Fisher diagonal");
        }

        std::string objArr = "[";
        for (auto j = 0UL; j < ind->Fitness.size(); ++j) {
            if (j > 0) {
                objArr += ", ";
            }
            objArr += jsonNum(ind->Fitness[j]);
        }
        objArr += "]";

        if (i > 0) {
            out.print(",\n");
        }
        out.print("  {{\"id\": {}, \"expression\": \"{}\", \"length\": {}, \"complexity\": {}, \"objectives\": {},\n"
                  "   \"r2_train\": {}, \"r2_test\": {},\n"
                  "   \"mse_train\": {}, \"mse_test\": {},\n"
                  "   \"nmse_train\": {}, \"nmse_test\": {},\n"
                  "   \"mae_train\": {}, \"mae_test\": {},\n"
                  "   \"mdl\": {}, \"fbf\": {}}}",
            i, EscapeJson(fmt::format("{:infix:roundtrip}", Fmt::TreeFormatArgs { ind->Genotype, *ds })),
            ind->Genotype.AdjustedLength(), static_cast<size_t>(k), objArr, jsonNum(r2Train), jsonNum(r2Test),
            jsonNum(mseTrain), jsonNum(mseTest), jsonNum(nmseTrain), jsonNum(nmseTest), jsonNum(maeTrain),
            jsonNum(maeTest), jsonNum(mdl), jsonNum(fbf));
    }
    out.print("\n]\n");
}

} // namespace Operon
