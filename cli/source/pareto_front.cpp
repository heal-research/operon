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
#include "operon/information_criteria/information_criteria.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"

namespace Operon {

namespace {

auto EscapeJson(std::string const& s) -> std::string
{
    std::string result;
    result.reserve(s.size());
    for (char const c : s) {
        if      (c == '"')  { result += "\\\""; }
        else if (c == '\\') { result += "\\\\"; }
        else                { result += c; }
    }
    return result;
}

} // namespace

auto WriteParetoFront(std::string const& path,
                      Operon::Span<Individual const> population,
                      ScalarDispatch const& dtable,
                      Problem const& problem,
                      bool linearScaling) -> void
{
    auto const* ds         = problem.GetDataset();
    auto const trainRange  = problem.TrainingRange();
    auto const testRange   = problem.TestRange();
    auto const targetTrain = problem.TargetValues(trainRange);
    auto const targetTest  = problem.TargetValues(testRange);

    std::vector<Individual const*> front;
    for (auto const& ind : population) {
        if (ind.Rank == 0) { front.push_back(&ind); }
    }
    std::ranges::sort(front, [](auto const* a, auto const* b) -> bool { return (*a)[0] < (*b)[0]; });

    auto jsonNum = [](double v) -> std::string {
        if (!std::isfinite(v)) { return "null"; }
        return fmt::format("{:.17g}", v);
    };

    auto out = fmt::output_file(path);
    out.print("[\n");
    for (auto i = 0UL; i < front.size(); ++i) {
        auto const* ind = front[i];
        Interpreter<Scalar, ScalarDispatch> const interp{&dtable, ds, &ind->Genotype};
        auto estimTrain = interp.Evaluate(ind->Genotype.GetCoefficients(), trainRange);
        auto estimTest  = interp.Evaluate(ind->Genotype.GetCoefficients(), testRange);

        // Scale factor for the raw tree's Jacobian: the reported model is
        // y = a * tree(x; coeffs) + b when linear scaling is on, so
        // d(y)/d(coeffs) = a * d(tree)/d(coeffs) — this must multiply the
        // Jacobian used for the MDL Fisher-information term below, or the
        // parameter-cost term is biased by a missing a^2 factor whenever
        // the fitted slope isn't ~1.
        auto scale = Scalar{1};
        if (linearScaling) {
            auto [a, b] = FitLeastSquares(Span<Scalar const>{estimTrain}, Span<Scalar const>{targetTrain});
            scale = static_cast<Scalar>(a);
            auto const bs = static_cast<Scalar>(b);
            for (auto& v : estimTrain) { v = (v * scale) + bs; }
            for (auto& v : estimTest)  { v = (v * scale) + bs; }
        }

        auto const r2Train   = -R2{}(estimTrain, targetTrain);
        auto const r2Test    = -R2{}(estimTest, targetTest);
        auto const mseTrain  =  MSE{}(estimTrain, targetTrain);
        auto const mseTest   =  MSE{}(estimTest, targetTest);
        auto const nmseTrain =  NMSE{}(estimTrain, targetTrain);
        auto const nmseTest  =  NMSE{}(estimTest, targetTest);
        auto const maeTrain  =  MAE{}(estimTrain, targetTrain);
        auto const maeTest   =  MAE{}(estimTest, targetTest);

        auto const k = WeightedComplexity(ind->Genotype).first; // fComplexity: computed internally by MDL/FBF below

        auto const n        = static_cast<double>(trainRange.Size());
        auto const sigmaArr = std::array<Scalar, 1>{static_cast<Scalar>(std::sqrt(mseTrain))};
        auto const nll      = static_cast<double>(GaussianLikelihood<Scalar>::ComputeLikelihood(
                                  {estimTrain.data(), estimTrain.size()},
                                  targetTrain,
                                  {sigmaArr.data(), sigmaArr.size()}));

        auto const fbf = FractionalBayesFactor(ind->Genotype, n, nll);

        auto const coeffs  = ind->Genotype.GetCoefficients();
        auto jac           = interp.JacRev(coeffs, trainRange);
        jac *= scale; // d(a*tree)/d(coeffs) = a * d(tree)/d(coeffs); scale == 1 when linearScaling is off
        auto fisherMatrix  = GaussianLikelihood<Scalar>::ComputeFisherMatrix(
                                  {estimTrain.data(), estimTrain.size()},
                                  {jac.data(), static_cast<std::size_t>(jac.size())},
                                  {sigmaArr.data(), sigmaArr.size()});
        auto const mdl = MinimumDescriptionLength(ind->Genotype, coeffs, fisherMatrix.diagonal().array(), nll);

        std::string objArr = "[";
        for (auto j = 0UL; j < ind->Fitness.size(); ++j) {
            if (j > 0) { objArr += ", "; }
            objArr += jsonNum(ind->Fitness[j]);
        }
        objArr += "]";

        if (i > 0) { out.print(",\n"); }
        out.print(
            "  {{\"id\": {}, \"expression\": \"{}\", \"length\": {}, \"complexity\": {}, \"objectives\": {},\n"
            "   \"r2_train\": {}, \"r2_test\": {},\n"
            "   \"mse_train\": {}, \"mse_test\": {},\n"
            "   \"nmse_train\": {}, \"nmse_test\": {},\n"
            "   \"mae_train\": {}, \"mae_test\": {},\n"
            "   \"mdl\": {}, \"fbf\": {}}}",
            i, EscapeJson(InfixFormatter::Format(ind->Genotype, *ds, std::numeric_limits<Scalar>::digits10)),
            ind->Genotype.AdjustedLength(), static_cast<size_t>(k), objArr,
            jsonNum(r2Train), jsonNum(r2Test),
            jsonNum(mseTrain), jsonNum(mseTest),
            jsonNum(nmseTrain), jsonNum(nmseTest),
            jsonNum(maeTrain), jsonNum(maeTest),
            jsonNum(mdl), jsonNum(fbf));
    }
    out.print("\n]\n");
}

} // namespace Operon
