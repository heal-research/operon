// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "pareto_front.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <fmt/core.h>
#include <fmt/os.h>

#include "operon/core/node.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"

namespace Operon {

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

    // collect rank-0 individuals sorted by first objective
    std::vector<Individual const*> front;
    for (auto const& ind : population) {
        if (ind.Rank == 0) { front.push_back(&ind); }
    }
    std::ranges::sort(front, [](auto const* a, auto const* b) { return (*a)[0] < (*b)[0]; });

    auto jsonNum = [](double v) -> std::string {
        if (!std::isfinite(v)) { return "null"; }
        return fmt::format("{:.17g}", v);
    };

    auto out = fmt::output_file(path);
    out.print("[\n");
    for (auto i = 0UL; i < front.size(); ++i) {
        auto const* ind = front[i];
        Interpreter<Scalar, ScalarDispatch> interp{&dtable, ds, &ind->Genotype};
        auto estimTrain = interp.Evaluate(ind->Genotype.GetCoefficients(), trainRange);
        auto estimTest  = interp.Evaluate(ind->Genotype.GetCoefficients(), testRange);

        if (linearScaling) {
            auto [a, b] = FitLeastSquares(
                Span<Scalar const>{estimTrain},
                Span<Scalar const>{targetTrain});
            auto const as = static_cast<Scalar>(a);
            auto const bs = static_cast<Scalar>(b);
            for (auto& v : estimTrain) { v = (v * as) + bs; }
            for (auto& v : estimTest)  { v = (v * as) + bs; }
        }

        auto const r2Train   = -R2{}(estimTrain, targetTrain);
        auto const r2Test    = -R2{}(estimTest, targetTest);
        auto const mseTrain  =  MSE{}(estimTrain, targetTrain);
        auto const mseTest   =  MSE{}(estimTest, targetTest);
        auto const nmseTrain =  NMSE{}(estimTrain, targetTrain);
        auto const nmseTest  =  NMSE{}(estimTest, targetTest);
        auto const maeTrain  =  MAE{}(estimTrain, targetTrain);
        auto const maeTest   =  MAE{}(estimTest, targetTest);

        // weighted node count and structural complexity
        static auto const MulHash   = Node{NodeType::Mul}.HashValue;
        static auto const ParamHash = Node{NodeType::Constant}.HashValue;
        Set<Hash> uniqueSymbols;
        auto k = 0.0;
        for (auto const& node : ind->Genotype.Nodes()) {
            auto const isWeighted = node.IsVariable() && node.Value != Scalar{1};
            k += isWeighted ? 3.0 : 1.0;
            uniqueSymbols.insert(node.HashValue);
            if (isWeighted) {
                uniqueSymbols.insert(MulHash);
                uniqueSymbols.insert(ParamHash);
            }
        }
        auto const q      = static_cast<double>(uniqueSymbols.size());
        auto const fCompl = q > 0.0 ? k * std::log(q) : 0.0;

        // MLE sigma estimate from training residuals
        auto const n        = static_cast<double>(trainRange.Size());
        auto const sigmaHat = static_cast<Scalar>(std::sqrt(mseTrain));
        Scalar sigmaArr[]   = {sigmaHat};
        Span<Scalar const> sigmaSpan{sigmaArr, 1};

        // FBF (no Jacobian required)
        auto const p           = static_cast<double>(ind->Genotype.GetCoefficients().size());
        auto const b           = 1.0 / std::sqrt(n);
        auto const fbfParams   = (p / 2.0) * (0.5 * std::log(n) + std::log(Math::Tau) + 1.0 - std::log(3.0));
        auto const nll         = static_cast<double>(GaussianLikelihood<Scalar>::ComputeLikelihood(
                                     {estimTrain.data(), estimTrain.size()},
                                     targetTrain,
                                     sigmaSpan));
        auto const fbfLikelihood = (1.0 - b) * nll;
        auto fbf = fCompl + fbfParams + fbfLikelihood;
        if (!std::isfinite(fbf)) { fbf = std::numeric_limits<double>::quiet_NaN(); }

        // MDL (requires Jacobian for Fisher diagonal)
        // Fisher = J^T J / σ²  (Gaussian)
        auto const coeffs  = ind->Genotype.GetCoefficients();
        auto const jac     = interp.JacRev(coeffs, trainRange);
        auto const nrows   = static_cast<Eigen::Index>(trainRange.Size());
        auto const ncols   = static_cast<Eigen::Index>(coeffs.size());
        Eigen::Map<Eigen::Matrix<Scalar, -1, -1> const> jacMap(jac.data(), nrows, ncols);
        auto const sigma2     = static_cast<Scalar>(mseTrain);
        auto const fisherDiag = (jacMap.colwise().squaredNorm().transpose().array() / sigma2);

        auto cComplexity = fCompl;
        auto cParameters = 0.0;
        constexpr auto eps = std::numeric_limits<Scalar>::epsilon();
        auto pi = 0;
        for (auto const& node : ind->Genotype.Nodes()) {
            if (node.Optimize) {
                auto const fi = static_cast<double>(fisherDiag(pi));
                auto const di = std::sqrt(12.0 / fi);
                auto const ci = std::abs(static_cast<double>(coeffs[pi]));
                if (std::isfinite(ci) && std::isfinite(di) && ci / di >= 1.0) {
                    cParameters += 0.5 * std::log(fi) + std::log(ci);
                }
                ++pi;
            } else {
                if (std::abs(node.Value) >= static_cast<double>(eps)) {
                    cComplexity += std::log(std::abs(node.Value));
                }
            }
        }
        cParameters -= (p / 2.0) * std::log(3.0);
        auto const cLikelihood = nll;
        auto mdl = cComplexity + cParameters + cLikelihood;
        if (!std::isfinite(mdl)) { mdl = std::numeric_limits<double>::quiet_NaN(); }

        auto expr = InfixFormatter::Format(ind->Genotype, *ds, std::numeric_limits<Scalar>::digits10);
        std::string escaped;
        escaped.reserve(expr.size());
        for (char c : expr) {
            if      (c == '"')  { escaped += "\\\""; }
            else if (c == '\\') { escaped += "\\\\"; }
            else                { escaped += c; }
        }

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
            i, escaped, ind->Genotype.AdjustedLength(), static_cast<size_t>(k), objArr,
            jsonNum(r2Train), jsonNum(r2Test),
            jsonNum(mseTrain), jsonNum(mseTest),
            jsonNum(nmseTrain), jsonNum(nmseTest),
            jsonNum(maeTrain), jsonNum(maeTest),
            jsonNum(mdl), jsonNum(fbf));
    }
    out.print("\n]\n");
}

} // namespace Operon
