// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "pareto_front.hpp"

#include <algorithm>
#include <array>
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

namespace {

// A weighted variable node expands to (weight * variable), contributing
// three logical symbols: Mul, Constant, Variable.
constexpr auto kWeightedNodeCost = 3.0;

// Uniform-prior precision: di = sqrt(12 / fi) comes from Var(Uniform[-c,c]) = (2c)²/12.
constexpr auto kUniformPriorScale = 12.0;

auto ComputeWeightedComplexity(Tree const& tree) -> std::pair<double, double>
{
    static auto const MulHash   = Node{NodeType::Mul}.HashValue;
    static auto const ParamHash = Node{NodeType::Constant}.HashValue;
    Set<Hash> uniqueSymbols;
    auto k = 0.0;
    for (auto const& node : tree.Nodes()) {
        auto const isWeighted = node.IsVariable() && node.Value != Scalar{1};
        k += isWeighted ? kWeightedNodeCost : 1.0;
        uniqueSymbols.insert(node.HashValue);
        if (isWeighted) {
            uniqueSymbols.insert(MulHash);
            uniqueSymbols.insert(ParamHash);
        }
    }
    auto const q      = static_cast<double>(uniqueSymbols.size());
    auto const fCompl = q > 0.0 ? k * std::log(q) : 0.0;
    return {k, fCompl};
}

auto ComputeFBF(double nll, double n, double p, double fCompl) -> double
{
    auto const b             = 1.0 / std::sqrt(n);
    auto const fbfParams     = (p / 2.0) * ((0.5 * std::log(n)) + std::log(Math::Tau) + 1.0 - std::log(3.0)); // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    auto const fbfLikelihood = (1.0 - b) * nll;
    auto const fbf           = fCompl + fbfParams + fbfLikelihood;
    return std::isfinite(fbf) ? fbf : std::numeric_limits<double>::quiet_NaN();
}

template<typename JacMap>
auto ComputeMDL(JacMap const& jacMap, Scalar sigma2, Span<Scalar const> coeffs,
                Tree const& tree, double p, double fCompl, double nll) -> double
{
    constexpr auto eps    = std::numeric_limits<Scalar>::epsilon();
    auto const fisherDiag = (jacMap.colwise().squaredNorm().transpose().array() / sigma2);

    auto cComplexity = fCompl;
    auto cParameters = 0.0;
    auto pi          = 0;
    for (auto const& node : tree.Nodes()) {
        if (node.Optimize) {
            auto const fi = static_cast<double>(fisherDiag(pi));
            auto const di = std::sqrt(kUniformPriorScale / fi);
            auto const ci = std::abs(static_cast<double>(coeffs[pi]));
            if (std::isfinite(ci) && std::isfinite(di) && ci / di >= 1.0) {
                cParameters += (0.5 * std::log(fi)) + std::log(ci); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
            }
            ++pi;
        } else {
            if (std::abs(node.Value) >= static_cast<double>(eps)) {
                cComplexity += std::log(std::abs(node.Value));
            }
        }
    }
    cParameters -= (p / 2.0) * std::log(3.0); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    auto const mdl = cComplexity + cParameters + nll;
    return std::isfinite(mdl) ? mdl : std::numeric_limits<double>::quiet_NaN();
}

auto EscapeJson(std::string const& s) -> std::string
{
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
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
        Interpreter<Scalar, ScalarDispatch> interp{&dtable, ds, &ind->Genotype};
        auto estimTrain = interp.Evaluate(ind->Genotype.GetCoefficients(), trainRange);
        auto estimTest  = interp.Evaluate(ind->Genotype.GetCoefficients(), testRange);

        if (linearScaling) {
            auto [a, b] = FitLeastSquares(Span<Scalar const>{estimTrain}, Span<Scalar const>{targetTrain});
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

        auto const [k, fCompl] = ComputeWeightedComplexity(ind->Genotype);

        auto const n        = static_cast<double>(trainRange.Size());
        auto const sigmaArr = std::array<Scalar, 1>{static_cast<Scalar>(std::sqrt(mseTrain))};
        auto const p        = static_cast<double>(ind->Genotype.GetCoefficients().size());
        auto const nll      = static_cast<double>(GaussianLikelihood<Scalar>::ComputeLikelihood(
                                  {estimTrain.data(), estimTrain.size()},
                                  targetTrain,
                                  {sigmaArr.data(), sigmaArr.size()}));

        auto const fbf = ComputeFBF(nll, n, p, fCompl);

        auto const coeffs  = ind->Genotype.GetCoefficients();
        auto const jac     = interp.JacRev(coeffs, trainRange);
        auto const nrows   = static_cast<Eigen::Index>(trainRange.Size());
        auto const ncols   = static_cast<Eigen::Index>(coeffs.size());
        Eigen::Map<Eigen::Matrix<Scalar, -1, -1> const> jacMap(jac.data(), nrows, ncols);
        auto const mdl = ComputeMDL(jacMap, static_cast<Scalar>(mseTrain), coeffs, ind->Genotype, p, fCompl, nll);

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
