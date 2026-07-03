// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../operon_test.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/information_criteria/information_criteria.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"

namespace Operon::Test {

// Regression test for the linear-scaling MDL bug (fixed in 6a709bf2): when
// a fitted model is y = a * tree(x; coeffs) + b, the Jacobian used to
// derive the Fisher information must be scaled by `a`
// (d(a*tree)/d(coeffs) = a * d(tree)/d(coeffs)) before it reaches
// MinimumDescriptionLength, or the parameter-cost term silently loses a
// log(a) contribution per significant parameter. This locks in that
// invariant directly against MinimumDescriptionLength/GaussianLikelihood::
// ComputeFisherMatrix, independent of any CLI/evaluator glue code.
TEST_CASE("Minimum description length reflects Jacobian scale", "[information-criteria][mdl]")
{
    Operon::Dataset ds(std::vector<std::string>{"x1"},
                       std::vector<std::vector<Operon::Scalar>>{
                           {-1.8F, -1.1F, -0.4F, 0.3F, 1.0F, 1.7F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};
    auto const x1Hash = ds.GetVariable("x1")->Hash;

    // A single weighted-variable leaf node (Value = weight != 1) is one
    // Optimize-flagged parameter — built manually rather than via
    // InfixParser, which expands written multiplication ("15 * x1") into a
    // separate Mul(Constant, Variable) subtree with two parameters instead
    // of folding it into the variable's own weight.
    Operon::Node v(Operon::NodeType::Variable);
    v.HashValue = v.CalculatedHashValue = x1Hash;
    v.Value = 15.0F;
    Operon::Tree tree{ Operon::Vector<Operon::Node>{v} };

    auto coeffs = tree.GetCoefficients();
    REQUIRE(coeffs.size() == 1);

    using Interp = Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>>;
    Interp const interpreter{&dtable, &ds, &tree};

    auto pred = interpreter.Evaluate(coeffs, range);
    auto jac  = interpreter.JacRev(coeffs, range); // d(tree)/d(coeffs), unscaled

    // Trivial fit (target == prediction) so nll is identical whether or not
    // the Jacobian is scaled — isolates the effect on the Fisher/parameter
    // cost term, which is what the fix touches.
    constexpr Operon::Scalar sigma = 0.1F;
    auto const sigmaArr = std::array<Operon::Scalar, 1>{sigma};
    auto const nll = static_cast<double>(Operon::GaussianLikelihood<Operon::Scalar>::ComputeLikelihood(
        {pred.data(), pred.size()}, {pred.data(), pred.size()}, {sigmaArr.data(), sigmaArr.size()}));

    auto const fisherUnscaled = Operon::GaussianLikelihood<Operon::Scalar>::ComputeFisherMatrix(
        {pred.data(), pred.size()}, {jac.data(), static_cast<std::size_t>(jac.size())}, {sigmaArr.data(), sigmaArr.size()});
    auto const mdlUnscaled = Operon::MinimumDescriptionLength(tree, coeffs, fisherUnscaled.diagonal().array(), nll);

    constexpr auto a = Operon::Scalar{3.0F};
    auto jacScaled = jac;
    jacScaled *= a; // what WriteParetoFront's "jac *= scale" does for a fitted y = a*tree(x;coeffs)+b
    auto const fisherScaled = Operon::GaussianLikelihood<Operon::Scalar>::ComputeFisherMatrix(
        {pred.data(), pred.size()}, {jacScaled.data(), static_cast<std::size_t>(jacScaled.size())}, {sigmaArr.data(), sigmaArr.size()});
    auto const mdlScaled = Operon::MinimumDescriptionLength(tree, coeffs, fisherScaled.diagonal().array(), nll);

    // fisherScaled = a^2 * fisherUnscaled, so the (single, well above its
    // quantization threshold at this sigma/coefficient magnitude)
    // significant parameter's 0.5*log(fi) term grows by 0.5*log(a^2) = log(a).
    auto const expectedDelta = static_cast<double>(coeffs.size()) * std::log(static_cast<double>(a));
    CHECK(mdlScaled - mdlUnscaled == Catch::Approx(expectedDelta).margin(1e-6));

    // The bug this guards against: silently using the unscaled Jacobian
    // (as if `a` were 1) must NOT reproduce the correctly-scaled MDL.
    CHECK(mdlScaled != Catch::Approx(mdlUnscaled).margin(1e-3));
}

} // namespace Operon::Test
