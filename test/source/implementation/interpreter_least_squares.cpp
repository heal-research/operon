// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/core/dataset.hpp"
#include "operon/optimizer/interpreter_least_squares.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"

namespace {

using Extents = std::dextents<std::size_t, 2>;
using Mapping = std::layout_stride::mapping<Extents>;

// y = X1 + X2 + X3 (linear, unique solution w1=w2=w3=1). Variable weights
// start at 0.1: a well-conditioned linear problem with 3 optimizable
// coefficients, matching test/source/implementation/optimizer.cpp's fixture.
struct InterpreterFixture {
    static constexpr auto Nrow { 40 };
    static constexpr auto Ncol { 4 };

    Operon::RandomGenerator rng { 0 }; // NOLINT(readability-identifier-naming)
    Operon::Dataset ds; // NOLINT(readability-identifier-naming)
    Operon::Tree tree; // NOLINT(readability-identifier-naming)
    using DTable = Operon::DispatchTable<Operon::Scalar>;
    DTable dtable; // NOLINT(readability-identifier-naming)

    InterpreterFixture()
        : ds([&]() -> Operon::Dataset {
            std::vector<std::vector<Operon::Scalar>> cols(Ncol, std::vector<Operon::Scalar>(Nrow));
            for (auto j = 0; j < Ncol - 1; ++j) {
                for (auto i = 0; i < Nrow; ++i) {
                    cols[j][i] = Operon::Random::Uniform(rng, -1.0F, +1.0F);
                }
            }
            for (auto i = 0; i < Nrow; ++i) {
                cols[Ncol - 1][i] = cols[0][i] + cols[1][i] + cols[2][i];
            }
            return Operon::Dataset(cols);
        }())
        , tree([&]() -> Operon::Tree {
            auto t = Operon::InfixParser::ParseOrThrow("X1 + X2 + X3", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) { node.Value = Operon::Scalar { 0.1 }; }
            }
            return t;
        }())
    {
    }
};

} // namespace

TEST_CASE("InterpreterLeastSquaresCostFunction: dimensions match the tree and range", "[interpreter-least-squares]")
{
    InterpreterFixture fix;
    Operon::Interpreter<Operon::Scalar, InterpreterFixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    std::vector<Operon::Scalar> target(fix.ds.Rows(), 0);
    Operon::InterpreterLeastSquaresCostFunction cost { &interpreter, target, Operon::Range { 0, InterpreterFixture::Nrow } };

    CHECK(cost.NumParameters() == static_cast<std::size_t>(fix.tree.CoefficientsCount()));
    CHECK(cost.NumResiduals() == InterpreterFixture::Nrow);
}

TEST_CASE("InterpreterLeastSquaresCostFunction: residuals and Jacobian match a direct interpreter evaluation", "[interpreter-least-squares]")
{
    InterpreterFixture fix;
    Operon::Interpreter<Operon::Scalar, InterpreterFixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, InterpreterFixture::Nrow };
    auto const n = range.Size();
    auto const p = static_cast<std::size_t>(fix.tree.CoefficientsCount());
    std::vector<Operon::Scalar> params { 0.2F, -0.3F, 0.5F };

    std::vector<Operon::Scalar> refResiduals(n);
    REQUIRE(interpreter.Evaluate(params, range, refResiduals).has_value());
    for (std::size_t i = 0; i < n; ++i) {
        refResiduals[i] -= target[range.Start() + i];
    }
    std::vector<Operon::Scalar> refJacobian(n * p);
    REQUIRE(interpreter.JacRev(params, range, refJacobian).has_value());

    Operon::InterpreterLeastSquaresCostFunction cost { &interpreter, target, range };
    std::vector<Operon::Scalar> residuals(n);
    std::vector<Operon::Scalar> jacBuffer(n * p);
    Operon::ScalarMatrixView jac { jacBuffer.data(), Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };
    auto result = cost.Evaluate(params, residuals, jac);
    REQUIRE(result.has_value());

    for (std::size_t i = 0; i < n; ++i) {
        CHECK_THAT(static_cast<double>(residuals[i]), Catch::Matchers::WithinAbs(static_cast<double>(refResiduals[i]), 1e-5));
    }
    // interpreter JacRev output is column-major flat (stride {1, n})
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            CHECK_THAT(static_cast<double>(Operon::At(jac, i, j)), Catch::Matchers::WithinAbs(static_cast<double>(refJacobian[(j * n) + i]), 1e-5));
        }
    }
}

TEST_CASE("InterpreterLeastSquaresCostFunction: row-major, column-major, and padded Jacobian views agree", "[interpreter-least-squares]")
{
    InterpreterFixture fix;
    Operon::Interpreter<Operon::Scalar, InterpreterFixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, InterpreterFixture::Nrow };
    auto const n = range.Size();
    auto const p = static_cast<std::size_t>(fix.tree.CoefficientsCount());
    std::vector<Operon::Scalar> params { 0.1F, 0.2F, 0.3F };

    Operon::InterpreterLeastSquaresCostFunction cost { &interpreter, target, range };

    std::vector<Operon::Scalar> residuals(n);
    std::vector<Operon::Scalar> rowMajorBuffer(n * p);
    Operon::ScalarMatrixView rowMajorJac { rowMajorBuffer.data(), Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };
    REQUIRE(cost.Evaluate(params, residuals, rowMajorJac).has_value());

    std::vector<Operon::Scalar> colMajorBuffer(n * p);
    Operon::ScalarMatrixView colMajorJac { colMajorBuffer.data(), Mapping { Extents { n, p }, std::array<std::size_t, 2> { 1, n } } };
    REQUIRE(cost.Evaluate(params, residuals, colMajorJac).has_value());

    constexpr std::size_t padding = 2;
    std::vector<Operon::Scalar> paddedBuffer(n * (p + padding));
    Operon::ScalarMatrixView paddedJac { paddedBuffer.data(), Mapping { Extents { n, p }, std::array<std::size_t, 2> { p + padding, 1 } } };
    REQUIRE(cost.Evaluate(params, residuals, paddedJac).has_value());

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            auto const reference = static_cast<double>(Operon::At(rowMajorJac, i, j));
            CHECK_THAT(static_cast<double>(Operon::At(colMajorJac, i, j)), Catch::Matchers::WithinAbs(reference, 1e-6));
            CHECK_THAT(static_cast<double>(Operon::At(paddedJac, i, j)), Catch::Matchers::WithinAbs(reference, 1e-6));
        }
    }
}

TEST_CASE("InterpreterLeastSquaresCostFunction: residual-only evaluation skips the Jacobian", "[interpreter-least-squares]")
{
    InterpreterFixture fix;
    Operon::Interpreter<Operon::Scalar, InterpreterFixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, InterpreterFixture::Nrow };
    Operon::InterpreterLeastSquaresCostFunction cost { &interpreter, target, range };

    std::vector<Operon::Scalar> params { 0.1F, 0.2F, 0.3F };
    std::vector<Operon::Scalar> residuals(range.Size());
    auto result = cost.Evaluate(params, residuals, std::nullopt);
    REQUIRE(result.has_value());
    CHECK(std::any_of(residuals.begin(), residuals.end(), [](auto r) -> bool { return r != Operon::Scalar { 0 }; }));
}

TEST_CASE("InterpreterLeastSquaresCostFunction: invalid parameter/residual/Jacobian shapes are rejected", "[interpreter-least-squares]")
{
    InterpreterFixture fix;
    Operon::Interpreter<Operon::Scalar, InterpreterFixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, InterpreterFixture::Nrow };
    Operon::InterpreterLeastSquaresCostFunction cost { &interpreter, target, range };

    SECTION("wrong parameter count") {
        std::vector<Operon::Scalar> params { 0.1F };
        std::vector<Operon::Scalar> residuals(range.Size());
        auto result = cost.Evaluate(params, residuals, std::nullopt);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidShape);
    }

    SECTION("wrong residual count") {
        std::vector<Operon::Scalar> params { 0.1F, 0.2F, 0.3F };
        std::vector<Operon::Scalar> residuals(range.Size() + 1);
        auto result = cost.Evaluate(params, residuals, std::nullopt);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidShape);
    }
}

TEST_CASE("InterpreterLeastSquaresCostFunction: interpreter failures are typed with the original cause", "[interpreter-least-squares]")
{
    InterpreterFixture fix;
    constexpr auto missingVariable = Operon::Hash { 0xBADF00D };
    auto const variableTree = Operon::Tree({ Operon::Node { Operon::NodeType::Variable, missingVariable } });
    Operon::Interpreter<Operon::Scalar, InterpreterFixture::DTable> interpreter { &fix.dtable, &fix.ds, &variableTree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, InterpreterFixture::Nrow };
    Operon::InterpreterLeastSquaresCostFunction cost { &interpreter, target, range };

    std::vector<Operon::Scalar> params(cost.NumParameters());
    std::vector<Operon::Scalar> residuals(range.Size());
    auto result = cost.Evaluate(params, residuals, std::nullopt);

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().Code == Operon::LeastSquaresErrorCode::EvaluationFailure);
    REQUIRE(result.error().Cause.has_value());
    CHECK(result.error().Cause->Kind == Operon::InterpreterError::Code::MissingVariable);
    CHECK(result.error().Cause->Hash == missingVariable);
}
