// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Package-consumer contract fixture (see test/package-consumer/CMakeLists.txt).
// Intentionally standalone: it must compile and link using ONLY the public
// headers and operon::operon target as they appear after `find_package(operon
// CONFIG REQUIRED)` against an *installed, relocated* package -- it never
// sees the operon source or build tree directly.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>

#include <operon/core/node.hpp>
#include <operon/core/tree.hpp>
#include <operon/core/version.hpp>
#include <operon/optimizer/least_squares_fit.hpp>

namespace {
// r_i = p * x_i - 3 * x_i: one parameter, minimum at p = 3.
class ScaleCost final : public Operon::LeastSquaresCostFunction {
public:
    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 1; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return X.size(); }
    [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const> parameters, std::span<Operon::Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> jacobian) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        for (std::size_t i = 0; i < X.size(); ++i) {
            residuals[i] = (parameters[0] - Operon::Scalar { 3 }) * X[i];
            if (jacobian) {
                Operon::At(*jacobian, i, 0) = X[i];
            }
        }
        return {};
    }
    std::array<Operon::Scalar, 4> X { 1, 2, 3, 4 };
};
} // namespace

auto main() -> int
{
    // A one-node tree is enough to prove that the installed public headers
    // are self-sufficient (no missing transitive includes) and that
    // liboperon's implementation is actually reachable through the
    // operon::operon imported target (UpdateNodes() is defined in
    // source/core/tree.cpp, not header-only).
    auto node = Operon::Node::Constant(2.0);
    Operon::Tree tree({ node });
    tree.UpdateNodes();

    if (tree.Length() != 1) {
        std::cerr << "package-consumer: unexpected tree length " << tree.Length() << "\n";
        return EXIT_FAILURE;
    }

    // The public least-squares entry point links and runs through the installed
    // package for both backends without any detail:: type.
    for (auto backend : { Operon::OptimizerType::Tiny, Operon::OptimizerType::Eigen }) {
        ScaleCost cost;
        std::array<Operon::Scalar, 1> const start { 0 };
        auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend });
        if (!outcome || std::abs(outcome->FinalParameters.front() - Operon::Scalar { 3 }) > Operon::Scalar { 1e-3 }) {
            std::cerr << "package-consumer: FitLeastSquares did not recover the minimum\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "package-consumer: " << Operon::Version();
    std::cout << "package-consumer: OK\n";
    return EXIT_SUCCESS;
}
