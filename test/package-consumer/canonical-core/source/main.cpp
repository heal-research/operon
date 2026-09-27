// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Package-consumer contract fixture for operon::canonical_core (see
// test/package-consumer/canonical-core/CMakeLists.txt). Intentionally
// standalone: it must compile and link using ONLY the canonical-core public
// headers and the operon::canonical_core imported target as they appear
// after `find_package(operon CONFIG REQUIRED)` against an *installed,
// relocated* package -- it never sees the operon source or build tree
// directly, and it never links operon::backend_adapter or operon::operon.
//
// Representative headers from both halves of canonical-core's contract:
// view (memory_view.hpp, view_descriptor.hpp) and numerical
// (least_squares.hpp, fisher_information.hpp). None of them, nor anything
// they transitively include, may pull in Eigen -- the CMakeLists.txt in
// this directory additionally asserts that structurally.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <operon/core/memory_view.hpp>
#include <operon/core/view_descriptor.h>
#include <operon/core/view_descriptor.hpp>
#include <operon/optimizer/fisher_information.hpp>
#include <operon/optimizer/least_squares.hpp>

namespace {

auto NearlyEqual(double actual, double expected, double tolerance = 1e-3) -> bool
{
    return std::abs(actual - expected) <= tolerance;
}

} // namespace

auto main() -> int {
    // 3 observations x 2 parameters, row-major: J = [[1,0],[0,1],[1,1]].
    std::array<Operon::Scalar, 6> jacobianStorage {1, 0, 0, 1, 1, 1};
    using Extents = std::dextents<std::size_t, 2>;
    using Mapping = std::layout_stride::mapping<Extents>;
    Operon::ConstScalarMatrixView jacobian {
        jacobianStorage.data(), Mapping {Extents {3, 2}, std::array<std::size_t, 2> {2, 1}}};

    // Same view, built through the C-compatible descriptor
    // (view_descriptor.h + view_descriptor.hpp) instead of a bare mdspan
    // mapping, to exercise the FFI-facing half of the view contract too.
    OperonViewDescriptor descriptor {};
    descriptor.version = OPERON_VIEW_DESCRIPTOR_VERSION;
    descriptor.struct_size = sizeof(OperonViewDescriptor);
    descriptor.rank = 2;
    descriptor.scalar_code = sizeof(Operon::Scalar) == sizeof(float) ? OPERON_SCALAR_F32 : OPERON_SCALAR_F64;
    descriptor.element_size = sizeof(Operon::Scalar);
    descriptor.flags = OPERON_VIEW_READONLY;
    descriptor.data = jacobianStorage.data();
    descriptor.extents[0] = 3;
    descriptor.extents[1] = 2;
    descriptor.byte_strides[0] = 2 * static_cast<std::ptrdiff_t>(sizeof(Operon::Scalar));
    descriptor.byte_strides[1] = static_cast<std::ptrdiff_t>(sizeof(Operon::Scalar));

    auto descriptorView = Operon::MakeConstMatrixView(descriptor);
    if (!descriptorView) {
        std::cerr << "package-consumer(canonical-core): view_descriptor rejected a valid descriptor\n";
        return EXIT_FAILURE;
    }
    if (Operon::At(*descriptorView, 2, 1) != Operon::Scalar {1}) {
        std::cerr << "package-consumer(canonical-core): view_descriptor built an incorrect view\n";
        return EXIT_FAILURE;
    }

    // F = J^T J (unweighted, sigma empty): [[2,1],[1,2]].
    std::array<Operon::Scalar, 4> fisherStorage {};
    Operon::ScalarMatrixView fisher {
        fisherStorage.data(), Mapping {Extents {2, 2}, std::array<std::size_t, 2> {2, 1}}};
    auto fisherResult = Operon::ComputeFisherMatrix(jacobian, {}, fisher);
    if (!fisherResult) {
        std::cerr << "package-consumer(canonical-core): ComputeFisherMatrix failed\n";
        return EXIT_FAILURE;
    }
    if (!NearlyEqual(Operon::At(fisher, 0, 0), 2.0) || !NearlyEqual(Operon::At(fisher, 1, 1), 2.0)
        || !NearlyEqual(Operon::At(fisher, 0, 1), 1.0) || !NearlyEqual(Operon::At(fisher, 1, 0), 1.0)) {
        std::cerr << "package-consumer(canonical-core): unexpected Fisher matrix\n";
        return EXIT_FAILURE;
    }

    // Cost = 0.5 * sum(r_i^2) = 7; gradient = J^T r = [4, 5].
    std::array<Operon::Scalar, 3> residuals {1, 2, 3};
    std::array<Operon::Scalar, 2> gradient {};
    auto gradientResult = Operon::ComputeGradient(residuals, jacobian, gradient);
    if (!gradientResult) {
        std::cerr << "package-consumer(canonical-core): ComputeGradient failed\n";
        return EXIT_FAILURE;
    }
    if (!NearlyEqual(*gradientResult, 7.0) || !NearlyEqual(gradient[0], 4.0) || !NearlyEqual(gradient[1], 5.0)) {
        std::cerr << "package-consumer(canonical-core): unexpected gradient/cost\n";
        return EXIT_FAILURE;
    }

    // ResidualNorm = sqrt(14); GradientNorm = sqrt(16 + 25) = sqrt(41).
    auto diagnostics = Operon::ComputeDiagnostics(residuals, jacobian);
    if (!diagnostics) {
        std::cerr << "package-consumer(canonical-core): ComputeDiagnostics failed\n";
        return EXIT_FAILURE;
    }
    if (!NearlyEqual(diagnostics->Cost, 7.0) || !NearlyEqual(diagnostics->ResidualNorm, std::sqrt(14.0))
        || !NearlyEqual(diagnostics->GradientNorm, std::sqrt(41.0))) {
        std::cerr << "package-consumer(canonical-core): unexpected diagnostics\n";
        return EXIT_FAILURE;
    }

    std::cout << "package-consumer(canonical-core): OK\n";
    return EXIT_SUCCESS;
}
