// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Package-consumer contract fixture for operon::core (see
// test/package-consumer/core/CMakeLists.txt). Intentionally
// standalone: it must compile and link using ONLY the core public
// headers and the operon::core imported target as they appear
// after `find_package(operon CONFIG REQUIRED)` against an *installed,
// relocated* package -- it never sees the operon source or build tree
// directly, and it never links operon::backend_adapter or operon::operon.
//
// Includes every header of operon::core's installed contract (the
// CMakeLists.txt in this directory scans this file's include closure for
// backend headers): view (memory_view.hpp, view_descriptor.hpp), numerical
// (least_squares.hpp, fisher_information.hpp, gradient_cost.hpp), and the
// compiled tree/grammar/primitive-set/enumeration-canonicalizer
// implementation (node.hpp, tree.hpp, grammar.hpp, pset.hpp,
// enumeration_canonicalizer.hpp). None of them, nor anything they
// transitively include, may pull in Eigen or any other backend dependency.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <operon/algorithms/enumeration_canonicalizer.hpp>
#include <operon/collections/bitset.hpp>
#include <operon/core/aligned_allocator.hpp>
#include <operon/core/concepts.hpp>
#include <operon/core/constants.hpp>
#include <operon/core/contracts.hpp>
#include <operon/core/grammar.hpp>
#include <operon/core/interpreter_error.hpp>
#include <operon/core/memory_view.hpp>
#include <operon/core/node.hpp>
#include <operon/core/pset.hpp>
#include <operon/core/range.hpp>
#include <operon/core/standard_library.hpp>
#include <operon/core/subtree.hpp>
#include <operon/core/tree.hpp>
#include <operon/core/types.hpp>
#include <operon/core/view_descriptor.h>
#include <operon/core/view_descriptor.hpp>
#include <operon/hash/hash.hpp>
#include <operon/hash/metrohash64.hpp>
#include <operon/mdspan/mdspan.hpp>
#include <operon/optimizer/fisher_information.hpp>
#include <operon/optimizer/gradient_cost.hpp>
#include <operon/optimizer/least_squares.hpp>
#include <operon/optimizer/least_squares_gradient_adapter.hpp>

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
        std::cerr << "package-consumer(core): view_descriptor rejected a valid descriptor\n";
        return EXIT_FAILURE;
    }
    if (Operon::At(*descriptorView, 2, 1) != Operon::Scalar {1}) {
        std::cerr << "package-consumer(core): view_descriptor built an incorrect view\n";
        return EXIT_FAILURE;
    }

    // F = J^T J (unweighted, sigma empty): [[2,1],[1,2]].
    std::array<Operon::Scalar, 4> fisherStorage {};
    Operon::ScalarMatrixView fisher {
        fisherStorage.data(), Mapping {Extents {2, 2}, std::array<std::size_t, 2> {2, 1}}};
    auto fisherResult = Operon::ComputeFisherMatrix(jacobian, {}, fisher);
    if (!fisherResult) {
        std::cerr << "package-consumer(core): ComputeFisherMatrix failed\n";
        return EXIT_FAILURE;
    }
    if (!NearlyEqual(Operon::At(fisher, 0, 0), 2.0) || !NearlyEqual(Operon::At(fisher, 1, 1), 2.0)
        || !NearlyEqual(Operon::At(fisher, 0, 1), 1.0) || !NearlyEqual(Operon::At(fisher, 1, 0), 1.0)) {
        std::cerr << "package-consumer(core): unexpected Fisher matrix\n";
        return EXIT_FAILURE;
    }

    // Cost = 0.5 * sum(r_i^2) = 7; gradient = J^T r = [4, 5].
    std::array<Operon::Scalar, 3> residuals {1, 2, 3};
    std::array<Operon::Scalar, 2> gradient {};
    auto gradientResult = Operon::ComputeGradient(residuals, jacobian, gradient);
    if (!gradientResult) {
        std::cerr << "package-consumer(core): ComputeGradient failed\n";
        return EXIT_FAILURE;
    }
    if (!NearlyEqual(*gradientResult, 7.0) || !NearlyEqual(gradient[0], 4.0) || !NearlyEqual(gradient[1], 5.0)) {
        std::cerr << "package-consumer(core): unexpected gradient/cost\n";
        return EXIT_FAILURE;
    }

    // ResidualNorm = sqrt(14); GradientNorm = sqrt(16 + 25) = sqrt(41).
    auto diagnostics = Operon::ComputeDiagnostics(residuals, jacobian);
    if (!diagnostics) {
        std::cerr << "package-consumer(core): ComputeDiagnostics failed\n";
        return EXIT_FAILURE;
    }
    if (!NearlyEqual(diagnostics->Cost, 7.0) || !NearlyEqual(diagnostics->ResidualNorm, std::sqrt(14.0))
        || !NearlyEqual(diagnostics->GradientNorm, std::sqrt(41.0))) {
        std::cerr << "package-consumer(core): unexpected diagnostics\n";
        return EXIT_FAILURE;
    }

    // Typed weight validation and the public location-preserving error
    // conversion are Eigen-free core symbols (header-only here).
    std::array<Operon::Scalar, 3> badWeights {1, -1, 1};
    auto weightResult = Operon::ValidateWeights(badWeights, 3);
    if (weightResult || weightResult.error().Code != Operon::WeightErrorCode::NegativeValue || weightResult.error().Row != 1) {
        std::cerr << "package-consumer(core): ValidateWeights did not report the negative weight\n";
        return EXIT_FAILURE;
    }
    auto const converted = Operon::ToGradientError(weightResult.error());
    if (converted.Code != Operon::GradientErrorCode::InvalidWeights || converted.Row != 1) {
        std::cerr << "package-consumer(core): ToGradientError lost the weight error location\n";
        return EXIT_FAILURE;
    }

    // Tree validation: a well-formed postfix tree (1 + 2) must Validate().
    // Node::Function/Node::Constant, Tree::UpdateNodes(), and Tree::Validate()
    // are all compiled into operon_core (source/core/node.cpp,
    // source/core/tree.cpp) rather than declared-only, so this is a real link
    // check, not just a header-compiles-standalone check.
    auto const addHash = static_cast<Operon::Hash>(Operon::BuiltinOp::Add);
    Operon::Tree const sumTree = Operon::Tree({
        Operon::Node::Constant(1.0),
        Operon::Node::Constant(2.0),
        Operon::Node::Function(addHash, /*arity=*/2),
    }).UpdateNodes();
    if (!sumTree.Validate()) {
        std::cerr << "package-consumer(core): well-formed tree failed Validate()\n";
        return EXIT_FAILURE;
    }

    // A tree with a dangling forward Ref must fail Validate() (RefTo must
    // point strictly backward) -- exercises the compiled error path too.
    Operon::Node dangling(Operon::NodeType::Ref);
    dangling.RefTo = 1; // forward reference: invalid
    Operon::Tree const invalidTree = Operon::Tree({ dangling, Operon::Node::Constant(2.0) }).UpdateNodes();
    if (invalidTree.Validate()) {
        std::cerr << "package-consumer(core): malformed tree unexpectedly passed Validate()\n";
        return EXIT_FAILURE;
    }

    // Grammar: PrimitiveSet::Full includes unary ops (Exp/Log/Sin/Sqrt/Cbrt/
    // Aq) that Grammar::Configure(PrimitiveSetConfig) translates into
    // RecurringFactor productions (see grammar.hpp's class comment) -- unlike
    // PrimitiveSet::Arithmetic, which is Add/Sub/Mul/Div only and translates
    // to zero RecurringFactor productions by design. Must produce a
    // non-empty RecurringFactor production set and a finite MinComplexity
    // for it -- both computed by Grammar::Rebuild(), compiled into
    // operon_core (source/core/grammar.cpp).
    // A non-empty variableHashes list is required for MinComplexity to reach
    // any nonterminal at all: RecurringFactor/SimpleTerm's MinComplexity is
    // only seeded (to 1, for a bare Variable leaf) when at least one
    // variable is registered; with none, every nonterminal -- including
    // Expression -- stays permanently Unreachable.
    Operon::Grammar const grammar(Operon::PrimitiveSet::Full, /*variableHashes=*/{ 1 });
    if (grammar.Productions(Operon::GrammarSymbol::RecurringFactor).empty()) {
        std::cerr << "package-consumer(core): Grammar produced no RecurringFactor productions\n";
        return EXIT_FAILURE;
    }
    constexpr auto Unreachable = std::numeric_limits<std::size_t>::max();
    if (grammar.MinComplexity(Operon::GrammarSymbol::Expression) == Unreachable) {
        std::cerr << "package-consumer(core): Grammar reports Expression as unreachable\n";
        return EXIT_FAILURE;
    }

    // Enumeration canonicalization: x+y and y+x must canonicalize to the
    // same Key (commutative reordering), exercising
    // CanonicalizeEnumerationTree, compiled into operon_core
    // (source/algorithms/enumeration_canonicalizer.cpp).
    Operon::Node varX(Operon::NodeType::Variable); varX.HashValue = 1;
    Operon::Node varY(Operon::NodeType::Variable); varY.HashValue = 2;
    Operon::Tree const xy = Operon::Tree({ varX, varY, Operon::Node::Function(addHash, 2) }).UpdateNodes();
    Operon::Tree const yx = Operon::Tree({ varY, varX, Operon::Node::Function(addHash, 2) }).UpdateNodes();
    if (Operon::CanonicalizeEnumerationTree(xy).Key != Operon::CanonicalizeEnumerationTree(yx).Key) {
        std::cerr << "package-consumer(core): x+y and y+x canonicalized to different keys\n";
        return EXIT_FAILURE;
    }

    std::cout << "package-consumer(core): OK\n";
    return EXIT_SUCCESS;
}
