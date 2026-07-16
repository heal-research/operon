// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "operon/core/node.hpp"
#include "operon/core/standard_library.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/parser/infix.hpp"

// Phase 1 of the generic-node-registry migration: prove that populating a
// DispatchTable via StandardLibrary::Register (the same public
// RegisterFunction<T> API used for custom Dynamic functions) is exactly
// equivalent to DispatchTable's default constructor (compile-time
// index_sequence loop) - same callables, same evaluation results. This is
// the load-bearing "no behavior change" proof the rest of the migration
// plan depends on.

namespace Operon::Test {

TEST_CASE("StandardLibrary::Register matches the default DispatchTable construction", "[interpreter]") // NOLINT(readability-function-cognitive-complexity)
{
    using DT = Operon::DispatchTable<Operon::Scalar>;
    using Scalar = Operon::Scalar;
    constexpr auto S = DT::BatchSize<Scalar>;
    using FnPtr  = void (*)(Operon::Vector<Node> const&, Backend::View<Scalar, S>, size_t, Operon::Range);
    using DfPtr  = void (*)(Operon::Vector<Node> const&, Backend::View<Scalar const, S>, Backend::View<Scalar, S>, int, int);

    DT const dtDefault; // populated by the existing compile-time enum loop

    DT dtRuntime;
    dtRuntime.GetMap().clear();
    Operon::StandardLibrary::Register(dtRuntime);

    SECTION("Both tables contain the same set of built-in hashes") {
        for (auto i = 0UL; i < Operon::NodeTypes::Count - 4; ++i) {
            auto const h = Operon::Node(static_cast<Operon::NodeType>(i)).HashValue;
            CHECK(dtDefault.Contains(h));
            CHECK(dtRuntime.Contains(h));
        }
    }

    SECTION("Registered callables are the identical compiled kernels (same function pointer target)") {
        for (auto i = 0UL; i < Operon::NodeTypes::Count - 4; ++i) {
            auto const h = Operon::Node(static_cast<Operon::NodeType>(i)).HashValue;

            auto const& fDefault = dtDefault.GetFunction<Scalar>(h);
            auto const& fRuntime = dtRuntime.GetFunction<Scalar>(h);
            auto const* pDefault = fDefault.target<FnPtr>();
            auto const* pRuntime = fRuntime.target<FnPtr>();
            REQUIRE(pDefault != nullptr);
            REQUIRE(pRuntime != nullptr);
            CHECK(*pDefault == *pRuntime);

            auto const& dDefault = dtDefault.GetDerivative<Scalar>(h);
            auto const& dRuntime = dtRuntime.GetDerivative<Scalar>(h);
            auto const* qDefault = dDefault.target<DfPtr>();
            auto const* qRuntime = dRuntime.target<DfPtr>();
            REQUIRE(qDefault != nullptr);
            REQUIRE(qRuntime != nullptr);
            CHECK(*qDefault == *qRuntime);
        }
    }

    SECTION("Evaluation results are identical across a representative expression set") {
        std::string const x{"x"};
        std::vector<Scalar> const v{0.3, 1.2, 2.7, -0.5, 4.1}; // NOLINT
        Operon::Dataset const ds({x}, {v});

        auto const exprs = {
            "1 + 2 + 3 + 4",           // n-ary: Add
            "2 * 3 * 4",               // n-ary: Mul
            "10 - 3 - 2",              // n-ary: Sub
            "100 / 5 / 2",             // n-ary: Div
            "aq(6, 3)",                // binary: Aq
            "pow(2, 3)",               // binary: Pow
            "sin(x) + cos(x)",         // unary: Sin, Cos, plus Add
            "sqrt(abs(x)) * exp(log(4))", // unary chain
            "tanh(x) / (1 + x * x)",   // mixed
        };

        for (auto const& expr : exprs) {
            auto t = InfixParser::Parse(expr);
            auto rDefault = Interpreter<Scalar, DT>(&dtDefault, &ds, &t).Evaluate(t.GetCoefficients(), Operon::Range(0, v.size()));
            auto rRuntime = Interpreter<Scalar, DT>(&dtRuntime, &ds, &t).Evaluate(t.GetCoefficients(), Operon::Range(0, v.size()));

            REQUIRE(rDefault.size() == rRuntime.size());
            for (size_t i = 0; i < rDefault.size(); ++i) {
                CHECK(rDefault[i] == rRuntime[i]); // bit-identical: same compiled kernels
            }
        }
    }
}

} // namespace Operon::Test
