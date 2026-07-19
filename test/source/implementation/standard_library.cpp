// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/standard_library.hpp"
#include "operon/core/symbol_library.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/parser/infix.hpp"

// StandardLibrary::Register must be exactly equivalent to DispatchTable's
// default constructor: same callables, same evaluation results.

namespace Operon::Test {

TEST_CASE("StandardLibrary populates dispatch tables and node names consistently", "[interpreter]") // NOLINT(readability-function-cognitive-complexity)
{
    using DT = Operon::DispatchTable<Operon::Scalar>;
    using Scalar = Operon::Scalar;
    constexpr auto S = DT::BatchSize<Scalar>;
    using FnPtr  = void (*)(Operon::Vector<Node> const&, Backend::View<Scalar, S>, size_t, Operon::Range);
    using DfPtr  = void (*)(Operon::Vector<Node> const&, Backend::View<Scalar const, S>, Backend::View<Scalar, S>, int, int);

    DT const dtDefault;

    DT dtRuntime;
    dtRuntime.GetMap().clear();
    Operon::StandardLibrary::Register(dtRuntime);

    SECTION("Both tables contain the same set of built-in hashes") {
        for (auto i = 0UL; i < Operon::BuiltinOpCount; ++i) {
            auto const h = static_cast<Operon::Hash>(static_cast<Operon::BuiltinOp>(i));
            CHECK(dtDefault.Contains(h));
            CHECK(dtRuntime.Contains(h));
        }
    }

    SECTION("Built-in names and descriptions are available through the unified hash registry") {
        CHECK(Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Add), 2).Name() == "+");
        CHECK(Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Add), 2).Desc() == "n-ary addition f(a,b,c,...) = a + b + c + ...");
        CHECK(Operon::Node(Operon::NodeType::Function).Name() == "dyn"); // generic fallback, no specific hash registered
        CHECK(Operon::Node(Operon::NodeType::Variable).Name() == "variable");
    }

    SECTION("Registered callables are the identical compiled kernels (same function pointer target)") {
        for (auto i = 0UL; i < Operon::BuiltinOpCount; ++i) {
            auto const h = static_cast<Operon::Hash>(static_cast<Operon::BuiltinOp>(i));

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

TEST_CASE("Unregistered Function nodes fall back to the generic name/desc", "[interpreter]")
{
    Operon::StandardLibrary::RegisterNames();

    Operon::Hash const unregisteredHash = Operon::Hasher{}("test::unregistered_dynamic_fallback");
    Operon::Node const dynNode(Operon::NodeType::Function, unregisteredHash);

    SECTION("Name() returns the generic \"dyn\" fallback") {
        CHECK(dynNode.Name() == "dyn");
    }

    SECTION("Desc() returns the generic \"user-defined function\" fallback") {
        CHECK(dynNode.Desc() == "user-defined function");
    }

    SECTION("A registered Function node still resolves to its specific name") {
        Operon::Hash const registeredHash = Operon::Hasher{}("test::registered_dynamic_fallback");
        Operon::Node::RegisterName(registeredHash, "myop", "my custom op");
        Operon::Node const registeredDyn(Operon::NodeType::Function, registeredHash);
        CHECK(registeredDyn.Name() == "myop");
        CHECK(registeredDyn.Desc() == "my custom op");
    }
}

TEST_CASE("StandardLibrary::ArityLimits agrees with IsNaryOp/IsBinaryOp/IsUnaryOp classification", "[interpreter]")
{
    // Arity is registry-sourced now, not inferred from any enum position
    // (that inference was removed along with NodeType's per-op enumerators).
    // What should still hold: n-ary/binary ops report MinArity==MaxArity==2,
    // unary ops report MinArity==MaxArity==1.
    for (auto i = 0UL; i < Operon::BuiltinOpCount; ++i) {
        auto const op = static_cast<Operon::BuiltinOp>(i);
        auto const [minArity, maxArity] = Operon::StandardLibrary::ArityLimits(op);
        CHECK(minArity == maxArity);
    }
    // Spot-check a representative op from each category directly.
    {
        auto const [lo, hi] = Operon::StandardLibrary::ArityLimits(Operon::BuiltinOp::Add);
        CHECK(lo == 2); CHECK(hi == 2);
    }
    {
        auto const [lo, hi] = Operon::StandardLibrary::ArityLimits(Operon::BuiltinOp::Pow);
        CHECK(lo == 2); CHECK(hi == 2);
    }
    {
        auto const [lo, hi] = Operon::StandardLibrary::ArityLimits(Operon::BuiltinOp::Sin);
        CHECK(lo == 1); CHECK(hi == 1);
    }
}

TEST_CASE("PrimitiveSet preset configs source arity from the registry, not Node(t)", "[interpreter]")
{
    Operon::PrimitiveSet pset;
    pset.SetConfig(Operon::PrimitiveSet::Full);

    auto const addHash = static_cast<Operon::Hash>(Operon::BuiltinOp::Add);
    auto const sinHash = static_cast<Operon::Hash>(Operon::BuiltinOp::Sin);
    auto const constHash = Operon::Node(Operon::NodeType::Constant).HashValue;
    auto const varHash = Operon::Node(Operon::NodeType::Variable).HashValue;

    CHECK(pset.MinMaxArity(addHash) == std::tuple<size_t, size_t>{2, 2});
    CHECK(pset.MinMaxArity(sinHash) == std::tuple<size_t, size_t>{1, 1});
    CHECK(pset.MinMaxArity(constHash) == std::tuple<size_t, size_t>{0, 0});
    CHECK(pset.MinMaxArity(varHash) == std::tuple<size_t, size_t>{0, 0});
}

TEST_CASE("RegisterNaryFunction registers a variable-arity function end-to-end", "[interpreter]")
{
    using DT = Operon::DispatchTable<Operon::Scalar>;
    using Scalar = Operon::Scalar;

    DT dt;
    Operon::PrimitiveSet pset;
    pset.SetConfig(Operon::PrimitiveSet::Arithmetic);

    Operon::FunctionInfo const info{
        .Name      = "sum3",
        .Desc      = "n-ary sum",
        .Arity     = 3,
        .Frequency = 1
    };
    auto const hash = Operon::Hasher{}(info.Name);
    auto primal = [](auto acc, auto x) -> auto { return acc + x; };

    Operon::RegisterNaryFunction<DT, Scalar>(dt, pset, info, /*maxArity=*/5, primal);

    auto makeTree = [&] {
        auto dyn = Operon::Node::Function(hash, 3); // sets Arity=Length=3, Optimize=false
        return Operon::Tree({
            Operon::Node::Constant(1.0),
            Operon::Node::Constant(2.0),
            Operon::Node::Constant(3.0),
            dyn,
        });
    };

    SECTION("PrimitiveSet arity range widens to [Arity, maxArity]") {
        CHECK(pset.MinimumArity(hash) == 3);
        CHECK(pset.MaximumArity(hash) == 5);
    }

    SECTION("Evaluation folds children left-to-right") {
        auto const tree = makeTree();

        std::string const x{"x"};
        Operon::Dataset const ds({x}, {std::vector<Scalar>{0.0}});
        auto coeff = tree.GetCoefficients();
        auto r = Operon::Interpreter<Scalar, DT>(&dt, &ds, &tree).Evaluate(coeff, Operon::Range(0, 1));

        REQUIRE(std::ssize(r) == 1);
        CHECK(r[0] == Catch::Approx(6.0).epsilon(1e-6)); // (1 + 2) + 3
    }

    SECTION("Auto-diff assigns unit partials to every child (fn is linear)") {
        auto const tree = makeTree();

        std::string const x{"x"};
        Operon::Dataset const ds({x}, {std::vector<Scalar>{0.0}});
        auto coeff = tree.GetCoefficients();
        auto jac = Operon::Interpreter<Scalar, DT>(&dt, &ds, &tree).JacFwd(coeff, Operon::Range(0, 1));

        REQUIRE(jac.rows() == 1);
        REQUIRE(jac.cols() == 3);
        for (auto c = 0; c < jac.cols(); ++c) {
            CHECK(jac(0, c) == Catch::Approx(1.0).epsilon(1e-5));
        }
    }

    SECTION("Auto-diff seeds exactly one child per column (fn is non-linear)") {
        // Add's fold gives every child a partial of 1 regardless of seeding
        // correctness, so it can't tell a correct single-child seed apart
        // from the historical bug where every non-target child was also
        // seeded to unit tangent. A product fold can: partials differ by
        // position, so a wrong seed produces a value that doesn't match the
        // closed-form product-rule partial for that position.
        DT dtProd;
        Operon::PrimitiveSet psetProd;
        Operon::FunctionInfo const prodInfo{
            .Name      = "prod3",
            .Desc      = "n-ary product",
            .Arity     = 3,
            .Frequency = 1
        };
        auto const prodHash = Operon::Hasher{}(prodInfo.Name);
        auto prodPrimal = [](auto acc, auto x) -> auto { return acc * x; };
        Operon::RegisterNaryFunction<DT, Scalar>(dtProd, psetProd, prodInfo, /*maxArity=*/3, prodPrimal);

        auto dyn = Operon::Node::Function(prodHash, 3);
        Operon::Tree const tree({
            Operon::Node::Constant(2.0),
            Operon::Node::Constant(3.0),
            Operon::Node::Constant(4.0),
            dyn,
        });

        std::string const x{"x"};
        Operon::Dataset const ds({x}, {std::vector<Scalar>{0.0}});
        auto coeff = tree.GetCoefficients();
        auto jac = Operon::Interpreter<Scalar, DT>(&dtProd, &ds, &tree).JacFwd(coeff, Operon::Range(0, 1));

        REQUIRE(jac.rows() == 1);
        REQUIRE(jac.cols() == 3);
        CHECK(jac(0, 0) == Catch::Approx(12.0).epsilon(1e-5)); // d(c0*c1*c2)/dc0 = c1*c2 = 3*4
        CHECK(jac(0, 1) == Catch::Approx(8.0).epsilon(1e-5));  // d/dc1 = c0*c2 = 2*4
        CHECK(jac(0, 2) == Catch::Approx(6.0).epsilon(1e-5));  // d/dc2 = c0*c1 = 2*3
    }
}

} // namespace Operon::Test
