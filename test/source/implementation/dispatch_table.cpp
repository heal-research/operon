// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <numbers>
#include <numeric>

#include "operon/core/pset.hpp"
#include "operon/core/symbol_library.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/parser/infix.hpp"
#include "operon/formatter/formatter.hpp"

namespace Operon::Test {

TEST_CASE("DispatchTable constructors", "[interpreter]")
{
    using DT = Operon::DispatchTable<Operon::Scalar>;

    std::string const x{"x"};
    std::vector<Operon::Scalar> const v{0};
    Operon::Dataset const ds({x}, {v});

    auto check = [&](DT const& dt, std::string const& expr, Operon::Scalar expected) -> void {
        auto t = InfixParser::Parse(expr);
        auto p = t.GetCoefficients();
        auto r = Operon::Interpreter<Operon::Scalar, DT>(&dt, &ds, &t).Evaluate(p, Operon::Range(0, 1));
        CHECK(r[0] == Catch::Approx(expected));
    };

    SECTION("Default constructor") {
        DT const dt;
        check(dt, "1 + 2 + 3", 6);
        check(dt, "1 - 2 - 3", -4);
        check(dt, "6 / 3 / 2", 1);
        check(dt, "6 / 3 * 2", 4);
    }

    SECTION("Copy constructor") {
        DT const dt;
        const DT& dt1(dt);
        check(dt1, "2 * 3 / 4", 1.5);
    }

    SECTION("Move constructor") {
        DT const dt;
        DT dt1(dt);
        DT const dt2(std::move(dt1));
        check(dt2, "sin(1 / 2 * 3.141519)", std::sin(1.0 / 2.0 * std::numbers::pi));
    }

    SECTION("Construct from map") {
        DT const dt;
        auto const& map = dt.GetMap();
        DT const dt3(map);
        check(dt3, "cos(3.141519)", std::cos(std::numbers::pi_v<float>));

        DT const dt4(map);
        check(dt4, "exp(log(10))", std::exp(std::numbers::ln10_v<float>));
    }
}

TEST_CASE("DispatchTable evaluation of expressions", "[interpreter]")
{
    using DT = Operon::DispatchTable<Operon::Scalar>;
    DT const dtable;

    std::string const x{"x"};
    std::vector<Operon::Scalar> const v{0};
    Operon::Dataset const ds({x}, {v});

    SECTION("Arithmetic") {
        auto t = InfixParser::Parse("2 + 3 * 4");
        auto r = Interpreter<Operon::Scalar, DT>::Evaluate(t, ds, Range(0, 1));
        CHECK(r[0] == Catch::Approx(14.0f));
    }

    SECTION("Transcendental functions") {
        auto t = InfixParser::Parse("exp(1)");
        auto r = Interpreter<Operon::Scalar, DT>::Evaluate(t, ds, Range(0, 1));
        CHECK(r[0] == Catch::Approx(std::exp(1.0f)));

        t = InfixParser::Parse("log(exp(1))");
        r = Interpreter<Operon::Scalar, DT>::Evaluate(t, ds, Range(0, 1));
        CHECK(r[0] == Catch::Approx(1.0f).epsilon(1e-3));
    }
}

TEST_CASE("RegisterFunction - user-defined symbol", "[interpreter]")
{
    using DT    = Operon::DispatchTable<Operon::Scalar>;
    using Scalar = Operon::Scalar;
    constexpr auto S = DT::BatchSize<Scalar>;

    std::string const x{"x"};
    std::vector<Scalar> xvals(10);
    std::iota(xvals.begin(), xvals.end(), 1.0F); // x = {1, 2, ..., 10}
    Operon::Dataset ds({x}, {xvals});

    DT dt;
    auto const myHash = Operon::Hasher{}("test::negate");

    // Register a batch callable that negates its argument.
    // Follows the same convention as built-in Func<> specialisations:
    // reads nodes[i].Value as the node weight and applies it.
    DT::Callable<Scalar> const primal = [](
        Operon::Vector<Operon::Node> const& nodes,
        Operon::Backend::View<Scalar, S> data,
        size_t i,
        Operon::Range /*rg*/)
    -> void {
        auto const  w   = static_cast<Scalar>(nodes[i].Value);
        auto*       dst = data.data_handle() + (i * S);
        auto const* src = data.data_handle() + ((i - 1) * S);
        for (auto k = 0UL; k < S; ++k) { dst[k] = w * -src[k]; }
    };

    dt.RegisterFunction<Scalar>(myHash, primal);

    SECTION("Callable is present in the map after registration") {
        CHECK(dt.Contains(myHash));
        CHECK(dt.TryGetFunction<Scalar>(myHash).has_value());
    }

    SECTION("Evaluate a tree: negate(x)") {
        // Post-order: [Variable(x), Dynamic(negate)]
        Operon::Node varNode(Operon::NodeType::Variable);
        varNode.HashValue = ds.GetVariable(x).value().Hash;

        Operon::Node dynNode(Operon::NodeType::Dynamic, myHash);
        dynNode.Arity  = 1;
        dynNode.Length = 1;

        Operon::Tree const tree({ varNode, dynNode });
        auto coeff = tree.GetCoefficients();

        auto r = Operon::Interpreter<Scalar, DT>(&dt, &ds, &tree).Evaluate(coeff, Operon::Range(0, 10));

        REQUIRE(std::ssize(r) == 10);
        for (auto i = 0; i < 10; ++i) {
            CHECK(r[i] == Catch::Approx(-xvals[i]));
        }
    }
}

TEST_CASE("RegisterUnary - scalar lambda adapter", "[interpreter]")
{
    using DT     = Operon::DispatchTable<Operon::Scalar>;
    using Scalar = Operon::Scalar;

    std::string const x{"x"};
    std::vector<Scalar> xvals(10);
    std::iota(xvals.begin(), xvals.end(), 1.0F); // x = {1, 2, ..., 10}
    Operon::Dataset ds({x}, {xvals});

    DT dt;
    auto const h = Operon::Hasher{}("test::sincos");

    // f(x) = sin(x) + cos(x),  f'(x) = cos(x) - sin(x)
    Operon::RegisterUnary<DT, Scalar>(dt, h,
        [](auto x) -> auto { return std::sin(x) + std::cos(x); },
        [](auto x) -> auto { return std::cos(x) - std::sin(x); });

    SECTION("Evaluate with unit weight") {
        Operon::Node varNode(Operon::NodeType::Variable);
        varNode.HashValue = ds.GetVariable(x).value().Hash;

        Operon::Node dynNode(Operon::NodeType::Dynamic, h);
        dynNode.Arity  = 1;
        dynNode.Length = 1;

        Operon::Tree const tree({ varNode, dynNode });
        auto coeff = tree.GetCoefficients();
        auto r = Operon::Interpreter<Scalar, DT>(&dt, &ds, &tree).Evaluate(coeff, Operon::Range(0, 10));

        REQUIRE(std::ssize(r) == 10);
        for (auto i = 0; i < 10; ++i) {
            auto expected = std::sin(xvals[i]) + std::cos(xvals[i]);
            CHECK(r[i] == Catch::Approx(expected).epsilon(1e-5));
        }
    }

    SECTION("Weight is applied by the adapter") {
        Operon::Node varNode(Operon::NodeType::Variable);
        varNode.HashValue = ds.GetVariable(x).value().Hash;

        Operon::Node dynNode(Operon::NodeType::Dynamic, h);
        dynNode.Arity  = 1;
        dynNode.Length = 1;
        dynNode.Value  = 3.0F; // non-unit weight

        Operon::Tree const tree({ varNode, dynNode });
        auto coeff = tree.GetCoefficients();
        auto r = Operon::Interpreter<Scalar, DT>(&dt, &ds, &tree).Evaluate(coeff, Operon::Range(0, 10));

        REQUIRE(std::ssize(r) == 10);
        for (auto i = 0; i < 10; ++i) {
            auto expected = 3.0F * (std::sin(xvals[i]) + std::cos(xvals[i]));
            CHECK(r[i] == Catch::Approx(expected).epsilon(1e-5));
        }
    }
}

TEST_CASE("RegisterBinary - scalar lambda adapter", "[interpreter]")
{
    using DT     = Operon::DispatchTable<Operon::Scalar>;
    using Scalar = Operon::Scalar;

    // Two variables: x = {1..10}, y = {2..11}
    std::vector<Scalar> xvals(10);
    std::vector<Scalar> yvals(10);
    std::iota(xvals.begin(), xvals.end(), 1.0F);
    std::iota(yvals.begin(), yvals.end(), 2.0F);
    Operon::Dataset ds({"x", "y"}, {xvals, yvals});

    DT dt;
    auto const h = Operon::Hasher{}("test::hypot");

    // f(a, b) = sqrt(a^2 + b^2),  ∂f/∂a = a/f,  ∂f/∂b = b/f
    Operon::RegisterBinary<DT, Scalar>(dt, h,
        [](auto a, auto b) -> auto { return std::sqrt((a*a) + (b*b)); },
        [](auto a, auto b) -> auto { return a / std::sqrt((a*a) + (b*b)); },
        [](auto a, auto b) -> auto { return b / std::sqrt((a*a) + (b*b)); });

    SECTION("Evaluate with unit weight") {
        // Tree: [Variable(x), Variable(y), Dynamic(hypot)]
        // post-order: x at 0, y at 1, hypot at 2
        Operon::Node varX(Operon::NodeType::Variable);
        varX.HashValue = ds.GetVariable("x").value().Hash;

        Operon::Node varY(Operon::NodeType::Variable);
        varY.HashValue = ds.GetVariable("y").value().Hash;

        Operon::Node dynNode(Operon::NodeType::Dynamic, h);
        dynNode.Arity  = 2;
        dynNode.Length = 2;

        Operon::Tree const tree({ varX, varY, dynNode });
        auto coeff = tree.GetCoefficients();
        auto r = Operon::Interpreter<Scalar, DT>(&dt, &ds, &tree).Evaluate(coeff, Operon::Range(0, 10));

        REQUIRE(std::ssize(r) == 10);
        for (auto i = 0; i < 10; ++i) {
            auto expected = std::sqrt((xvals[i]*xvals[i]) + (yvals[i]*yvals[i]));
            CHECK(r[i] == Catch::Approx(expected).epsilon(1e-4));
        }
    }
}

TEST_CASE("PrimitiveSet::AddFunction - user-defined symbol in tree generation", "[interpreter]")
{
    auto const h1 = Operon::Hasher{}("test::sincos");  // unary
    auto const h2 = Operon::Hasher{}("test::hypot");   // binary

    Operon::PrimitiveSet pset;
    pset.SetConfig(Operon::PrimitiveSet::Arithmetic); // built-ins
    pset.AddFunction(h1, /*arity=*/1, /*frequency=*/5);
    pset.AddFunction(h2, /*arity=*/2, /*frequency=*/3);

    SECTION("Both hashes are present in the primitive set") {
        CHECK(pset.Contains(h1));
        CHECK(pset.Contains(h2));
    }

    SECTION("Arity is stored correctly") {
        CHECK(pset.MinimumArity(h1) == 1);
        CHECK(pset.MaximumArity(h1) == 1);
        CHECK(pset.MinimumArity(h2) == 2);
        CHECK(pset.MaximumArity(h2) == 2);
    }

    SECTION("Frequency is stored correctly") {
        CHECK(pset.Frequency(h1) == 5);
        CHECK(pset.Frequency(h2) == 3);
    }

    SECTION("SampleRandomSymbol returns nodes with correct arity") {
        Operon::RandomGenerator rng(1234);
        auto const nSamples = 200;

        // Sample unary — only sincos and built-in unary functions qualify
        for (auto i = 0; i < nSamples; ++i) {
            auto node = pset.SampleRandomSymbol(rng, /*minArity=*/1, /*maxArity=*/1);
            CHECK(node.Arity == 1);
        }

        // Sample binary — only hypot and built-in binary functions qualify
        for (auto i = 0; i < nSamples; ++i) {
            auto node = pset.SampleRandomSymbol(rng, /*minArity=*/2, /*maxArity=*/2);
            CHECK(node.Arity == 2);
        }
    }

    SECTION("AddFunction returns false if the hash is already registered") {
        CHECK_FALSE(pset.AddFunction(h1, 1));
    }
}

TEST_CASE("Auto-diff fallback via Jet<T,1>", "[interpreter]")
{
    using DT     = Operon::DispatchTable<Operon::Scalar>;
    using Scalar = Operon::Scalar;

    std::string x{"x"};
    std::vector<Scalar> xvals(10);
    std::iota(xvals.begin(), xvals.end(), 1.0F); // x = {1, 2, ..., 10}
    Operon::Dataset ds({x}, {xvals});

    auto const hAuto     = Operon::Hasher{}("test::sincos_auto");
    auto const hExplicit = Operon::Hasher{}("test::sincos_explicit");

    // Same function registered two ways: auto-diff vs explicit derivative.
    // Unqualified sin/cos with using-declarations allow ADL to resolve to
    // ceres::sin/cos when called with Jet<T,1> during auto-diff.
    auto primal  = [](auto const& v) -> auto { using std::sin, std::cos; return sin(v) + cos(v); };
    auto dprimal = [](auto const& v) -> auto { using std::sin, std::cos; return cos(v) - sin(v); };

    DT dt;
    Operon::RegisterUnary<DT, Scalar>(dt, hAuto,     primal);           // Jet fallback
    Operon::RegisterUnary<DT, Scalar>(dt, hExplicit, primal, dprimal);  // explicit

    auto makeTree = [&](Operon::Hash h) -> Operon::Tree {
        Operon::Node varNode(Operon::NodeType::Variable);
        varNode.HashValue = ds.GetVariable(x).value().Hash;
        Operon::Node dynNode(Operon::NodeType::Dynamic, h);
        dynNode.Arity  = 1;
        dynNode.Length = 1;
        return Operon::Tree({ varNode, dynNode });
    };

    auto const tAuto     = makeTree(hAuto);
    auto const tExplicit = makeTree(hExplicit);
    auto const coeff     = tAuto.GetCoefficients();
    auto const range     = Operon::Range(0, 10);

    Operon::Interpreter<Scalar, DT> const interpAuto    (&dt, &ds, &tAuto);
    Operon::Interpreter<Scalar, DT> const interpExplicit(&dt, &ds, &tExplicit);

    SECTION("JacRev: auto-diff matches explicit derivative") {
        auto jacAuto     = interpAuto.JacRev(coeff, range);
        auto jacExplicit = interpExplicit.JacRev(coeff, range);
        CHECK(jacAuto.isApprox(jacExplicit, 1e-4F));
    }

    SECTION("JacFwd: auto-diff matches explicit derivative") {
        auto jacAuto     = interpAuto.JacFwd(coeff, range);
        auto jacExplicit = interpExplicit.JacFwd(coeff, range);
        CHECK(jacAuto.isApprox(jacExplicit, 1e-4F));
    }

    SECTION("JacRev matches expected analytic values") {
        // output = f(w*x),  d/dw at w=1 = x * f'(x) = x*(cos(x)-sin(x))
        auto jac = interpAuto.JacRev(coeff, range);
        REQUIRE(jac.rows() == 10);
        for (auto i = 0; i < 10; ++i) {
            auto expected = xvals[i] * (std::cos(xvals[i]) - std::sin(xvals[i]));
            CHECK(jac(i, 0) == Catch::Approx(expected).epsilon(1e-4));
        }
    }
}

TEST_CASE("RegisterFunction - FunctionInfo convenience wrapper", "[interpreter]")
{
    using DT     = Operon::DispatchTable<Operon::Scalar>;
    using Scalar = Operon::Scalar;

    std::string const x{"x"};
    std::vector<Scalar> xvals(5);
    std::iota(xvals.begin(), xvals.end(), 1.0F); // x = {1, 2, 3, 4, 5}
    Operon::Dataset ds({x}, {xvals});

    DT dt;
    Operon::PrimitiveSet pset;
    pset.SetConfig(Operon::PrimitiveSet::Arithmetic);

    Operon::FunctionInfo const info{
        .Hash      = Operon::Hasher{}("test::cube"),
        .Name      = "cube",
        .Desc      = "cube function f(x) = x^3",
        .Arity     = 1,
        .Frequency = 1
    };

    auto primal  = [](auto v) -> auto { return v * v * v; };
    auto dprimal = [](auto v) -> auto { return 3 * v * v; };

    Operon::RegisterUnaryFunction<DT, Scalar>(dt, pset, info, primal, dprimal);

    SECTION("Name and Desc are registered on the node") {
        Operon::Node const dynNode(Operon::NodeType::Dynamic, info.Hash);
        CHECK(dynNode.Name() == "cube");
        CHECK(dynNode.Desc() == "cube function f(x) = x^3");
    }

    SECTION("Hash is present in dispatch table and primitive set") {
        CHECK(dt.Contains(info.Hash));
        CHECK(pset.Contains(info.Hash));
        CHECK(pset.MinimumArity(info.Hash) == 1);
    }

    SECTION("Evaluation is correct") {
        Operon::Node varNode(Operon::NodeType::Variable);
        varNode.HashValue = ds.GetVariable(x).value().Hash;

        Operon::Node dynNode(Operon::NodeType::Dynamic, info.Hash);
        dynNode.Arity  = 1;
        dynNode.Length = 1;

        Operon::Tree const tree({ varNode, dynNode });
        auto coeff = tree.GetCoefficients();
        auto r = Operon::Interpreter<Scalar, DT>(&dt, &ds, &tree).Evaluate(coeff, Operon::Range(0, 5));

        REQUIRE(std::ssize(r) == 5);
        for (auto i = 0; i < 5; ++i) {
            CHECK(r[i] == Catch::Approx(xvals[i] * xvals[i] * xvals[i]).epsilon(1e-5));
        }
    }

    SECTION("InfixFormatter uses registered name") {
        Operon::Node varNode(Operon::NodeType::Variable);
        varNode.HashValue = ds.GetVariable(x).value().Hash;

        Operon::Node dynNode(Operon::NodeType::Dynamic, info.Hash);
        dynNode.Arity  = 1;
        dynNode.Length = 1;

        Operon::Tree const tree({ varNode, dynNode });
        auto formatted = InfixFormatter::Format(tree, ds);
        CHECK(formatted.find("cube") != std::string::npos);
    }
}

} // namespace Operon::Test
