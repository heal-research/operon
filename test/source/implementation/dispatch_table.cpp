// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "operon/interpreter/interpreter.hpp"
#include "operon/parser/infix.hpp"
#include "operon/formatter/formatter.hpp"

namespace Operon::Test {

TEST_CASE("DispatchTable constructors", "[interpreter]")
{
    using DT = Operon::DispatchTable<Operon::Scalar>;

    std::string x{"x"};
    std::vector<Operon::Scalar> v{0};
    Operon::Dataset ds({x}, {v});

    auto check = [&](DT const& dt, std::string const& expr, Operon::Scalar expected) {
        Operon::Map<std::string, Operon::Hash> vars;
        auto t = InfixParser::Parse(expr, vars);
        auto p = t.GetCoefficients();
        auto r = Operon::Interpreter<Operon::Scalar, DT>(&dt, &ds, &t).Evaluate(p, Operon::Range(0, 1));
        CHECK(r[0] == Catch::Approx(expected));
    };

    SECTION("Default constructor") {
        DT dt;
        check(dt, "1 + 2 + 3", 6);
        check(dt, "1 - 2 - 3", -4);
        check(dt, "6 / 3 / 2", 1);
        check(dt, "6 / 3 * 2", 4);
    }

    SECTION("Copy constructor") {
        DT dt;
        DT dt1(dt);
        check(dt1, "2 * 3 / 4", 1.5);
    }

    SECTION("Move constructor") {
        DT dt;
        DT dt1(dt);
        DT dt2(std::move(dt1));
        check(dt2, "sin(1 / 2 * 3.141519)", std::sin(1.0 / 2.0 * 3.141519));
    }

    SECTION("Construct from map") {
        DT dt;
        auto const& map = dt.GetMap();
        DT dt3(map);
        check(dt3, "cos(3.141519)", std::cos(3.141519f));

        DT dt4(std::move(map));
        check(dt4, "exp(log(10))", std::exp(std::log(10.f)));
    }
}

TEST_CASE("DispatchTable evaluation of expressions", "[interpreter]")
{
    using DT = Operon::DispatchTable<Operon::Scalar>;
    DT dtable;

    std::string x{"x"};
    std::vector<Operon::Scalar> v{0};
    Operon::Dataset ds({x}, {v});

    Operon::Map<std::string, Operon::Hash> vars;

    SECTION("Arithmetic") {
        auto t = InfixParser::Parse("2 + 3 * 4", vars);
        auto r = Interpreter<Operon::Scalar, DT>::Evaluate(t, ds, Range(0, 1));
        CHECK(r[0] == Catch::Approx(14.0f));
    }

    SECTION("Transcendental functions") {
        auto t = InfixParser::Parse("exp(1)", vars);
        auto r = Interpreter<Operon::Scalar, DT>::Evaluate(t, ds, Range(0, 1));
        CHECK(r[0] == Catch::Approx(std::exp(1.0f)));

        t = InfixParser::Parse("log(exp(1))", vars);
        r = Interpreter<Operon::Scalar, DT>::Evaluate(t, ds, Range(0, 1));
        CHECK(r[0] == Catch::Approx(1.0f).epsilon(1e-3));
    }
}

} // namespace Operon::Test
