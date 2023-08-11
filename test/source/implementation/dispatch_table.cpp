// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <doctest/doctest.h>
#include <fmt/ranges.h>

#include "operon/interpreter/interpreter.hpp"
#include "operon/parser/infix.hpp"
#include "operon/formatter/formatter.hpp"

namespace dt = doctest;

namespace Operon::Test {

    TEST_CASE("constructors" * dt::test_suite("dispatch_table")) {

        using DT = Operon::DispatchTable<Operon::Scalar>;

        std::string x{"x"};
        std::vector<Operon::Scalar> v{0};
        Operon::Dataset ds({x}, {v});

        auto check = [&](DT const& dt, std::string const& expr, Operon::Scalar expected) {
            Operon::Map<std::string, Operon::Hash> vars;
            auto t = InfixParser::Parse(expr, vars);
            fmt::print("Check expression {} == {}\n", InfixFormatter::Format(t, ds), expected);
            fmt::print("Tree representation:\n{}\n", TreeFormatter::Format(t, ds));
            auto p = t.GetCoefficients();
            auto r = Operon::Interpreter<Operon::Scalar, DT>(dt, ds, t).Evaluate(p, Operon::Range(0, 1));
            CHECK(r[0] == expected);
        };

        // default ctor
        DT dt;
        check(dt, "1 + 2 + 3", 6);
        check(dt, "1 - 2 - 3", -4);
        check(dt, "6 / 3 / 2", 1);
        check(dt, "6 / 3 * 2", 4);

        // copy ctor
        DT dt1(dt);
        check(dt1, "2 * 3 / 4", 1.5);

        // move ctor
        DT dt2(std::move(dt1));
        check(dt2, "sin(1 / 2 * 3.141519)", std::sin(1.0 / 2.0 * 3.141519));

        // ctor from map const&
        auto const& map = dt.GetMap();
        DT dt3(map);
        check(dt3, "cos(3.141519)", std::cos(3.141519f));

        // ctor from map&&
        DT dt4(std::move(map));
        check(dt4, "exp(log(10))", std::exp(std::log(10.f)));
    }
} // namespace Operon::Test
