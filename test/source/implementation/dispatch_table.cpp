// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include <fmt/ranges.h>

#include "operon/interpreter/interpreter.hpp"
#include "operon/parser/infix.hpp"
#include "operon/core/format.hpp"

namespace dt = doctest;

namespace Operon::Test {

    TEST_CASE("constructors" * dt::test_suite("dispatch_table")) {

        using DispatchTable = Operon::DispatchTable<Operon::Scalar>;
        using Interpreter = Operon::GenericInterpreter<Operon::Scalar>;

        Operon::Variable x{ "X", Operon::Hash{1234}, 0 };
        std::vector<Operon::Scalar> v{0};
        Operon::Dataset ds({x}, {v});

        auto check = [&](DispatchTable const& dt, std::string const& expr, Operon::Scalar expected) {
            std::unordered_map<std::string, Operon::Hash> vars;
            auto t = InfixParser::ParseDefault(expr, vars);
            fmt::print("Check expression {} == {}\n", InfixFormatter::Format(t, ds), expected);
            Interpreter interp(dt);
            auto r = interp.Evaluate<Operon::Scalar>(t, ds, Operon::Range{0, 1});
            CHECK(r[0] == expected);
        };

        // default ctor
        DispatchTable dt;
        check(dt, "1 + 2 + 3", 6);

        // copy ctor
        DispatchTable dt1(dt);
        check(dt1, "2 * 3 / 4", 1.5);

        // move ctor
        DispatchTable dt2(std::move(dt1));
        check(dt2, "sin(1 / 2 * 3.141519)", std::sin(1.0 / 2.0 * 3.141519));

        // ctor from map const&
        auto const& map = dt.GetMap();
        DispatchTable dt3(map);
        check(dt3, "cos(3.141519)", std::cos(3.141519f));

        // ctor from map&&
        DispatchTable dt4(std::move(map));
        check(dt4, "exp(log(10))", std::exp(std::log(10.f)));
    }
} // namespace Operon::Test
