/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/nnls.hpp"
#include "core/format.hpp"
#include "core/stats.hpp"
#include "core/metrics.hpp"

#include <doctest/doctest.h>

#include "core/nnls_tiny.hpp"
#include <ceres/tiny_solver.h>

namespace Operon {
namespace Test {
TEST_CASE("Evaluation correctness")
{
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();

    auto range = Range { 0, 10 };
    auto targetValues = ds.GetValues("Y").subspan(range.Start(), range.Size());

    auto x1Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X1"; });
    auto x2Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X2"; });
    auto x3Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X3"; });
    auto x4Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X4"; });
    auto x5Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X5"; });
    auto x6Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X6"; });
    auto x7Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X7"; });
    auto x8Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X8"; });
    auto x9Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X9"; });
    auto x10Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X10"; });

    auto x1Val = ds.Values().col(x1Var.Index).segment(range.Start(), range.Size());
    auto x2Val = ds.Values().col(x2Var.Index).segment(range.Start(), range.Size());
    auto x3Val = ds.Values().col(x3Var.Index).segment(range.Start(), range.Size());
    auto x4Val = ds.Values().col(x4Var.Index).segment(range.Start(), range.Size());
    auto x5Val = ds.Values().col(x5Var.Index).segment(range.Start(), range.Size());
    auto x6Val = ds.Values().col(x6Var.Index).segment(range.Start(), range.Size());
    auto x7Val = ds.Values().col(x7Var.Index).segment(range.Start(), range.Size());
    auto x8Val = ds.Values().col(x8Var.Index).segment(range.Start(), range.Size());
    auto x9Val = ds.Values().col(x9Var.Index).segment(range.Start(), range.Size());
    auto x10Val = ds.Values().col(x10Var.Index).segment(range.Start(), range.Size());

    auto x1 = Node(NodeType::Variable, x1Var.Hash); x1.Value = 1;
    auto x2 = Node(NodeType::Variable, x2Var.Hash); x2.Value = 1;
    auto x3 = Node(NodeType::Variable, x3Var.Hash); x3.Value = 1;
    auto x4 = Node(NodeType::Variable, x4Var.Hash); x4.Value = 1;
    auto x5 = Node(NodeType::Variable, x5Var.Hash); x5.Value = 1;
    auto x6 = Node(NodeType::Variable, x6Var.Hash); x6.Value = 1;
    auto x7 = Node(NodeType::Variable, x7Var.Hash); x7.Value = 1;
    auto x8 = Node(NodeType::Variable, x8Var.Hash); x8.Value = 1;
    auto x9 = Node(NodeType::Variable, x9Var.Hash); x9.Value = 1;
    auto x10 = Node(NodeType::Variable, x10Var.Hash); x10.Value = 1;

    auto add = Node(NodeType::Add);
    auto sub = Node(NodeType::Sub);
    auto mul = Node(NodeType::Mul);
    auto div = Node(NodeType::Div);

    Tree tree;

    auto decimals = [](auto v) {
        auto s = fmt::format("{:.50f}", v);
        size_t d = 0;
        auto p = s.find('.');
        while(s[++p] == '0')
            ++d;
        return d;
    };

    SUBCASE("Addition")
    {
        auto eps = 1e-6;

        // binary evaluation
        add.Arity = 2;
        tree = Tree({ x1, x2, add });
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res = x2Val + x1Val; // this is the order of arguments in the evaluation routine
        auto estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(std::fabs(estimatedValues[i] - res[i]) < eps);
        }

        // add 5 terms
        add.Arity = 5;
        tree = Tree({ x1, x2, x3, x4, x5, add});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res5 = x5Val + x4Val + x3Val + x2Val + x1Val;
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res5[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }

        // add 6 terms
        add.Arity = 6;
        tree = Tree({ x1, x2, x3, x4, x5, x6, add});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res6 = x6Val + x5Val + x4Val + x3Val + x2Val + x1Val;
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res6[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(std::fabs(estimatedValues[i] - res6[i]) < eps);
        }

        // add 10 terms
        add.Arity = 10;
        tree = Tree({ x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, add});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res10 = x10Val + x9Val + x8Val + x7Val + x6Val + x5Val + x4Val + x3Val + x2Val + x1Val;
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res10[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(std::fabs(estimatedValues[i] - res10[i]) < eps);
        }
    }

    SUBCASE("Subtraction")
    {
        auto eps = 1e-6;

        // binary evaluation
        sub.Arity = 2;
        tree = Tree({ x1, x2, sub });
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res = x2Val - x1Val;
        auto estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }

        // 5 terms
        sub.Arity = 5;
        tree = Tree({ x1, x2, x3, x4, x5, sub });
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res5 = x5Val - (x4Val + x3Val + x2Val + x1Val);
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res5[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }

        // 6 terms
        sub.Arity = 6;
        tree = Tree({ x1, x2, x3, x4, x5, x6, sub});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res6 = x6Val - (x5Val + x4Val + x3Val + x2Val + x1Val);
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res6[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }

        // 9 terms
        sub.Arity = 9;
        tree = Tree({ x1, x2, x3, x4, x5, x6, x7, x8, x9,  sub});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res9 = x9Val - (x8Val + x7Val + x6Val + x5Val + x4Val + x3Val + x2Val + x1Val);
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(std::fabs(estimatedValues[i] - res9[i]) < eps);
        }

        // 10 terms
        sub.Arity = 10;
        tree = Tree({ x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, sub});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res10 = x10Val - (x9Val + x8Val + x7Val + x6Val + x5Val + x4Val + x3Val + x2Val + x1Val);
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(std::fabs(estimatedValues[i] - res10[i]) < eps);
        }
    }

    SUBCASE("Multiplication")
    {
        auto eps = 1e-6;

        // binary evaluation
        mul.Arity = 2;
        tree = Tree({ x1, x2, mul });
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res = x2Val * x1Val;
        auto estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }

        // 5 terms
        mul.Arity = 5;
        tree = Tree({ x1, x2, x3, x4, x5, mul });
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res5 = x5Val * x4Val * x3Val * x2Val * x1Val;
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res5[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }

        // 6 terms
        mul.Arity = 6;
        tree = Tree({ x1, x2, x3, x4, x5, x6, mul});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res6 = x6Val * x5Val * x4Val * x3Val * x2Val * x1Val;
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res6[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }

        // 9 terms
        mul.Arity = 9;
        tree = Tree({ x1, x2, x3, x4, x5, x6, x7, x8, x9, mul});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res9 = x9Val * x8Val * x7Val * x6Val * x5Val * x4Val * x3Val * x2Val * x1Val;
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res9[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }

        // 10 terms
        mul.Arity = 10;
        tree = Tree({ x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, mul});
        fmt::print("{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res10 = x10Val * x9Val * x8Val * x7Val * x6Val * x5Val * x4Val * x3Val * x2Val * x1Val;
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res10[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision.\n", decimals(v));
            }
            CHECK(v < eps);
        }
    }

    SUBCASE("Division")
    {
        auto eps = 1e-6;

        // binary evaluation
        div.Arity = 2;
        tree = Tree({ x1, x2, div });
        fmt::print(fmt::fg(fmt::terminal_color::green), "{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res = x2Val / x1Val;
        auto estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision  (v = {:.12f}, r = {:.12f}), |v-r| = {:.12f}).\n", decimals(v), estimatedValues[i], res[i], v);
            }
            CHECK(v < eps);
        }

        // 5 terms
        div.Arity = 5;
        tree = Tree({ x1, x2, x3, x4, x5, div });
        fmt::print(fmt::fg(fmt::terminal_color::green), "{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res5 = x5Val / (x4Val * x3Val * x2Val * x1Val);
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res5[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision  (v = {:.12f}, r = {:.12f}), |v-r| = {:.12f}).\n", decimals(v), estimatedValues[i], res5[i], v);
            }
            CHECK(v < eps);
        }

        // 6 terms
        div.Arity = 6;
        tree = Tree({ x1, x2, x3, x4, x5, x6, div});
        fmt::print(fmt::fg(fmt::terminal_color::green), "{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res6 = x6Val / (x5Val * x4Val * x3Val * x2Val * x1Val);
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res6[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision  (v = {:.12f}, r = {:.12f}, |v-r| = {:.12f}).\n", decimals(v), estimatedValues[i], res6[i], v);
            }
            CHECK(v < eps);
        }

        // 9 terms
        div.Arity = 9;
        tree = Tree({ x1, x2, x3, x4, x5, x6, x7, x8, x9, div});
        fmt::print(fmt::fg(fmt::terminal_color::green), "{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res9 = x9Val / (x8Val * x7Val * x6Val * x5Val * x4Val * x3Val * x2Val * x1Val);
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res9[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision  (v = {:.12f}, r = {:.12f}), |v-r| = {:.12f}).\n", decimals(v), estimatedValues[i], res9[i], v);
            }
            CHECK(v < eps);
        }

        // 10 terms
        div.Arity = 10;
        tree = Tree({ x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, div});
        fmt::print(fmt::fg(fmt::terminal_color::green), "{}\n", InfixFormatter::Format(tree, ds, 2));

        auto res10 = x10Val / (x9Val * x8Val * x7Val * x6Val * x5Val * x4Val * x3Val * x2Val * x1Val);
        estimatedValues = Evaluate<Operon::Scalar>(tree, ds, range);

        for (size_t i = 0; i < estimatedValues.size(); ++i) {
            auto v = std::fabs(estimatedValues[i] - res10[i]);
            if (v > 1e-12) {
                fmt::print(fmt::fg(fmt::terminal_color::yellow), "warning: {} decimals result precision  (v = {:.12f}, r = {:.12f}), |v-r| = {:.12f}).\n", decimals(v), estimatedValues[i], res10[i], v);
            }
            CHECK(v < eps);
        }
    }
}

TEST_CASE("Constant optimization (autodiff)")
{
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();

    auto range = Range { 0, 250 };
    auto targetValues = ds.GetValues("Y").subspan(range.Start(), range.Size());

    auto x1Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X1"; });
    auto x2Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X2"; });
    auto x3Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X3"; });
    auto x4Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X4"; });
    auto x5Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X5"; });
    auto x6Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X6"; });
    auto x7Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X7"; });
    auto x8Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X8"; });
    auto x9Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X9"; });
    auto x10Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X10"; });

    auto x1 = Node(NodeType::Variable, x1Var.Hash);
    x1.Value = 0.001;
    auto x2 = Node(NodeType::Variable, x2Var.Hash);
    x2.Value = 0.001;
    auto x3 = Node(NodeType::Variable, x3Var.Hash);
    x3.Value = 0.001;
    auto x4 = Node(NodeType::Variable, x4Var.Hash);
    x4.Value = 0.001;
    auto x5 = Node(NodeType::Variable, x5Var.Hash);
    x5.Value = 0.001;
    auto x6 = Node(NodeType::Variable, x6Var.Hash);
    x6.Value = 0.001;
    auto x7 = Node(NodeType::Variable, x7Var.Hash);
    x7.Value = 0.001;
    auto x9 = Node(NodeType::Variable, x9Var.Hash);
    x9.Value = 0.001;
    auto x10 = Node(NodeType::Variable, x10Var.Hash);
    x10.Value = 0.001;

    auto add = Node(NodeType::Add);
    auto mul = Node(NodeType::Mul);

    auto poly10 = Tree {
        x1,
        x2,
        mul,
        x3,
        x4,
        mul,
        add,
        x5,
        x6,
        mul,
        add,
        x1,
        x7,
        mul,
        x9,
        mul,
        add,
        x3,
        x6,
        mul,
        x10,
        mul,
        add,
    };
    poly10.UpdateNodes();
    fmt::print("{}\n", InfixFormatter::Format(poly10, ds, 6));

    auto coef = OptimizeAutodiff(poly10, ds, targetValues, range, 100, true, true);
    fmt::print("{}\n", InfixFormatter::Format(poly10, ds, 6));
}

TEST_CASE("Constant optimization (tiny solver)") 
{
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();

    auto range = Range { 0, 250 };
    auto targetValues = ds.GetValues("Y").subspan(range.Start(), range.Size());

    auto x1Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X1"; });
    auto x2Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X2"; });
    auto x3Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X3"; });
    auto x4Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X4"; });
    auto x5Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X5"; });
    auto x6Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X6"; });
    auto x7Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X7"; });
    auto x8Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X8"; });
    auto x9Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X9"; });
    auto x10Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X10"; });

    auto x1 = Node(NodeType::Variable, x1Var.Hash);
    x1.Value = 0.001;
    auto x2 = Node(NodeType::Variable, x2Var.Hash);
    x2.Value = 0.001;
    auto x3 = Node(NodeType::Variable, x3Var.Hash);
    x3.Value = 0.001;
    auto x4 = Node(NodeType::Variable, x4Var.Hash);
    x4.Value = 0.001;
    auto x5 = Node(NodeType::Variable, x5Var.Hash);
    x5.Value = 0.001;
    auto x6 = Node(NodeType::Variable, x6Var.Hash);
    x6.Value = 0.001;
    auto x7 = Node(NodeType::Variable, x7Var.Hash);
    x7.Value = 0.001;
    auto x9 = Node(NodeType::Variable, x9Var.Hash);
    x9.Value = 0.001;
    auto x10 = Node(NodeType::Variable, x10Var.Hash);
    x10.Value = 0.001;

    auto add = Node(NodeType::Add);
    auto mul = Node(NodeType::Mul);

    auto poly10 = Tree {
        x1,
        x2,
        mul,
        x3,
        x4,
        mul,
        add,
        x5,
        x6,
        mul,
        add,
        x1,
        x7,
        mul,
        x9,
        mul,
        add,
        x3,
        x6,
        mul,
        x10,
        mul,
        add,
    };
    poly10.UpdateNodes();
    fmt::print("{}\n", InfixFormatter::Format(poly10, ds, 6));

    auto coeff = poly10.GetCoefficients();
    Eigen::Matrix<double, Eigen::Dynamic, 1> x0(coeff.size());

    for (size_t i = 0; i < coeff.size(); ++i) {
        x0(i) = coeff[i];
    }

    std::cout << "x0: " << x0.transpose() << "\n";

    auto target = ds.GetValues("Y").subspan(range.Start(), range.Size());

    ceres::TinySolver<TinyCostFunction> solver;
    TinyCostFunction function(poly10, ds, target, range);
    auto summary = solver.Solve(function, &x0); 

    std::cout << "x_final: " << x0.transpose() << "\n";
}

TEST_CASE("Constant optimization (numeric)")
{
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();

    auto range = Range { 0, 250 };
    auto targetValues = ds.GetValues("Y").subspan(range.Start(), range.Size());

    auto x1Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X1"; });
    auto x2Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X2"; });
    auto x3Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X3"; });
    auto x4Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X4"; });
    auto x5Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X5"; });
    auto x6Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X6"; });
    auto x7Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X7"; });
    auto x8Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X8"; });
    auto x9Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X9"; });
    auto x10Var = *std::find_if(variables.begin(), variables.end(), [](auto& v) { return v.Name == "X10"; });

    auto x1 = Node(NodeType::Variable, x1Var.Hash);
    x1.Value = 0.001;
    auto x2 = Node(NodeType::Variable, x2Var.Hash);
    x2.Value = 0.001;
    auto x3 = Node(NodeType::Variable, x3Var.Hash);
    x3.Value = 0.001;
    auto x4 = Node(NodeType::Variable, x4Var.Hash);
    x4.Value = 0.001;
    auto x5 = Node(NodeType::Variable, x5Var.Hash);
    x5.Value = 0.001;
    auto x6 = Node(NodeType::Variable, x6Var.Hash);
    x6.Value = 0.001;
    auto x7 = Node(NodeType::Variable, x7Var.Hash);
    x7.Value = 0.001;
    auto x9 = Node(NodeType::Variable, x9Var.Hash);
    x9.Value = 0.001;
    auto x10 = Node(NodeType::Variable, x10Var.Hash);
    x10.Value = 0.001;

    auto add = Node(NodeType::Add);
    auto mul = Node(NodeType::Mul);

    auto poly10 = Tree {
        x1,
        x2,
        mul,
        x3,
        x4,
        mul,
        add,
        x5,
        x6,
        mul,
        add,
        x1,
        x7,
        mul,
        x9,
        mul,
        add,
        x3,
        x6,
        mul,
        x10,
        mul,
        add,
    };
    poly10.UpdateNodes();
    fmt::print("{}\n", InfixFormatter::Format(poly10, ds, 6));

    auto coef = OptimizeNumeric(poly10, ds, targetValues, range, 100, true, true);
    fmt::print("{}\n", InfixFormatter::Format(poly10, ds, 6));
}

} // namespace Test
} // namespace Operon

