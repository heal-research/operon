// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERPRETER_DERIVATIVES_HPP
#define OPERON_INTERPRETER_DERIVATIVES_HPP

#include "functions.hpp"
#include <fmt/format.h>

namespace Operon {
    // n-ary functions
    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Add, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Add<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Sub, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sub<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Mul, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Mul<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Div, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Div<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Fmin, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Min<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Fmax, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Max<T, S>(nodes, primal, trace, i, j);
        }
    };

    // binary functions
    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Aq, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Aq<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Pow, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Pow<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Powabs, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Powabs<T, S>(nodes, primal, trace, i, j);
        }
    };

    // unary functions
    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Abs, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Abs<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Acos, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Acos<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Asin, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Asin<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Atan, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Atan<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Cbrt, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Cbrt<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Ceil, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Ceil<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Cos, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Cos<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Cosh, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Cosh<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Exp, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Exp<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Floor, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Floor<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Log, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Log<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Logabs, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Logabs<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Log1p, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Log1p<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Sin, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sin<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Sinh, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sinh<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Sqrt, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sqrt<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Sqrtabs, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sqrtabs<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Square, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Square<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Tan, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Tan<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::BuiltinOp::Tanh, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Tanh<T, S>(nodes, primal, trace, i, j);
        }
    };
} // namespace Operon

#endif
