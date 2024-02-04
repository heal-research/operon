// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_DERIVATIVES_HPP
#define OPERON_INTERPRETER_DERIVATIVES_HPP

#include "functions.hpp"

namespace Operon {
    // default function to catch any missing template specializations
    template<typename T, Operon::NodeType N  = Operon::NodeTypes::NoType, std::size_t S = Backend::BatchSize<T>>
    struct Diff {
        auto operator()(std::vector<Operon::Node> const&, Backend::View<T const, S>, Backend::View<T>, std::integral auto, std::integral auto) {
            throw std::runtime_error(fmt::format("backend error: missing specialization for derivative: {}\n", Operon::Node{N}.Name()));
        }
    };

    // n-ary functions
    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Add, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Add<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Sub, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sub<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Mul, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Mul<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Div, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Div<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Fmin, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Min<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Fmax, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Max<T, S>(nodes, primal, trace, i, j);
        }
    };

    // binary functions
    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Aq, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Aq<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Pow, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Pow<T, S>(nodes, primal, trace, i, j);
        }
    };

    // unary functions
    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Abs, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Abs<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Acos, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Acos<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Asin, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Asin<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Atan, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Atan<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Cbrt, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Cbrt<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Ceil, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Ceil<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Cos, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Cos<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Cosh, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Cosh<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Exp, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Exp<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Floor, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Floor<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Log, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Log<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Logabs, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Logabs<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Log1p, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Log1p<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Sin, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sin<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Sinh, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sinh<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Sqrt, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sqrt<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Sqrtabs, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Sqrtabs<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Square, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Square<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Tan, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Tan<T, S>(nodes, primal, trace, i, j);
        }
    };

    template<typename T, std::size_t S>
    struct Diff<T, Operon::NodeType::Tanh, S> {
        auto operator()(std::vector<Operon::Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T> trace, std::integral auto i, std::integral auto j) {
            Backend::Tanh<T, S>(nodes, primal, trace, i, j);
        }
    };
} // namespace Operon

#endif