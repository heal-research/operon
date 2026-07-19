// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERPRETER_FUNCTIONS_HPP
#define OPERON_INTERPRETER_FUNCTIONS_HPP

#include "operon/core/node.hpp"
#include "operon/core/dispatch.hpp"
#if defined(OPERON_MATH_EVE)
#include "operon/interpreter/backend/eve.hpp"
#elif defined(OPERON_MATH_MAD_EVE)
#include "operon/interpreter/backend/mad_eve.hpp"
#elif defined(OPERON_MATH_EIGEN)
#include "operon/interpreter/backend/eigen.hpp"
#elif defined(OPERON_MATH_STL)
#include "operon/interpreter/backend/plain.hpp"
#endif

namespace Operon {
    template<typename T, std::size_t S>
    auto Ptr(Backend::View<T, S> view, std::integral auto i) {
        return view.data_handle() + (i * S);
    }

    // n-ary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Add, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto... args) {
            auto const w = nodes[result].Value;
            if constexpr (C) {
                Backend::Add<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, args)...);
            } else {
                Backend::Add<T, S>(Ptr<T, S>(view, result), w, (Ptr<T, S>(view, args))...);
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Mul, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto... args) {
            auto const w = nodes[result].Value;
            if constexpr (C) {
                Backend::Mul<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), (Ptr<T, S>(view, args))...);
            } else {
                Backend::Mul<T, S>(Ptr<T, S>(view, result), w, (Ptr<T, S>(view, args))...);
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Sub, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto first, std::integral auto... args) {
            auto const w = nodes[result].Value;
            if constexpr (C) {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Sub<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, first));
                } else {
                    Backend::Sub<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, first), (Ptr<T, S>(view, args))...);
                }
            } else {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Neg<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, first));
                } else {
                    Backend::Sub<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, first), (Ptr<T, S>(view, args))...);
                }
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Div, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto first, std::integral auto... args) {
            auto const w = nodes[result].Value;
            if constexpr (C) {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Div<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, first));
                } else {
                    Backend::Div<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, first), (Ptr<T, S>(view, args))...);
                }
            } else {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Inv<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, first));
                } else {
                    Backend::Div<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, first), (Ptr<T, S>(view, args))...);
                }
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Fmin, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto first, std::integral auto... args) {
            auto const w = nodes[result].Value;
            if constexpr (C) {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Min<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, first));
                } else {
                    Backend::Min<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, first), (Ptr<T, S>(view, args))...);
                }
            } else {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Cpy<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, first));
                } else {
                    Backend::Min<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, first), (Ptr<T, S>(view, args))...);
                }
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Fmax, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto first, std::integral auto... args) {
            auto const w = nodes[result].Value;
            if constexpr (C) {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Max<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, first));
                } else {
                    Backend::Max<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, result), Ptr<T, S>(view, first), (Ptr<T, S>(view, args))...);
                }
            } else {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Cpy<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, first));
                } else {
                    Backend::Max<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, first), (Ptr<T, S>(view, args))...);
                }
            }
        }
    };

    // binary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Aq, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i, std::integral auto j) {
            auto const w = nodes[result].Value;
            Backend::Aq<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i), Ptr<T, S>(view, j));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Pow, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i, std::integral auto j) {
            auto const w = nodes[result].Value;
            Backend::Pow<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i), Ptr<T, S>(view, j));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Powabs, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i, std::integral auto j) {
            auto const w = nodes[result].Value;
            Backend::Powabs<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i), Ptr<T, S>(view, j));
        }
    };

    // unary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Abs, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Abs<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    // unary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Square, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Square<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Exp, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Exp<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Log, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Log<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Logabs, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Logabs<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Log1p, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Log1p<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Sqrt, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Sqrt<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Sqrtabs, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Sqrtabs<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Cbrt, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Cbrt<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Ceil, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Ceil<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Floor, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Floor<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Sin, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Sin<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Cos, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Cos<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Tan, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Tan<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Asin, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Asin<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Acos, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Acos<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Atan, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Atan<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Sinh, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Sinh<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Cosh, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Cosh<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::BuiltinOp::Tanh, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Tanh<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };
} // namespace Operon

#endif
