// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_FUNCTIONS_HPP
#define OPERON_INTERPRETER_FUNCTIONS_HPP

#include "operon/core/node.hpp"
#if defined(OPERON_MATH_EIGEN)
#include "operon/interpreter/backend/eigen.hpp"
#elif defined(OPERON_MATH_EVE)
#include "operon/interpreter/backend/eve.hpp"
#elif defined(OPERON_MATH_ARMA)
#include "operon/interpreter/backend/arma.hpp"
#elif defined(OPERON_MATH_BLAZE)
#include "operon/interpreter/backend/blaze.hpp"
#elif defined(OPERON_MATH_FASTOR)
#include "operon/interpreter/backend/fastor.hpp"
#elif defined(OPERON_MATH_STL)
#include "operon/interpreter/backend/plain.hpp"
#elif defined(OPERON_MATH_VDT)
#include "operon/interpreter/backend/vdt.hpp"
#elif defined(OPERON_MATH_XTENSOR)
#include "operon/interpreter/backend/xtensor.hpp"
#elif defined(OPERON_MATH_FAST_V1)
#include "operon/interpreter/backend/fast_v1.hpp"
#elif defined(OPERON_MATH_FAST_V2)
#include "operon/interpreter/backend/fast_v2.hpp"
#elif defined(OPERON_MATH_FAST_V3)
#include "operon/interpreter/backend/fast_v3.hpp"
#endif

namespace Operon {
    // utility
    template<typename T, std::size_t S>
    auto Fill(Backend::View<T, S> view, int idx, T value) {
        auto* p = view.data_handle() + idx * S;
        std::fill_n(p, S, value);
    };

    // detect missing specializations
    template<typename T, Operon::NodeType N = Operon::NodeTypes::NoType, bool C = false, std::size_t S = Backend::BatchSize<T>>
    struct Func {
        auto operator()(Backend::View<T, S> /*unused*/, std::integral auto /*unused*/, std::integral auto... /*unused*/) {
            throw std::runtime_error(fmt::format("backend error: missing specialization for function: {}\n", Operon::Node{N}.Name()));
        }
    };

    // n-ary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Add, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto... args) {
            auto* h = view.data_handle();
            if constexpr (C) {
                Backend::Add<T, S>(h + result * S, h + result * S, (h + args * S)...);
            } else {
                Backend::Add<T, S>(h + result * S, (h + args * S)...);
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Mul, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto... args) {
            auto* h = view.data_handle();
            if constexpr (C) {
                Backend::Mul<T, S>(h + result * S, h + result * S, (h + args * S)...);
            } else {
                Backend::Mul<T, S>(h + result * S, (h + args * S)...);
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sub, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto first, std::integral auto... args) {
            auto* h = view.data_handle();

            if constexpr (C) {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Sub<T, S>(h + result * S, h + result * S, h + first * S);
                } else {
                    Backend::Sub<T, S>(h + result * S, h + result * S, h + first * S, (h + args * S)...);
                }
            } else {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Neg<T, S>(h + result * S, h + first * S);
                } else {
                    Backend::Sub<T, S>(h + result * S, h + first * S, (h + args * S)...);
                }
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Div, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto first, std::integral auto... args) {
            auto* h = view.data_handle();

            if constexpr (C) {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Div<T, S>(h + result * S, h + result * S, h + first * S);
                } else {
                    Backend::Div<T, S>(h + result * S, h + result * S, h + first * S, (h + args * S)...);
                }
            } else {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Inv<T, S>(h + result * S, h + first * S);
                } else {
                    Backend::Div<T, S>(h + result * S, h + first * S, (h + args * S)...);
                }
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Fmin, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto first, std::integral auto... args) {
            auto* h = view.data_handle();

            if constexpr (C) {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Min<T, S>(h + result * S, h + result * S, h + first * S);
                } else {
                    Backend::Min<T, S>(h + result * S, h + result * S, h + first * S, (h + args * S)...);
                }
            } else {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Cpy<T, S>(h + result * S, h + first * S);
                } else {
                    Backend::Min<T, S>(h + result * S, h + first * S, (h + args * S)...);
                }
            }
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Fmax, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto first, std::integral auto... args) {
            auto* h = view.data_handle();

            if constexpr (C) {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Max<T, S>(h + result * S, h + result * S, h + first * S);
                } else {
                    Backend::Max<T, S>(h + result * S, h + result * S, h + first * S, (h + args * S)...);
                }
            } else {
                if constexpr (sizeof...(args) == 0) {
                    Backend::Cpy<T, S>(h + result * S, h + first * S);
                } else {
                    Backend::Max<T, S>(h + result * S, h + first * S, (h + args * S)...);
                }
            }
        }
    };

    // binary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Aq, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i, std::integral auto j) {
            auto* h = view.data_handle();
            Backend::Aq<T, S>(h + result * S, h + i * S, h + j * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Pow, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i, std::integral auto j) {
            auto* h = view.data_handle();
            Backend::Pow<T, S>(h + result * S, h + i * S, h + j * S);
        }
    };

    // unary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Abs, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Abs<T, S>(h + result * S, h + i * S);
        }
    };

    // unary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Square, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Square<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Exp, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Exp<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Log, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Log<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Logabs, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Logabs<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Log1p, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Log1p<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sqrt, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Sqrt<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sqrtabs, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Sqrtabs<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Cbrt, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Cbrt<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Ceil, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Ceil<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Floor, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Floor<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sin, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Sin<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Cos, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Cos<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Tan, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Tan<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Asin, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Asin<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Acos, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Acos<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Atan, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Atan<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sinh, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Sinh<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Cosh, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Cosh<T, S>(h + result * S, h + i * S);
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Tanh, C, S> {
        auto operator()(Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto* h = view.data_handle();
            Backend::Tanh<T, S>(h + result * S, h + i * S);
        }
    };
} // namespace Operon

#endif
