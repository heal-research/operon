// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INTERPRETER_FUNCTIONS_HPP
#define OPERON_INTERPRETER_FUNCTIONS_HPP

#include "operon/core/node.hpp"
#include "operon/core/dispatch.hpp"
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
    template<typename T, std::size_t S>
    auto Ptr(Backend::View<T, S> view, std::integral auto i) {
        return view.data_handle() + (i * S);
    }

    // n-ary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Add, C, S> {
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
    struct Func<T, Operon::NodeType::Mul, C, S> {
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
    struct Func<T, Operon::NodeType::Sub, C, S> {
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
    struct Func<T, Operon::NodeType::Div, C, S> {
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
    struct Func<T, Operon::NodeType::Fmin, C, S> {
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
    struct Func<T, Operon::NodeType::Fmax, C, S> {
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
    struct Func<T, Operon::NodeType::Aq, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i, std::integral auto j) {
            auto const w = nodes[result].Value;
            Backend::Aq<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i), Ptr<T, S>(view, j));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Pow, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i, std::integral auto j) {
            auto const w = nodes[result].Value;
            Backend::Pow<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i), Ptr<T, S>(view, j));
        }
    };

    // unary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Abs, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Abs<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    // unary operations
    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Square, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Square<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Exp, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Exp<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Log, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Log<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Logabs, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Logabs<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Log1p, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Log1p<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sqrt, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Sqrt<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sqrtabs, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Sqrtabs<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Cbrt, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Cbrt<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Ceil, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Ceil<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Floor, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Floor<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sin, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Sin<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Cos, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Cos<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Tan, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Tan<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Asin, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Asin<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Acos, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Acos<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Atan, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Atan<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Sinh, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Sinh<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Cosh, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Cosh<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };

    template<typename T, bool C, std::size_t S>
    struct Func<T, Operon::NodeType::Tanh, C, S> {
        auto operator()(Operon::Vector<Operon::Node> const& nodes, Backend::View<T, S> view, std::integral auto result, std::integral auto i) {
            auto const w = nodes[result].Value;
            Backend::Tanh<T, S>(Ptr<T, S>(view, result), w, Ptr<T, S>(view, i));
        }
    };
} // namespace Operon

#endif
