// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_EVAL_DETAIL
#define OPERON_EVAL_DETAIL

#include "core/node.hpp"
#include "core/types.hpp"
#include "interpreter/functions.hpp"

#include "robin_hood.h"
#include <Eigen/Dense>
#include <fmt/core.h>

#include <tuple>

namespace Operon {

namespace detail {
    // this should be good enough - tests show 512 is about optimal
    template<typename T>
    struct batch_size {
        static const size_t value = 512 / sizeof(T);
    };

    template<typename T>
    using eigen_t = typename Eigen::Array<T, batch_size<T>::value, Eigen::Dynamic, Eigen::ColMajor>;

    template<typename T>
    using eigen_ref = Eigen::Ref<eigen_t<T>, Eigen::Unaligned, Eigen::Stride<batch_size<T>::value, 1>>;

    // dispatching mechanism
    // compared to the simple/naive way of evaluating n-ary symbols, this method has the following advantages:
    // 1) improved performance: the naive method accumulates into the result for each argument, leading to unnecessary assignments
    // 2) minimizing the number of intermediate steps which might improve floating point accuracy of some operations
    //    if arity > 4, one accumulation is performed every 4 args
    template<NodeType Type, typename T>
    inline void dispatch_op_nary(eigen_t<T>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, size_t /* row number - not used */)
    {
        static_assert(Type < NodeType::Aq);
        auto result = m.col(parentIndex);
        const auto f = [](bool cont, decltype(result) res, auto&&... args) {
            if (cont) {
                ContinuedFunction<Type>{}(res, std::forward<decltype(args)>(args)...);
            } else {
                Function<Type>{}(res, std::forward<decltype(args)>(args)...);
            }
        };
        const auto nextArg = [&](size_t i) { return i - (nodes[i].Length + 1); };

        auto arg1 = parentIndex - 1;

        bool continued = false;

        int arity = nodes[parentIndex].Arity;
        while (arity > 0) {
            switch (arity) {
            case 1: {
                f(continued, result, m.col(arg1));
                arity = 0;
                break;
            }
            case 2: {
                auto arg2 = nextArg(arg1);
                f(continued, result, m.col(arg1), m.col(arg2));
                arity = 0;
                break;
            }
            case 3: {
                auto arg2 = nextArg(arg1), arg3 = nextArg(arg2);
                f(continued, result, m.col(arg1), m.col(arg2), m.col(arg3));
                arity = 0;
                break;
            }
            default: {
                auto arg2 = nextArg(arg1), arg3 = nextArg(arg2), arg4 = nextArg(arg3);
                f(continued, result, m.col(arg1), m.col(arg2), m.col(arg3), m.col(arg4));
                arity -= 4;
                arg1 = nextArg(arg4);
                break;
            }
            }
            continued = true;
        }
    }

    template<NodeType Type, typename T>
    inline void dispatch_op_unary(eigen_t<T>& m, Operon::Vector<Node> const&, size_t i, size_t /* row number - not used */)
    {
        static_assert(Type < NodeType::Constant && Type > NodeType::Pow);
        Function<Type>{}(m.col(i), m.col(i - 1));
    }

    template<NodeType Type, typename T>
    inline void dispatch_op_binary(eigen_t<T>& m, Operon::Vector<Node> const& nodes, size_t i, size_t /* row number - not used */)
    {
        static_assert(Type < NodeType::Abs && Type > NodeType::Div);
        auto j = i - 1;
        auto k = j - nodes[j].Length - 1;
        Function<Type>{}(m.col(i), m.col(j), m.col(k));
    }

    template<NodeType Type, typename T>
    inline void dispatch_op_simple_unary_or_binary(eigen_t<T>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, size_t /* row number - not used */)
    {
        auto r = m.col(parentIndex);
        size_t i = parentIndex - 1;
        size_t arity = nodes[parentIndex].Arity;

        Function<Type> f{};

        if (arity == 1) {
            f(r, m.col(i));
        } else {
            auto j = i - (nodes[i].Length + 1);
            f(r, m.col(i), m.col(j));
        }
    }

    template<NodeType Type, typename T>
    inline void dispatch_op_simple_nary(eigen_t<T>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, size_t /* row number - not used */)
    {
        auto r = m.col(parentIndex);
        size_t arity = nodes[parentIndex].Arity;

        auto i = parentIndex - 1;

        Function<Type> f{};

        if (arity == 1) {
            f(r, m.col(i));
        } else {
            r = m.col(i);

            for (size_t k = 1; k < arity; ++k) {
                i -= nodes[i].Length + 1;
                f(r, m.col(i));
            }
        }
    }

    struct noop {
        template<typename... Args>
        void operator()(Args&&...) {}
    };

    template<typename X, typename Tuple>
    class tuple_index;

    template<typename X, typename... T>
    class tuple_index<X, std::tuple<T...>> {
        template<std::size_t... idx>
        static constexpr ssize_t find_idx(std::index_sequence<idx...>)
        {
            return -1 + ((std::is_same<X, T>::value ? idx + 1 : 0) + ...);
        }

    public:
        static constexpr ssize_t value = find_idx(std::index_sequence_for<T...>{});
    };

    template<typename T>
    using Callable = typename std::function<void(detail::eigen_t<T>&, Operon::Vector<Node> const&, size_t, size_t)>;

    template<NodeType Type, typename T>
    static constexpr Callable<T> MakeCall()
    {
        if constexpr (Type < NodeType::Aq) { // nary: add, sub, mul, div
            return Callable<T>(detail::dispatch_op_nary<Type, T>);
        } else if constexpr (Type < NodeType::Abs) { // binary: aq, fmin, fmax, pow
            return Callable<T>(detail::dispatch_op_binary<Type, T>);
        } else if constexpr (Type < NodeType::Constant) { // unary: exp, log, ... square, dynamic
            return Callable<T>(detail::dispatch_op_unary<Type, T>);
        }
    }

    template<NodeType Type, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0, bool> = true>
    static constexpr auto MakeTuple()
    {
        return std::tuple(MakeCall<Type, Ts>()...);
    };

    template<typename F, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0 && (std::is_invocable_r_v<void, F, detail::eigen_t<Ts>&, Vector<Node> const&, size_t, size_t> && ...), bool> = true>
    static constexpr auto MakeTuple(F&& f)
    {
        return std::tuple(Callable<Ts>(std::forward<F&&>(f))...);
    }

    template<typename F, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0 && (std::is_invocable_r_v<void, F, detail::eigen_t<Ts>&, Vector<Node> const&, size_t, size_t> && ...), bool> = true>
    static constexpr auto MakeTuple(F const& f)
    {
        return std::tuple(Callable<Ts>(f)...);
    }

    template<NodeType Type>
    static constexpr auto MakeDefaultTuple()
    {
        return MakeTuple<Type, Operon::Scalar, Operon::Dual>();
    }

    template<typename F, std::enable_if_t<
        std::is_invocable_r_v<
            void, F, detail::eigen_t<Operon::Scalar>&, Operon::Vector<Node> const&, size_t, size_t
        > &&
        std::is_invocable_r_v<
            void, F, detail::eigen_t<Operon::Dual>&, Operon::Vector<Node> const&, size_t, size_t
        >, bool> = true>
    static constexpr auto MakeDefaultTuple(F&& f)
    {
        return MakeTuple<F, Operon::Scalar, Operon::Dual>(std::forward<F&&>(f));
    }

} // namespace detail

struct DispatchTable {
    template<typename T>
    using Callable = detail::Callable<T>;

    using Tuple    = std::tuple<Callable<Operon::Scalar>, Callable<Operon::Dual>>;
    using Map      = robin_hood::unordered_flat_map<Operon::Hash, Tuple>;
    using Pair     = robin_hood::pair<Operon::Hash, Tuple>;

    DispatchTable()
    {
        InitializeMap();
    }

    DispatchTable(DispatchTable const& other) : map(other.map) { }
    DispatchTable(DispatchTable &&other) : map(std::move(other.map)) { }

    void InitializeMap()
    {
        const auto hash = [](auto t) { return Node(t).HashValue; };

        map = Map{
            { hash(NodeType::Add), detail::MakeDefaultTuple<NodeType::Add>() },
            { hash(NodeType::Sub), detail::MakeDefaultTuple<NodeType::Sub>() },
            { hash(NodeType::Mul), detail::MakeDefaultTuple<NodeType::Mul>() },
            { hash(NodeType::Sub), detail::MakeDefaultTuple<NodeType::Sub>() },
            { hash(NodeType::Div), detail::MakeDefaultTuple<NodeType::Div>() },
            { hash(NodeType::Aq),  detail::MakeDefaultTuple<NodeType::Aq>() },
            { hash(NodeType::Fmax), detail::MakeDefaultTuple<NodeType::Fmax>() },
            { hash(NodeType::Fmin), detail::MakeDefaultTuple<NodeType::Fmin>() },
            { hash(NodeType::Pow), detail::MakeDefaultTuple<NodeType::Pow>() },
            { hash(NodeType::Abs), detail::MakeDefaultTuple<NodeType::Abs>() },
            { hash(NodeType::Acos), detail::MakeDefaultTuple<NodeType::Acos>() },
            { hash(NodeType::Asin), detail::MakeDefaultTuple<NodeType::Asin>() },
            { hash(NodeType::Atan), detail::MakeDefaultTuple<NodeType::Atan>() },
            { hash(NodeType::Cbrt), detail::MakeDefaultTuple<NodeType::Cbrt>() },
            { hash(NodeType::Ceil), detail::MakeDefaultTuple<NodeType::Ceil>() },
            { hash(NodeType::Cos), detail::MakeDefaultTuple<NodeType::Cos>() },
            { hash(NodeType::Cosh), detail::MakeDefaultTuple<NodeType::Sinh>() },
            { hash(NodeType::Exp), detail::MakeDefaultTuple<NodeType::Exp>() },
            { hash(NodeType::Floor), detail::MakeDefaultTuple<NodeType::Floor>() },
            { hash(NodeType::Log), detail::MakeDefaultTuple<NodeType::Log>() },
            { hash(NodeType::Log1p), detail::MakeDefaultTuple<NodeType::Log1p>() },
            { hash(NodeType::Sin), detail::MakeDefaultTuple<NodeType::Sin>() },
            { hash(NodeType::Sinh), detail::MakeDefaultTuple<NodeType::Cosh>() },
            { hash(NodeType::Sqrt), detail::MakeDefaultTuple<NodeType::Sqrt>() },
            { hash(NodeType::Square), detail::MakeDefaultTuple<NodeType::Square>() },
            { hash(NodeType::Tan), detail::MakeDefaultTuple<NodeType::Tan>() },
            { hash(NodeType::Tanh), detail::MakeDefaultTuple<NodeType::Tanh>() },
            /* constants and variables not needed here */
        };
    };

    template<typename T>
    inline Callable<T>& Get(Operon::Hash const h)
    {
        constexpr ssize_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map.find(h); it != map.end()) {
            return std::get<static_cast<size_t>(idx)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename T>
    inline Callable<T> const& Get(Operon::Hash const h) const
    {
        constexpr ssize_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map.find(h); it != map.end()) {
            return std::get<static_cast<size_t>(idx)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename F, std::enable_if_t<std::is_invocable_r_v<void, F, detail::eigen_t<Operon::Dual>&, Vector<Node> const&, size_t, size_t>, bool> = true>
    void RegisterCallable(Operon::Hash hash, F const& f) {
        map[hash] = detail::MakeTuple<F, Operon::Scalar, Operon::Dual>(f);
    }

private:
    Map map;
};

} // namespace Operon

#endif
