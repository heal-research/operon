// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_EVAL_DETAIL
#define OPERON_EVAL_DETAIL

#include <Eigen/Dense>
#include <fmt/core.h>
#include <robin_hood.h>
#include <cstddef>
#include <tuple>

#include "operon/core/node.hpp"
#include "operon/core/types.hpp"
#include "functions.hpp"

namespace Operon {

namespace detail {
    // this should be good enough - tests show 512 is about optimal
    template<typename T>
    struct BatchSize {
        static const size_t Value = 512 / sizeof(T);
    };

    template<typename T>
    using Array = typename Eigen::Array<T, BatchSize<T>::Value, Eigen::Dynamic, Eigen::ColMajor>;

    template<typename T>
    using EigenRef = Eigen::Ref<Array<T>, Eigen::Unaligned, Eigen::Stride<BatchSize<T>::Value, 1>>;

    // dispatching mechanism
    // compared to the simple/naive way of evaluating n-ary symbols, this method has the following advantages:
    // 1) improved performance: the naive method accumulates into the result for each argument, leading to unnecessary assignments
    // 2) minimizing the number of intermediate steps which might improve floating point accuracy of some operations
    //    if arity > 4, one accumulation is performed every 4 args
    template<NodeType Type, typename T>
    inline void DispatchOpNary(Array<T>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, size_t /* row number - not used */)
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
                auto arg2 = nextArg(arg1);
                auto arg3 = nextArg(arg2);
                f(continued, result, m.col(arg1), m.col(arg2), m.col(arg3));
                arity = 0;
                break;
            }
            default: {
                auto arg2 = nextArg(arg1);
                auto arg3 = nextArg(arg2);
                auto arg4 = nextArg(arg3);
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
    inline void DispatchOpUnary(Array<T>& m, Operon::Vector<Node> const& /*unused*/, size_t i, size_t /* row number - not used */)
    {
        static_assert(Type < NodeType::Constant && Type > NodeType::Pow);
        Function<Type>{}(m.col(i), m.col(i - 1));
    }

    template<NodeType Type, typename T>
    inline void DispatchOpBinary(Array<T>& m, Operon::Vector<Node> const& nodes, size_t i, size_t /* row number - not used */)
    {
        static_assert(Type < NodeType::Log && Type > NodeType::Div);
        auto j = i - 1;
        auto k = j - nodes[j].Length - 1;
        Function<Type>{}(m.col(i), m.col(j), m.col(k));
    }

    template<NodeType Type, typename T>
    inline void DispatchOpSimpleUnaryOrBinary(Array<T>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, size_t /* row number - not used */)
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
    inline void DispatchOpSimpleNary(Array<T>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, size_t /* row number - not used */)
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

    struct Noop {
        template<typename... Args>
        void operator()(Args&&... /*unused*/) {}
    };

    template<typename X, typename Tuple>
    class tuple_index;

    template<typename X, typename... T>
    class tuple_index<X, std::tuple<T...>> {
        template<std::size_t... Idx>
        static constexpr auto FindIdx(std::index_sequence<Idx...> /*unused*/) -> int64_t
        {
            return -1 + ((std::is_same<X, T>::value ? Idx + 1 : 0) + ...);
        }

    public:
        static constexpr int64_t value = FindIdx(std::index_sequence_for<T...>{});
    };

    template<typename T>
    using Callable = typename std::function<void(detail::Array<T>&, Operon::Vector<Node> const&, size_t, size_t)>;

    template<NodeType Type, typename T>
    static constexpr auto MakeCall() -> Callable<T>
    {
        if constexpr (Type < NodeType::Aq) { // nary: add, sub, mul, div
            return Callable<T>(detail::DispatchOpNary<Type, T>);
        } else if constexpr (Type < NodeType::Log) { // binary: aq, pow
            return Callable<T>(detail::DispatchOpBinary<Type, T>);
        } else if constexpr (Type < NodeType::Constant) { // unary: exp, log, sin, cos, tan, tanh, sqrt, cbrt, square, dynamic
            return Callable<T>(detail::DispatchOpUnary<Type, T>);
        }
    }

    template<NodeType Type, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0, bool> = true>
    static constexpr auto MakeTuple()
    {
        return std::make_tuple(MakeCall<Type, Ts>()...);
    };

    template<typename F, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0 && (std::is_invocable_r_v<void, F, detail::Array<Ts>&, Vector<Node> const&, size_t, size_t> && ...), bool> = true>
    static constexpr auto MakeTuple(F&& f)
    {
        return std::make_tuple(Callable<Ts>(std::forward<F&&>(f))...);
    }

    template<typename F, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0 && (std::is_invocable_r_v<void, F, detail::Array<Ts>&, Vector<Node> const&, size_t, size_t> && ...), bool> = true>
    static constexpr auto MakeTuple(F const& f)
    {
        return std::make_tuple(Callable<Ts>(f)...);
    }
} // namespace detail

template<typename... Ts>
struct DispatchTable {
    template<typename T>
    using Callable = detail::Callable<T>;

    using Tuple    = std::tuple<Callable<Ts>...>;
    using Map      = robin_hood::unordered_flat_map<Operon::Hash, Tuple>;
    using Pair     = robin_hood::pair<Operon::Hash, Tuple>;

private:
    Map map_;

    template<size_t I>
    constexpr void InsertType()
    {
        constexpr auto T = static_cast<NodeType>(1U << I);
        map_.insert({ Node(T).HashValue, detail::MakeTuple<T, Ts...>() });
    }

    template<std::size_t... Is>
    void InitMap(std::index_sequence<Is...> /*unused*/)
    {
        (InsertType<Is>(), ...);
    }

public:
    DispatchTable()
    {
        InitMap(std::make_index_sequence<NodeTypes::Count-2>{});
    }

    ~DispatchTable() = default;

    auto operator=(DispatchTable const& other) -> DispatchTable& {
        if (this != &other) {
            map_ = other.map_;
        }
        return *this;
    }

    auto operator=(DispatchTable&& other) noexcept -> DispatchTable& {
        map_ = std::move(other.map_);
        return *this;
    }

    DispatchTable(DispatchTable const& other) : map_(other.map_) { }
    DispatchTable(DispatchTable &&other) noexcept : map_(std::move(other.map_)) { }

    template<typename T>
    inline auto Get(Operon::Hash const h) -> Callable<T>&
    {
        constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(idx)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename T>
    inline auto Get(Operon::Hash const h) const -> Callable<T> const&
    {
        constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(idx)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename F>
    void RegisterCallable(Operon::Hash hash, F const& f) {
        map_[hash] = detail::MakeTuple<F, Ts...>(f);
    }
};

} // namespace Operon

#endif
