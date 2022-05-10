// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_EVAL_DETAIL
#define OPERON_EVAL_DETAIL

#include <Eigen/Dense>
#include <fmt/core.h>
#include <optional>
#include <robin_hood.h>
#include <cstddef>
#include <tuple>

#include "operon/core/node.hpp"
#include "operon/core/range.hpp"
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
    using Array = Eigen::Array<T, BatchSize<T>::Value, 1>;

    template<typename T>
    using Ref = Eigen::Ref<Array<T>>;

    // dispatching mechanism
    // compared to the simple/naive way of evaluating n-ary symbols, this method has the following advantages:
    // 1) improved performance: the naive method accumulates into the result for each argument, leading to unnecessary assignments
    // 2) minimizing the number of intermediate steps which might improve floating point accuracy of some operations
    //    if arity > 4, one accumulation is performed every 4 args
    template<NodeType Type, typename T>
    inline void DispatchOpNary(Operon::Vector<Array<T>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, Operon::Range /* not used here - provided for dynamic symbols */)
    {
        static_assert(Type < NodeType::Aq);
        auto result = Ref<T>(m[parentIndex]);
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

        using R = Ref<T>;

        int arity = nodes[parentIndex].Arity;
        while (arity > 0) {
            switch (arity) {
            case 1: {
                f(continued, result, R(m[arg1]));
                arity = 0;
                break;
            }
            case 2: {
                auto arg2 = nextArg(arg1);
                f(continued, result, R(m[arg1]), R(m[arg2]));
                arity = 0;
                break;
            }
            case 3: {
                auto arg2 = nextArg(arg1);
                auto arg3 = nextArg(arg2);
                f(continued, result, R(m[arg1]), R(m[arg2]), R(m[arg3]));
                arity = 0;
                break;
            }
            default: {
                auto arg2 = nextArg(arg1);
                auto arg3 = nextArg(arg2);
                auto arg4 = nextArg(arg3);
                f(continued, result, R(m[arg1]), R(m[arg2]), R(m[arg3]), R(m[arg4]));
                arity -= 4;
                arg1 = nextArg(arg4);
                break;
            }
            }
            continued = true;
        }
    }

    template<NodeType Type, typename T>
    inline void DispatchOpUnary(Operon::Vector<Array<T>>& m, Operon::Vector<Node> const& /*unused*/, size_t i, Operon::Range /* not used here - provided for dynamic symbols */)
    {
        static_assert(Type < NodeType::Dynamic && Type > NodeType::Pow);
        Function<Type>{}(Ref<T>(m[i]), Ref<T>(m[i-1]));
    }

    template<NodeType Type, typename T>
    inline void DispatchOpBinary(Operon::Vector<Array<T>>& m, Operon::Vector<Node> const& nodes, size_t i, Operon::Range /* not used here - provided for dynamic symbols */)
    {
        static_assert(Type < NodeType::Abs && Type > NodeType::Fmax);
        auto j = i - 1;
        auto k = j - nodes[j].Length - 1;
        Function<Type>{}(Ref<T>(m[i]), Ref<T>(m[j]), Ref<T>(m[k]));
    }

    template<NodeType Type, typename T>
    inline void DispatchOpSimpleUnaryOrBinary(Operon::Vector<Array<T>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, Operon::Range /* not used here - provided for dynamic symbols */)
    {
        auto r = Ref<T>(m[parentIndex]);
        size_t i = parentIndex - 1;
        size_t arity = nodes[parentIndex].Arity;

        Function<Type> f{};

        if (arity == 1) {
            f(r, Ref<T>(m[i]));
        } else {
            auto j = i - (nodes[i].Length + 1);
            f(r, Ref<T>(m[j]));
        }
    }

    template<NodeType Type, typename T>
    inline void DispatchOpSimpleNary(Operon::Vector<Array<T>>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, Operon::Range /* not used here - provided for dynamic symbols */)
    {
        auto r = Ref<T>(m[parentIndex]);
        size_t arity = nodes[parentIndex].Arity;

        auto i = parentIndex - 1;

        Function<Type> f{};

        if (arity == 1) {
            f(r, Ref<T>(m[i]));
        } else {
            r = m[i];

            for (size_t k = 1; k < arity; ++k) {
                i -= nodes[i].Length + 1;
                f(r, Ref<T>(m[i]));
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
    using Callable = typename std::function<void(Operon::Vector<Array<T>>&, Operon::Vector<Node> const&, size_t, Operon::Range)>;

    template<NodeType Type, typename T>
    static constexpr auto MakeCall() -> Callable<T>
    {
        if constexpr (Type < NodeType::Aq) { // nary: add, sub, mul, div, fmin, fmax
            return Callable<T>(detail::DispatchOpNary<Type, T>);
        } else if constexpr (Type < NodeType::Abs) { // binary: aq, pow
            return Callable<T>(detail::DispatchOpBinary<Type, T>);
        } else if constexpr (Type < NodeType::Dynamic) { // unary: exp, log, sin, cos, tan, tanh, sqrt, cbrt, square
            return Callable<T>(detail::DispatchOpUnary<Type, T>);
        }
    }

    template<NodeType Type, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0, bool> = true>
    static constexpr auto MakeTuple()
    {
        return std::make_tuple(MakeCall<Type, Ts>()...);
    };

    template<typename F, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0 && (std::is_invocable_r_v<void, F, detail::Array<Ts>&, Vector<Node> const&, size_t, Operon::Range> && ...), bool> = true>
    static constexpr auto MakeTuple(F&& f)
    {
        return std::make_tuple(Callable<Ts>(std::forward<F&&>(f))...);
    }
} // namespace detail

template<typename... Ts>
struct DispatchTable {
    template<typename T>
    using Callable = detail::Callable<T>;

    using Tuple    = std::tuple<Callable<Ts>...>;
    using Map      = robin_hood::unordered_flat_map<Operon::Hash, Tuple>;

private:
    Map map_;

    template<std::size_t... Is>
    void InitMap(std::index_sequence<Is...> /*unused*/)
    {
        auto f = [](auto i) { return static_cast<NodeType>(1U << i); };
        (map_.insert({ Node(f(Is)).HashValue, detail::MakeTuple<f(Is), Ts...>() }), ...);
    }

public:
    DispatchTable()
    {
        InitMap(std::make_index_sequence<NodeTypes::Count-3>{}); // exclude constant, variable, dynamic
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
        return const_cast<Callable<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->Get(h)); // NOLINT
    }

    template<typename T>
    [[nodiscard]] inline auto Get(Operon::Hash const h) const -> Callable<T> const&
    {
        constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(idx)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename F>
    void RegisterCallable(Operon::Hash hash, F&& f) {
        map_[hash] = detail::MakeTuple<F, Ts...>(std::forward<F&&>(f));
    }

    template<typename T>
    [[nodiscard]] inline auto TryGet(Operon::Hash const h) const noexcept -> std::optional<Callable<T>>
    {
        constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return { std::get<static_cast<size_t>(idx)>(it->second) };
        }
        return {};
    }

    [[nodiscard]] auto Contains(Operon::Hash hash) const noexcept -> bool { return map_.contains(hash); }
};

} // namespace Operon

#endif
