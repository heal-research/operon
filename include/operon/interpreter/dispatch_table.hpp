// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_EVAL_DETAIL
#define OPERON_EVAL_DETAIL

#include <Eigen/Dense>
#include <fmt/core.h>
#include <optional>
#include <cstddef>
#include <tuple>

#include "operon/core/node.hpp"
#include "operon/core/range.hpp"
#include "operon/core/types.hpp"
#include "derivatives.hpp"


namespace Operon {
// data types used by the dispatch table and the interpreter
struct Dispatch {

template<typename T>
//requires std::is_arithmetic_v<T>
static auto constexpr DefaultBatchSize{ 512UL / sizeof(T) };

template<typename T, std::size_t S>
using Callable = std::function<void(Operon::Vector<Node> const&, Backend::View<T, S>, size_t, Operon::Range)>;

template<typename T, std::size_t S>
using CallableDiff = std::function<void(Operon::Vector<Node> const&, Backend::View<T const, S>, Backend::View<T, S>, int, int)>;

// dispatching mechanism
// compared to the simple/naive way of evaluating n-ary symbols, this method has the following advantages:
// 1) improved performance: the naive method accumulates into the result for each argument, leading to unnecessary assignments
// 2) minimizing the number of intermediate steps which might improve floating point accuracy of some operations
//    if arity > 4, one accumulation is performed every 4 args
template<NodeType Type, typename T, std::size_t S>
requires Node::IsNary<Type>
static inline void NaryOp(Operon::Vector<Node> const& nodes, Backend::View<T, S> data, size_t parentIndex, Operon::Range /*unused*/)
{
    static_assert(Type < NodeType::Aq);
    const auto nextArg = [&](size_t i) { return i - (nodes[i].Length + 1); };
    auto arg1 = parentIndex - 1;
    bool continued = false;

    auto const call = [&](bool continued, int result, auto... args) {
        if (continued) { Func<T, Type, true , S>{}(data, result, args...); }
        else           { Func<T, Type, false, S>{}(data, result, args...); }
    };

    int arity = nodes[parentIndex].Arity;
    while (arity > 0) {
        switch (arity) {
        case 1: {
            call(continued, parentIndex, arg1);
            arity = 0;
            break;
        }
        case 2: {
            auto arg2 = nextArg(arg1);
            call(continued, parentIndex, arg1, arg2);
            arity = 0;
            break;
        }
        case 3: {
            auto arg2 = nextArg(arg1);
            auto arg3 = nextArg(arg2);
            call(continued, parentIndex, arg1, arg2, arg3);
            arity = 0;
            break;
        }
        default: {
            auto arg2 = nextArg(arg1);
            auto arg3 = nextArg(arg2);
            auto arg4 = nextArg(arg3);
            call(continued, parentIndex, arg1, arg2, arg3, arg4);
            arity -= 4;
            arg1 = nextArg(arg4);
            break;
        }
        }
        continued = true;
    }
}

template<NodeType Type, typename T, std::size_t S>
requires Node::IsBinary<Type>
static inline void BinaryOp(Operon::Vector<Node> const& nodes, Backend::View<T, S> m, size_t i, Operon::Range /*unused*/)
{
    auto j = i - 1;
    auto k = j - nodes[j].Length - 1;
    Func<T, Type, false>{}(m, i, j, k);
}

template<NodeType Type, typename T, std::size_t S>
requires Node::IsUnary<Type>
static inline void UnaryOp(Operon::Vector<Node> const& /*unused*/, Backend::View<T, S> m, size_t i, Operon::Range /*unused*/)
{
    Func<T, Type, false>{}(m, i, i-1);
}

struct Noop {
    template<typename... Args>
    void operator()(Args&&... /*unused*/) {}
};

template<NodeType Type, typename T, std::size_t S>
static inline void DiffOp(Operon::Vector<Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T, S> trace, int i, int j) {
   Diff<T, Type, S>{}(nodes, primal, trace, i, j);
}

template<NodeType Type, typename T, std::size_t S>
static constexpr auto MakeFunctionCall() -> Dispatch::Callable<T, S>
{
    if constexpr (Node::IsNary<Type>) {
        return Callable<T, S>{NaryOp<Type, T, S>};
    } else if constexpr (Node::IsBinary<Type>) {
        return Callable<T, S>{BinaryOp<Type, T, S>};
    } else if constexpr (Node::IsUnary<Type>) {
        return Callable<T, S>{UnaryOp<Type, T, S>};
    }
}

template<NodeType Type, typename T, std::size_t S>
static constexpr auto MakeDiffCall() -> Dispatch::CallableDiff<T, S>
{
    // this constexpr if here returns NOOP in case of non-arithmetic types (duals)
    if constexpr (std::is_arithmetic_v<T>) {
        return CallableDiff<T, S>{DiffOp<Type, T, S>};
    } else {
        return Dispatch::Noop{};
    }
}
}; // struct Dispatch

namespace detail {
    // return the index of type T in Tuple
    template<typename T, typename... Ts>
    static auto constexpr TypeIndexImpl() {
        std::size_t i{0};
        for (bool x : { std::is_same_v<T, Ts>... }) {
            if (x) { break; }
            ++i;
        }
        return i;
    }

    template<typename T>
    concept ExtentsLike = requires {
        { T::size() };
        { std::is_array_v<T> };
    };

    template<typename Tup, std::size_t N>
    struct ExtractTypes {
        static auto constexpr Extract() {
            return []<auto... Seq>(std::index_sequence<Seq...>){
                return std::make_tuple(std::tuple_element_t<Seq, Tup>{}...);
            }(std::make_index_sequence<N>{});
        }

        using Type = decltype(Extract());
    };
} // namespace detail

template<typename... Ts>
struct DispatchTable {

private:
    using Tup = std::tuple<Ts...>; // make the type parameters into a tuple
    static auto constexpr N = std::tuple_size_v<Tup>;

    // retrieve the last type in the template parameter pack
    using Lst = std::tuple_element_t<N-1, Tup>;
    using Typ = std::conditional_t<detail::ExtentsLike<Lst>, typename detail::ExtractTypes<Tup, N-1>::Type, Tup>;
    using Ext = std::conditional_t<detail::ExtentsLike<Lst>, Lst, std::index_sequence<Dispatch::DefaultBatchSize<Ts>...>>;

    template<typename T, auto SZ = std::tuple_size_v<Typ>>
    requires (detail::TypeIndexImpl<T, Ts...>() < SZ)
    static auto constexpr TypeIndex = detail::TypeIndexImpl<T, Ts...>();

    static auto constexpr Sizes = []<auto... Idx>(std::integer_sequence<typename Ext::value_type, Idx...>) {
        return std::array<typename Ext::value_type, Ext::size()>{ Idx... };
    }(Ext{});

public:
    using SupportedTypes = Tup;

    template<typename T>
    static constexpr typename Ext::value_type BatchSize = Sizes[TypeIndex<T>];

    template<typename T>
    using Backend = Backend::View<T, BatchSize<T>>;

    template<typename T>
    using Callable = Dispatch::Callable<T, BatchSize<T>>;

    template<typename T>
    using CallableDiff = Dispatch::CallableDiff<T, BatchSize<T>>;

private:
    template<NodeType Type, typename T>
    static constexpr auto MakeFunction() {
        return Dispatch::MakeFunctionCall<Type, T, BatchSize<T>>();
    }

    template<NodeType Type, typename T>
    static constexpr auto MakeDerivative() {
        return Dispatch::MakeDiffCall<Type, T, BatchSize<T>>();
    }

    template<NodeType Type>
    static constexpr auto MakeTuple()
    {
        return []<auto... Idx>(std::index_sequence<Idx...>){
            return std::make_tuple(
                std::make_tuple(MakeFunction<Type, std::tuple_element_t<Idx, Tup>>()...),
                std::make_tuple(MakeDerivative<Type, std::tuple_element_t<Idx, Tup>>()...)
            );
        }(std::index_sequence_for<Typ>{});
    }

    using TFun = decltype([]<auto... Idx>(std::index_sequence<Idx...>){
                    return std::make_tuple(Callable<std::tuple_element_t<Idx, Typ>>{}...);
                 }(std::index_sequence_for<Typ>{}));

    using TDif = decltype([]<auto... Idx>(std::index_sequence<Idx...>){
                    return std::make_tuple(CallableDiff<std::tuple_element_t<Idx, Typ>>{}...);
                 }(std::index_sequence_for<Typ>{}));

    using Tuple = std::tuple<TFun, TDif>;
    using Map   = Operon::Map<Operon::Hash, Tuple>;

    Map map_;

public:
    DispatchTable()
    {
        auto constexpr f = [](auto i) { return static_cast<NodeType>(1U << i); };
        [&]<auto ...I>(std::index_sequence<I...>){
            (map_.insert({ Node(f(I)).HashValue, MakeTuple<f(I)>() }), ...);
        }(std::make_index_sequence<NodeTypes::Count-3>{});
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

    template<typename U>
    static constexpr auto SupportsType = TypeIndex<U> < std::tuple_size_v<Typ>;

    explicit DispatchTable(Map const& map) : map_(map) { }
    explicit DispatchTable(Map&& map) : map_(std::move(map)) { }
    explicit DispatchTable(std::unordered_map<Operon::Hash, Tuple> const& map) : map_(map.begin(), map.end()) { }

    DispatchTable(DispatchTable const& other) : map_(other.map_) { }
    DispatchTable(DispatchTable &&other) noexcept : map_(std::move(other.map_)) { }

    auto GetMap() -> Map& { return map_; }
    auto GetMap() const -> Map const& { return map_; }

    template<typename T>
    inline auto GetFunction(Operon::Hash const h) -> Callable<T>&
    {
        return const_cast<Callable<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->GetFunction(h)); // NOLINT
    }

    template<typename T>
    inline auto GetDerivative(Operon::Hash const h) -> CallableDiff<T>&
    {
        return const_cast<CallableDiff<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->GetDerivative(h)); // NOLINT
    }

    template<typename T>
    [[nodiscard]] inline auto GetFunction(Operon::Hash const h) const -> Callable<T> const&
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(TypeIndex<T>)>(std::get<0>(it->second));
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename T>
    [[nodiscard]] inline auto GetDerivative(Operon::Hash const h) const -> CallableDiff<T> const&
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(TypeIndex<T>)>(std::get<1>(it->second));
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename T>
    [[nodiscard]] inline auto Get(Operon::Hash const h) const -> std::tuple<Callable<T>, CallableDiff<T>>
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(TypeIndex<T>)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename F>
    void RegisterCallable(Operon::Hash hash, F&& f) {
        map_[hash] = MakeTuple<F, Ts...>(std::forward<F&&>(f), Dispatch::Noop{});
    }

    template<typename F, typename DF>
    void RegisterCallable(Operon::Hash hash, F&& f, DF&& df) {
        map_[hash] = MakeTuple<F, Ts...>(std::forward<F&&>(f), std::forward<DF&&>(df));
    }

    template<typename T>
    [[nodiscard]] inline auto TryGetFunction(Operon::Hash const h) const noexcept -> std::optional<Callable<T>>
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::optional{ std::get<TypeIndex<T>>(std::get<0>(it->second)) };
        }
        return {};
    }

    template<typename T>
    [[nodiscard]] inline auto TryGetDerivative(Operon::Hash const h) const noexcept -> std::optional<CallableDiff<T>>
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::optional{ std::get<TypeIndex<T>>(std::get<1>(it->second)) };
        }
        return {};
    }

    [[nodiscard]] auto Contains(Operon::Hash hash) const noexcept -> bool { return map_.contains(hash); }
}; // struct DispatchTable

using DefaultDispatch = DispatchTable<Operon::Scalar>;
} // namespace Operon

#endif
