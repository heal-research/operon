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

namespace concepts {
    template<typename Derived>
    concept MatrixBase = std::is_base_of_v<Eigen::MatrixBase<Derived>, Derived>;

    template<typename Derived>
    concept ArrayBase = std::is_base_of_v<Eigen::ArrayBase<Derived>, Derived>;
} // namespace concepts

// forward declarations
struct Dataset;
struct Tree;

// data types used by the dispatch table and the interpreter
struct Dispatch {

template<typename T>
//requires std::is_arithmetic_v<T>
static auto constexpr DefaultBatchSize{ 512UL / sizeof(T) };

template<concepts::ArrayBase A>
using Callable = std::function<void(Operon::Vector<Node> const&, A&, size_t, Operon::Range)>;

template<concepts::ArrayBase A>
using CallableDiff = std::function<void(Operon::Vector<Node> const&, A const&, A&, int, int)>;

// dispatching mechanism
// compared to the simple/naive way of evaluating n-ary symbols, this method has the following advantages:
// 1) improved performance: the naive method accumulates into the result for each argument, leading to unnecessary assignments
// 2) minimizing the number of intermediate steps which might improve floating point accuracy of some operations
//    if arity > 4, one accumulation is performed every 4 args
template<NodeType Type, concepts::ArrayBase A>
requires Node::IsNary<Type>
static inline void NaryOp(Operon::Vector<Node> const& nodes, A& m, size_t parentIndex, Operon::Range /*unused*/)
{
    using R = Eigen::Ref<decltype(m.col(0))>;
    static_assert(Type < NodeType::Aq);
    auto result = R(m.col(parentIndex));
    const auto f = [](bool cont, decltype(result) res, auto&&... args) {
        if (cont) { res = Func<Type, false>{}(res, Func<Type, true>{}(args...)); }
        else      { res = Func<Type, false>{}(args...); }
    };
    const auto nextArg = [&](size_t i) { return i - (nodes[i].Length + 1); };
    auto arg1 = parentIndex - 1;
    bool continued = false;

    int arity = nodes[parentIndex].Arity;
    while (arity > 0) {
        switch (arity) {
        case 1: {
            f(continued, result, R(m.col(arg1)));
            arity = 0;
            break;
        }
        case 2: {
            auto arg2 = nextArg(arg1);
            f(continued, result, R(m.col(arg1)), R(m.col(arg2)));
            arity = 0;
            break;
        }
        case 3: {
            auto arg2 = nextArg(arg1);
            auto arg3 = nextArg(arg2);
            f(continued, result, R(m.col(arg1)), R(m.col(arg2)), R(m.col(arg3)));
            arity = 0;
            break;
        }
        default: {
            auto arg2 = nextArg(arg1);
            auto arg3 = nextArg(arg2);
            auto arg4 = nextArg(arg3);
            f(continued, result, R(m.col(arg1)), R(m.col(arg2)), R(m.col(arg3)), R(m.col(arg4)));
            arity -= 4;
            arg1 = nextArg(arg4);
            break;
        }
        }
        continued = true;
    }
}

template<NodeType Type, concepts::ArrayBase A>
requires Node::IsBinary<Type>
static inline void BinaryOp(Operon::Vector<Node> const& nodes, A& m, size_t i, Operon::Range /*unused*/)
{
    using R = Eigen::Ref<decltype(m.col(0))>;
    auto j = i - 1;
    auto k = j - nodes[j].Length - 1;
    m.col(i) = Func<Type, false>{}(R(m.col(j)), R(m.col(k)));
}

template<NodeType Type, concepts::ArrayBase A>
requires Node::IsUnary<Type>
static inline void UnaryOp(Operon::Vector<Node> const& /*unused*/, A& m, size_t i, Operon::Range /*unused*/)
{
    using R = Eigen::Ref<decltype(m.col(0))>;
    m.col(i) = Func<Type, false>{}(R(m.col(i-1)));
}

struct Noop {
    template<typename... Args>
    void operator()(Args&&... /*unused*/) {}
};

template<NodeType Type, concepts::ArrayBase A>
static inline void DiffOp(Operon::Vector<Node> const& nodes, A const& primal, A& trace, int i, int j) {
    Diff<Type>{}(nodes, primal, trace, i, j);
};

template<NodeType Type, concepts::ArrayBase A>
static constexpr auto MakeFunctionCall() -> Dispatch::Callable<A>
{
    if constexpr (Node::IsNary<Type>) {
        return Callable<A>{NaryOp<Type, A>};
    } else if constexpr (Node::IsBinary<Type>) {
        return Callable<A>{BinaryOp<Type, A>};
    } else if constexpr (Node::IsUnary<Type>) {
        return Callable<A>{UnaryOp<Type, A>};
    }
}

template<NodeType Type, concepts::ArrayBase A>
static constexpr auto MakeDiffCall() -> Dispatch::CallableDiff<A>
{
    // this constexpr if here returns NOOP in case of non-arithmetic types (duals)
    if constexpr (std::is_arithmetic_v<typename A::Scalar>) {
        return CallableDiff<A>{DiffOp<Type, A>};
    } else {
        return Dispatch::Noop{};
    }
}
};

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
} // namespace detail

template<typename... Ts>
struct DispatchTable {

private:
    using Tup = std::tuple<Ts...>; // make the type parameters into a tuple

    // retrieve the last type in the template parameter pack
    using Lst = std::tuple_element_t<sizeof...(Ts)-1, Tup>;

    using Typ = std::conditional_t<detail::ExtentsLike<Lst>, decltype([]<auto... Idx>(std::index_sequence<Idx...>){
                    return std::make_tuple(std::tuple_element_t<Idx, Tup>{}...);
                }(std::make_index_sequence<sizeof...(Ts)-1>{})), Tup>;

    using Ext = std::conditional_t<detail::ExtentsLike<Lst>, Lst, std::index_sequence<Dispatch::DefaultBatchSize<Ts>...>>;

    template<typename T, auto SZ = std::tuple_size_v<Typ>>
    requires (detail::TypeIndexImpl<T, Ts...>() < SZ)
    static auto constexpr TypeIndex = detail::TypeIndexImpl<T, Ts...>();

    static auto constexpr Sizes = []<auto... Idx>(std::integer_sequence<typename Ext::value_type, Idx...>) {
        return std::array<typename Ext::value_type, Ext::size()>{ Idx... };
    }(Ext{});

public:
    template<typename T>
    static constexpr typename Ext::value_type BatchSize = Sizes[TypeIndex<T>];

    template<typename T>
    using Array = Eigen::Array<T, BatchSize<T>, -1>;

    template<typename T>
    using Callable = Dispatch::Callable<Array<T>>;

private:
    template<NodeType Type>
    static constexpr auto MakeTuple()
    {
        return []<auto... Idx>(std::index_sequence<Idx...>){
            return std::make_tuple(
                std::make_tuple(Dispatch::MakeFunctionCall<Type, Array<std::tuple_element_t<Idx, Tup>>>()...),
                std::make_tuple(Dispatch::MakeDiffCall<Type, Array<std::tuple_element_t<Idx, Tup>>>()...)
            );
        }(std::index_sequence_for<Typ>{});
    };

    using TFun = decltype([]<auto... Idx>(std::index_sequence<Idx...>){
                    return std::make_tuple(Dispatch::Callable<Array<std::tuple_element_t<Idx, Typ>>>{}...);
                 }(std::index_sequence_for<Typ>{}));

    using TDif = decltype([]<auto... Idx>(std::index_sequence<Idx...>){
                    return std::make_tuple(Dispatch::CallableDiff<Array<std::tuple_element_t<Idx, Typ>>>{}...);
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
    inline auto GetFunction(Operon::Hash const h) -> Dispatch::Callable<Array<T>>&
    {
        return const_cast<Dispatch::Callable<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->GetFunction(h)); // NOLINT
    }

    template<typename T>
    inline auto GetDerivative(Operon::Hash const h) -> Dispatch::CallableDiff<Array<T>>&
    {
        return const_cast<Dispatch::CallableDiff<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->GetDerivative(h)); // NOLINT
    }

    template<typename T>
    [[nodiscard]] inline auto GetFunction(Operon::Hash const h) const -> Dispatch::Callable<Array<T>> const&
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(TypeIndex<T>)>(std::get<0>(it->second));
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename T>
    [[nodiscard]] inline auto GetDerivative(Operon::Hash const h) const -> Dispatch::CallableDiff<Array<T>> const&
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(TypeIndex<T>)>(std::get<1>(it->second));
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename T, typename A = Array<T>>
    [[nodiscard]] inline auto Get(Operon::Hash const h) const -> std::tuple<Dispatch::Callable<A>, Dispatch::CallableDiff<A>>
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
    [[nodiscard]] inline auto TryGetFunction(Operon::Hash const h) const noexcept -> std::optional<Dispatch::Callable<Array<T>>>
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::optional{ std::get<TypeIndex<T>>(std::get<0>(it->second)) };
        }
        return {};
    }

    template<typename T>
    [[nodiscard]] inline auto TryGetDerivative(Operon::Hash const h) const noexcept -> std::optional<Dispatch::CallableDiff<Array<T>>>
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::optional{ std::get<TypeIndex<T>>(std::get<1>(it->second)) };
        }
        return {};
    }

    [[nodiscard]] auto Contains(Operon::Hash hash) const noexcept -> bool { return map_.contains(hash); }
}; // struct DispatchTable

using DefaultDispatch = DispatchTable<Operon::Scalar, Operon::Seq<std::size_t, Dispatch::DefaultBatchSize<Operon::Scalar>>>;
} // namespace Operon

#endif
