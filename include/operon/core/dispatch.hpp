// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_EVAL_DETAIL
#define OPERON_EVAL_DETAIL

#include <Eigen/Dense>
#include <fmt/core.h>
#include <cstddef>
#include <tuple>

#include "concepts.hpp"
#include "node.hpp"
#include "range.hpp"
#include "types.hpp"
#include "aligned_allocator.hpp"

namespace Operon {

namespace Backend {
template<typename T>
static auto constexpr BatchSize = 512UL / sizeof(T);

static auto constexpr DefaultAlignment = 32UL;

template<typename T, std::size_t S = BatchSize<T>>
using View = Operon::MDSpan<T, std::extents<int, S, std::dynamic_extent>, std::layout_left>;

template<typename T, std::size_t S = BatchSize<T>>
using Buffer = Operon::MDArray<T, std::extents<int, S, std::dynamic_extent>, std::layout_left, std::vector<T, AlignedAllocator<T, DefaultAlignment>>>;

template<typename T, std::size_t S>
auto Ptr(View<T, S> view, std::integral auto col) -> Backend::View<T, S>::element_type* {
    return view.data_handle() + (col * S);
}

// utility
template<typename T, std::size_t S>
auto Fill(Backend::View<T, S> view, int idx, T value) {
    auto* p = view.data_handle() + (idx * S);
    std::fill_n(p, S, value);
};
} // namespace Backend

// detect missing specializations for functions
template<typename T, Operon::NodeType N = Operon::NodeTypes::NoType, bool C = false, std::size_t S = Backend::BatchSize<T>>
struct Func {
    auto operator()(Backend::View<T, S> /*primal*/, std::integral auto /*node index*/, std::integral auto... /*child indices*/) {
        throw std::runtime_error(fmt::format("backend error: missing specialization for function: {}\n", Operon::Node{N}.Name()));
    }
};

// detect missing specializations for function derivatives
template<typename T, Operon::NodeType N  = Operon::NodeTypes::NoType, std::size_t S = Backend::BatchSize<T>>
struct Diff {
    auto operator()(std::vector<Operon::Node> const& /*nodes*/, Backend::View<T const, S> /*primal*/, Backend::View<T> /*trace*/, std::integral auto /*node index*/, std::integral auto /*partial index*/) {
        throw std::runtime_error(fmt::format("backend error: missing specialization for derivative: {}\n", Operon::Node{N}.Name()));
    }
};

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
static void NaryOp(Operon::Vector<Node> const& nodes, Backend::View<T, S> data, size_t parentIndex, Operon::Range /*unused*/)
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
static void BinaryOp(Operon::Vector<Node> const& nodes, Backend::View<T, S> m, size_t i, Operon::Range /*unused*/)
{
    auto j = i - 1;
    auto k = j - nodes[j].Length - 1;
    Func<T, Type, false>{}(m, i, j, k);
}

template<NodeType Type, typename T, std::size_t S>
requires Node::IsUnary<Type>
static void UnaryOp(Operon::Vector<Node> const& /*unused*/, Backend::View<T, S> m, size_t i, Operon::Range /*unused*/)
{
    Func<T, Type, false>{}(m, i, i-1);
}

struct Noop {
    template<typename... Args>
    void operator()(Args&&... /*unused*/) {}
};

template<NodeType Type, typename T, std::size_t S>
static void DiffOp(Operon::Vector<Node> const& nodes, Backend::View<T const, S> primal, Backend::View<T, S> trace, int i, int j) {
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
        T::size() and std::is_array_v<T>;
    };

    template<typename T>
    struct DefaultIndex {
        using Type = std::size_t;
    };

    template<typename T>
    struct IntegerIndex {
        using Type = T::value_type;
    };
} // namespace detail

template<typename... Ts>
struct DispatchTable {

private:
    static constexpr auto S = sizeof...(Ts);
    using TypeHolder = std::tuple<Ts...>;
    using LastType   = std::tuple_element_t<S-1, TypeHolder>;
    using IndexType  = std::conditional_t<std::is_floating_point_v<LastType>, detail::DefaultIndex<LastType>, detail::IntegerIndex<LastType>>::Type;
    static constexpr auto N = detail::ExtentsLike<LastType> ? S-1 : S;

    static std::array constexpr Sizes = []<auto... Idx>(std::integer_sequence<IndexType, Idx...>){
        constexpr auto f = [&]<std::size_t TypeIdx>() {
            if constexpr (detail::ExtentsLike<LastType>) {
                constexpr auto t = []<auto... I>(std::integer_sequence<IndexType, I...>){ return std::tuple{I...}; }(LastType{});
                if constexpr (TypeIdx < LastType::size()) { return std::get<TypeIdx>(t); }
                else { return Dispatch::DefaultBatchSize<std::tuple_element_t<TypeIdx, TypeHolder>>; }
            } else {
                return Dispatch::DefaultBatchSize<std::tuple_element_t<TypeIdx, TypeHolder>>;
            }
        };
        return std::array{f.template operator()<Idx>()...};
    }(std::make_integer_sequence<IndexType, N>{});

    template<typename T>
    requires (detail::TypeIndexImpl<T, Ts...>() < N)
    static auto constexpr TypeIndex = detail::TypeIndexImpl<T, Ts...>();

public:
    using SupportedTypes = TypeHolder;

    template<typename T>
    requires Operon::Concepts::Arithmetic<T>
    static constexpr std::size_t BatchSize = Sizes[TypeIndex<T>];

    template<typename T>
    requires Operon::Concepts::Arithmetic<T>
    using Backend = Backend::View<T, BatchSize<T>>;

    template<typename T>
    requires Operon::Concepts::Arithmetic<T>
    using Callable = Dispatch::Callable<T, BatchSize<T>>;

    template<typename T>
    requires Operon::Concepts::Arithmetic<T>
    using CallableDiff = Dispatch::CallableDiff<T, BatchSize<T>>;

private:
    template<NodeType Type, typename T>
    requires Operon::Concepts::Arithmetic<T>
    static constexpr auto MakeFunction() {
        return Dispatch::MakeFunctionCall<Type, T, BatchSize<T>>();
    }

    template<NodeType Type, typename T>
    requires Operon::Concepts::Arithmetic<T>
    static constexpr auto MakeDerivative() {
        return Dispatch::MakeDiffCall<Type, T, BatchSize<T>>();
    }

    template<NodeType Type>
    static constexpr auto MakeTuple()
    {
        return []<auto... Idx>(std::index_sequence<Idx...>){
            return std::make_tuple(
                std::make_tuple(MakeFunction<Type, std::tuple_element_t<Idx, TypeHolder>>()...),
                std::make_tuple(MakeDerivative<Type, std::tuple_element_t<Idx, TypeHolder>>()...)
            );
        }(std::make_index_sequence<N>{});
    }

    using TFun = decltype([]<auto... Idx>(std::index_sequence<Idx...>){
                    return std::make_tuple(Callable<std::tuple_element_t<Idx, TypeHolder>>{}...);
                 }(std::make_index_sequence<N>{}));

    using TDer = decltype([]<auto... Idx>(std::index_sequence<Idx...>){
                    return std::make_tuple(CallableDiff<std::tuple_element_t<Idx, TypeHolder>>{}...);
                 }(std::make_index_sequence<N>{}));

    using Tuple = std::tuple<TFun, TDer>;
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
    static constexpr auto SupportsType = TypeIndex<U> < N;

    explicit DispatchTable(Map const& map) : map_(map) { }
    explicit DispatchTable(Map&& map) : map_(std::move(map)) { }
    explicit DispatchTable(std::unordered_map<Operon::Hash, Tuple> const& map) : map_(map.begin(), map.end()) { }

    DispatchTable(DispatchTable const& other) : map_(other.map_) { }
    DispatchTable(DispatchTable &&other) noexcept : map_(std::move(other.map_)) { }

    auto GetMap() -> Map& { return map_; }
    auto GetMap() const -> Map const& { return map_; }

    template<typename T>
    auto GetFunction(Operon::Hash const h) -> Callable<T>&
    {
        return const_cast<Callable<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->GetFunction(h)); // NOLINT
    }

    template<typename T>
    auto GetDerivative(Operon::Hash const h) -> CallableDiff<T>&
    {
        return const_cast<CallableDiff<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->GetDerivative(h)); // NOLINT
    }

    template<typename T>
    [[nodiscard]] auto GetFunction(Operon::Hash const h) const -> Callable<T> const&
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(TypeIndex<T>)>(std::get<0>(it->second));
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename T>
    [[nodiscard]] auto GetDerivative(Operon::Hash const h) const -> CallableDiff<T> const&
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(TypeIndex<T>)>(std::get<1>(it->second));
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename T>
    [[nodiscard]] auto Get(Operon::Hash const h) const -> std::tuple<Callable<T>, CallableDiff<T>>
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(TypeIndex<T>)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename F, typename DF = Dispatch::Noop>
    void RegisterCallable(Operon::Hash hash, F&& f, DF&& df = DF{}) {
        map_[hash] = std::make_tuple(std::forward<F&&>(f), std::forward<DF&&>(df));
    }

    template<typename T>
    [[nodiscard]] auto TryGetFunction(Operon::Hash const h) const noexcept -> std::optional<Callable<T>>
    {
        if (auto it = map_.find(h); it != map_.end()) {
            return std::optional{ std::get<TypeIndex<T>>(std::get<0>(it->second)) };
        }
        return {};
    }

    template<typename T>
    [[nodiscard]] auto TryGetDerivative(Operon::Hash const h) const noexcept -> std::optional<CallableDiff<T>>
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
