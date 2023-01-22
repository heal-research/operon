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
#include "functions.hpp"

namespace Operon {

// data types used by the dispatch table and the interpreter
struct Dispatch {

template<typename T>
static auto constexpr BatchSize{ 512UL / sizeof(T) };

template<typename T>
using Matrix = Eigen::Array<T, BatchSize<T>, -1>;

template<typename T>
using Ref = Eigen::Ref<typename Matrix<T>::ColXpr>;

template<typename T>
using Callable = std::function<void(Matrix<T>&, Operon::Vector<Node> const&, size_t, Operon::Range)>;

// dispatching mechanism
// compared to the simple/naive way of evaluating n-ary symbols, this method has the following advantages:
// 1) improved performance: the naive method accumulates into the result for each argument, leading to unnecessary assignments
// 2) minimizing the number of intermediate steps which might improve floating point accuracy of some operations
//    if arity > 4, one accumulation is performed every 4 args
template<NodeType Type, typename T>
requires Node::IsNary<Type>
static inline void NaryOp(Matrix<T>& m, Operon::Vector<Node> const& nodes, size_t parentIndex, Operon::Range /*unused*/)
{
    static_assert(Type < NodeType::Aq);
    auto result = Ref<T>(m.col(parentIndex));
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

    using R = Dispatch::Ref<T>;

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

template<NodeType Type, typename T>
requires Node::IsBinary<Type>
static inline void BinaryOp(Matrix<T>& m, Operon::Vector<Node> const& nodes, size_t i, Operon::Range /*unused*/)
{
    auto j = i - 1;
    auto k = j - nodes[j].Length - 1;
    Function<Type>{}(Ref<T>(m.col(i)), Ref<T>(m.col(j)), Ref<T>(m.col(k)));
}

template<NodeType Type, typename T>
requires Node::IsUnary<Type>
static inline void UnaryOp(Matrix<T>& m, Operon::Vector<Node> const& /*unused*/, size_t i, Operon::Range /*unused*/)
{
    Function<Type>{}(Ref<T>(m.col(i)), Ref<T>(m.col(i-1)));
}

struct Noop {
    template<typename... Args>
    void operator()(Args&&... /*unused*/) {}
};

template<NodeType Type, typename T>
static constexpr auto MakeCall() -> Dispatch::Callable<T>
{
    if constexpr (Node::IsNary<Type>) {
        return Callable<T>(NaryOp<Type, T>);
    } else if constexpr (Node::IsBinary<Type>) {
        return Callable<T>(BinaryOp<Type, T>);
    } else if constexpr (Node::IsUnary<Type>) {
        return Callable<T>(UnaryOp<Type, T>);
    }
}
};

template<typename... Ts>
struct DispatchTable {
    using Tuple    = std::tuple<Dispatch::Callable<Ts>...>;
    using Map      = Operon::Map<Operon::Hash, Tuple>;

private:
    Map map_;

    template<NodeType Type>
    static constexpr auto MakeTuple()
    {
        return std::make_tuple(Dispatch::MakeCall<Type, Ts>()...);
    };

    template<typename F>
    requires (std::is_invocable_r_v<void, F, Dispatch::Ref<Ts>&, Vector<Node> const&, size_t, Operon::Range> && ...)
    static constexpr auto MakeTuple(F&& f)
    {
        return std::make_tuple(Callable<Ts>(std::forward<F&&>(f))...);
    }

    // return the index of Callable<T> in Tuple 
    template<typename T>
    static auto constexpr IndexOf = []<auto... Idx>(std::index_sequence<Idx...>) {
        return -1 + ((std::is_same_v<T, std::tuple_element_t<Idx, Tuple>> ? Idx + 1 : 0) + ...);
    }(std::index_sequence_for<Ts...>{});

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

    explicit DispatchTable(Map const& map) : map_(map) { }
    explicit DispatchTable(Map&& map) : map_(std::move(map)) { }
    explicit DispatchTable(std::unordered_map<Operon::Hash, Tuple> const& map) : map_(map.begin(), map.end()) { }

    DispatchTable(DispatchTable const& other) : map_(other.map_) { }
    DispatchTable(DispatchTable &&other) noexcept : map_(std::move(other.map_)) { }

    auto GetMap() -> Map& { return map_; }
    auto GetMap() const -> Map const& { return map_; }

    template<typename T>
    inline auto Get(Operon::Hash const h) -> Dispatch::Callable<T>&
    {
        return const_cast<Dispatch::Callable<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->Get(h)); // NOLINT
    }

    template<typename T>
    [[nodiscard]] inline auto Get(Operon::Hash const h) const -> Dispatch::Callable<T> const&
    {
        constexpr int64_t idx = IndexOf<Dispatch::Callable<T>>;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(idx)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename F>
    void RegisterCallable(Operon::Hash hash, F&& f) {
        map_[hash] = MakeTuple<F, Ts...>(std::forward<F&&>(f));
    }

    template<typename T>
    [[nodiscard]] inline auto TryGet(Operon::Hash const h) const noexcept -> std::optional<Dispatch::Callable<T>>
    {
        constexpr int64_t idx = IndexOf<Dispatch::Callable<T>>;
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
