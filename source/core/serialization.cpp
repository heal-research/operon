// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/serialization.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"

#include <glaze/glaze.hpp>

// glz::meta specializations — all glaze details stay in this translation unit.
template <>
struct glz::meta<Operon::NodeType> {
    static constexpr auto value = glz::enumerate(
        "Add",      Operon::NodeType::Add,
        "Mul",      Operon::NodeType::Mul,
        "Sub",      Operon::NodeType::Sub,
        "Div",      Operon::NodeType::Div,
        "Fmin",     Operon::NodeType::Fmin,
        "Fmax",     Operon::NodeType::Fmax,
        "Aq",       Operon::NodeType::Aq,
        "Pow",      Operon::NodeType::Pow,
        "Powabs",   Operon::NodeType::Powabs,
        "Abs",      Operon::NodeType::Abs,
        "Acos",     Operon::NodeType::Acos,
        "Asin",     Operon::NodeType::Asin,
        "Atan",     Operon::NodeType::Atan,
        "Cbrt",     Operon::NodeType::Cbrt,
        "Ceil",     Operon::NodeType::Ceil,
        "Cos",      Operon::NodeType::Cos,
        "Cosh",     Operon::NodeType::Cosh,
        "Exp",      Operon::NodeType::Exp,
        "Floor",    Operon::NodeType::Floor,
        "Log",      Operon::NodeType::Log,
        "Logabs",   Operon::NodeType::Logabs,
        "Log1p",    Operon::NodeType::Log1p,
        "Sin",      Operon::NodeType::Sin,
        "Sinh",     Operon::NodeType::Sinh,
        "Sqrt",     Operon::NodeType::Sqrt,
        "Sqrtabs",  Operon::NodeType::Sqrtabs,
        "Tan",      Operon::NodeType::Tan,
        "Tanh",     Operon::NodeType::Tanh,
        "Square",   Operon::NodeType::Square,
        "Dynamic",  Operon::NodeType::Dynamic,
        "Constant", Operon::NodeType::Constant,
        "Variable", Operon::NodeType::Variable,
        "Ref",      Operon::NodeType::Ref
    );
};

// Serialization proxy for Node — excludes fields recomputed by UpdateNodes().
namespace {
struct NodeProxy {
    Operon::NodeType Type{};
    Operon::Hash     HashValue{};
    Operon::Scalar   Value{};
    bool             IsEnabled{true};
    bool             Optimize{false};
    uint16_t         RefTo{0};

    static auto FromNode(Operon::Node const& n) noexcept -> NodeProxy {
        return { n.Type, n.HashValue, n.Value, n.IsEnabled, n.Optimize, n.RefTo };
    }

    auto ToNode() const noexcept -> Operon::Node {
        Operon::Node n(Type, HashValue);
        n.Value     = Value;
        n.IsEnabled = IsEnabled;
        n.Optimize  = Optimize;
        n.RefTo     = RefTo;
        return n;
    }
};
} // anonymous namespace

template <>
struct glz::meta<NodeProxy> {
    using T = NodeProxy;
    static constexpr auto value = glz::object(
        "type",      &T::Type,
        "hash",      &T::HashValue,
        "value",     &T::Value,
        "enabled",   &T::IsEnabled,
        "optimize",  &T::Optimize,
        "ref_to",    &T::RefTo
    );
};

template <>
struct glz::meta<Operon::Individual> {
    using T = Operon::Individual;
    static constexpr auto value = glz::object(
        "fitness",  &T::Fitness,
        "rank",     &T::Rank,
        "distance", &T::Distance
    );
};

namespace {

auto NodesToProxies(Operon::Vector<Operon::Node> const& nodes) -> std::vector<NodeProxy>
{
    std::vector<NodeProxy> proxies;
    proxies.reserve(nodes.size());
    for (auto const& n : nodes) { proxies.push_back(NodeProxy::FromNode(n)); }
    return proxies;
}

auto ProxiesToTree(std::vector<NodeProxy> const& proxies) -> Operon::Tree
{
    Operon::Vector<Operon::Node> nodes;
    nodes.reserve(proxies.size());
    for (auto const& p : proxies) { nodes.push_back(p.ToNode()); }
    return Operon::Tree(std::move(nodes)).UpdateNodes();
}

struct TreeJson {
    std::vector<NodeProxy> nodes;
};

struct IndividualJson {
    TreeJson               tree;
    Operon::Vector<Operon::Scalar> fitness;
    std::size_t            rank{};
    Operon::Scalar         distance{};
};

} // anonymous namespace

template <>
struct glz::meta<TreeJson> {
    using T = TreeJson;
    static constexpr auto value = glz::object("nodes", &T::nodes);
};

template <>
struct glz::meta<IndividualJson> {
    using T = IndividualJson;
    static constexpr auto value = glz::object(
        "tree",     &T::tree,
        "fitness",  &T::fitness,
        "rank",     &T::rank,
        "distance", &T::distance
    );
};

namespace Operon::Serialization {

auto ToJson(Tree const& tree) -> std::string
{
    TreeJson tj{ NodesToProxies(tree.Nodes()) };
    auto result = glz::write_json(tj);
    if (!result) {
        return "{}";
    }
    return std::move(*result);
}

auto ToJson(Individual const& ind) -> std::string
{
    IndividualJson ij{
        { NodesToProxies(ind.Genotype.Nodes()) },
        ind.Fitness,
        ind.Rank,
        ind.Distance
    };
    auto result = glz::write_json(ij);
    if (!result) {
        return "{}";
    }
    return std::move(*result);
}

auto ToJson(std::span<Individual const> front) -> std::string
{
    std::vector<IndividualJson> arr;
    arr.reserve(front.size());
    for (auto const& ind : front) {
        arr.push_back({
            { NodesToProxies(ind.Genotype.Nodes()) },
            ind.Fitness,
            ind.Rank,
            ind.Distance
        });
    }
    auto result = glz::write_json(arr);
    if (!result) {
        return "[]";
    }
    return std::move(*result);
}

auto TreeFromJson(std::string_view json) -> Tree
{
    TreeJson tj;
    if (glz::read_json(tj, json)) { return {}; }
    return ProxiesToTree(tj.nodes);
}

auto IndividualFromJson(std::string_view json) -> Individual
{
    IndividualJson ij;
    if (glz::read_json(ij, json)) { return {}; }
    Individual ind;
    ind.Genotype = ProxiesToTree(ij.tree.nodes);
    ind.Fitness  = std::move(ij.fitness);
    ind.Rank     = ij.rank;
    ind.Distance = ij.distance;
    return ind;
}

} // namespace Operon::Serialization
