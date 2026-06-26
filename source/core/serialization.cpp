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

// Serialization proxies — exclude fields recomputed by UpdateNodes().
// Member names are CamelCase; serialized key names (in glz::meta) are lowercase.
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

struct TreeProxy {
    std::vector<NodeProxy> Nodes;
};

struct IndividualProxy {
    TreeProxy                      Tree;
    Operon::Vector<Operon::Scalar> Fitness;
    std::size_t                    Rank{};
    Operon::Scalar                 Distance{};
};

struct CheckpointProxy {
    std::array<uint64_t, 4>      RngState{};
    std::size_t                  Generation{0};
    std::vector<IndividualProxy> Population;
};

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

auto IndividualToProxy(Operon::Individual const& ind) -> IndividualProxy
{
    return { { NodesToProxies(ind.Genotype.Nodes()) }, ind.Fitness, ind.Rank, ind.Distance };
}

auto ProxyToIndividual(IndividualProxy const& p) -> Operon::Individual
{
    Operon::Individual ind;
    ind.Genotype = ProxiesToTree(p.Tree.Nodes);
    ind.Fitness  = p.Fitness;
    ind.Rank     = p.Rank;
    ind.Distance = p.Distance;
    return ind;
}
} // anonymous namespace

template <>
struct glz::meta<NodeProxy> {
    using T = NodeProxy;
    static constexpr auto value = glz::object(
        "type",     &T::Type,
        "hash",     &T::HashValue,
        "value",    &T::Value,
        "enabled",  &T::IsEnabled,
        "optimize", &T::Optimize,
        "ref_to",   &T::RefTo
    );
};

template <>
struct glz::meta<TreeProxy> {
    using T = TreeProxy;
    static constexpr auto value = glz::object("nodes", &T::Nodes);
};

template <>
struct glz::meta<IndividualProxy> {
    using T = IndividualProxy;
    static constexpr auto value = glz::object(
        "tree",     &T::Tree,
        "fitness",  &T::Fitness,
        "rank",     &T::Rank,
        "distance", &T::Distance
    );
};

template <>
struct glz::meta<CheckpointProxy> {
    using T = CheckpointProxy;
    static constexpr auto value = glz::object(
        "rng_state",  &T::RngState,
        "generation", &T::Generation,
        "population", &T::Population
    );
};

namespace Operon::Serialization {

// ---- JSON ----

auto ToJson(Tree const& tree) -> std::string
{
    TreeProxy tp{ NodesToProxies(tree.Nodes()) };
    auto result = glz::write_json(tp);
    return result ? std::move(*result) : "{}";
}

auto ToJson(Individual const& individual) -> std::string
{
    auto ip = IndividualToProxy(individual);
    auto result = glz::write_json(ip);
    return result ? std::move(*result) : "{}";
}

auto ToJson(std::span<Individual const> front) -> std::string
{
    std::vector<IndividualProxy> arr;
    arr.reserve(front.size());
    for (auto const& ind : front) { arr.push_back(IndividualToProxy(ind)); }
    auto result = glz::write_json(arr);
    return result ? std::move(*result) : "[]";
}

auto TreeFromJson(std::string_view json) -> Tree
{
    TreeProxy tp;
    if (glz::read_json(tp, json)) { return {}; }
    return ProxiesToTree(tp.Nodes);
}

auto IndividualFromJson(std::string_view json) -> Individual
{
    IndividualProxy ip;
    if (glz::read_json(ip, json)) { return {}; }
    return ProxyToIndividual(ip);
}

// ---- BEVE ----

auto ToBeve(Tree const& tree) -> std::string
{
    TreeProxy tp{ NodesToProxies(tree.Nodes()) };
    auto result = glz::write_beve(tp);
    return result ? std::move(*result) : std::string{};
}

auto ToBeve(Individual const& individual) -> std::string
{
    auto ip = IndividualToProxy(individual);
    auto result = glz::write_beve(ip);
    return result ? std::move(*result) : std::string{};
}

auto ToBeve(std::span<Individual const> front) -> std::string
{
    std::vector<IndividualProxy> arr;
    arr.reserve(front.size());
    for (auto const& ind : front) { arr.push_back(IndividualToProxy(ind)); }
    auto result = glz::write_beve(arr);
    return result ? std::move(*result) : std::string{};
}

auto TreeFromBeve(std::string_view data) -> Tree
{
    TreeProxy tp;
    if (glz::read_beve(tp, data)) { return {}; }
    return ProxiesToTree(tp.Nodes);
}

auto IndividualFromBeve(std::string_view data) -> Individual
{
    IndividualProxy ip;
    if (glz::read_beve(ip, data)) { return {}; }
    return ProxyToIndividual(ip);
}

// ---- Checkpoint ----

auto ToBeve(Checkpoint const& cp) -> std::string
{
    CheckpointProxy proxy;
    proxy.RngState    = cp.RngState;
    proxy.Generation  = cp.Generation;
    proxy.Population.reserve(cp.Population.size());
    for (auto const& ind : cp.Population) { proxy.Population.push_back(IndividualToProxy(ind)); }
    auto result = glz::write_beve(proxy);
    return result ? std::move(*result) : std::string{};
}

auto CheckpointFromBeve(std::string_view data) -> Checkpoint
{
    CheckpointProxy proxy;
    if (glz::read_beve(proxy, data)) { return {}; }
    Checkpoint cp;
    cp.RngState   = proxy.RngState;
    cp.Generation = proxy.Generation;
    cp.Population.reserve(proxy.Population.size());
    for (auto const& ip : proxy.Population) { cp.Population.push_back(ProxyToIndividual(ip)); }
    return cp;
}

auto SaveCheckpoint(Checkpoint const& cp, std::string_view path) -> void
{
    CheckpointProxy proxy;
    proxy.RngState   = cp.RngState;
    proxy.Generation = cp.Generation;
    proxy.Population.reserve(cp.Population.size());
    for (auto const& ind : cp.Population) { proxy.Population.push_back(IndividualToProxy(ind)); }
    std::string buf;
    (void)glz::write_file_beve(proxy, std::string(path), buf);
}

auto LoadCheckpoint(std::string_view path) -> Checkpoint
{
    CheckpointProxy proxy;
    std::string buf;
    if (glz::read_file_beve(proxy, std::string(path), buf)) { return {}; }
    Checkpoint cp;
    cp.RngState   = proxy.RngState;
    cp.Generation = proxy.Generation;
    cp.Population.reserve(proxy.Population.size());
    for (auto const& ip : proxy.Population) { cp.Population.push_back(ProxyToIndividual(ip)); }
    return cp;
}

} // namespace Operon::Serialization
