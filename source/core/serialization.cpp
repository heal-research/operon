// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/serialization.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>
#include <fmt/core.h>
#include <glaze/glaze.hpp>

// Ensure the enumerate below stays in sync with NodeType additions.
static_assert(Operon::NodeTypes::Count == 33,
              "NodeType count changed — update glz::meta<Operon::NodeType>");

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

// Bump Version whenever the on-disk layout changes in a backwards-incompatible way.
constexpr uint32_t CheckpointMagic   = 0x4F50434BU; // "OPCK"
constexpr uint32_t CheckpointVersion = 1U;

struct CheckpointProxy {
    uint32_t                                 Magic{CheckpointMagic};
    uint32_t                                 Version{CheckpointVersion};
    std::array<uint64_t, 4>                  RngState{};
    std::size_t                              Generation{0};
    std::vector<IndividualProxy>             Population;
    std::vector<std::array<uint64_t, 4>>     WorkerRngStates;
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
    if (proxies.empty()) {
        throw std::runtime_error("cannot deserialize tree: node list is empty");
    }
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
        "magic",            &T::Magic,
        "version",          &T::Version,
        "rng_state",        &T::RngState,
        "generation",       &T::Generation,
        "population",       &T::Population,
        "worker_rng_states", &T::WorkerRngStates
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
    if (auto ec = glz::read_json(tp, json); ec) {
        fmt::print(stderr, "serialization error (TreeFromJson): {}\n", glz::format_error(ec, json));
        return {};
    }
    try { return ProxiesToTree(tp.Nodes); } catch (std::exception const& e) {
        fmt::print(stderr, "serialization error (TreeFromJson): {}\n", e.what());
        return {};
    }
}

auto IndividualFromJson(std::string_view json) -> Individual
{
    IndividualProxy ip;
    if (auto ec = glz::read_json(ip, json); ec) {
        fmt::print(stderr, "serialization error (IndividualFromJson): {}\n", glz::format_error(ec, json));
        return {};
    }
    try { return ProxyToIndividual(ip); } catch (std::exception const& e) {
        fmt::print(stderr, "serialization error (IndividualFromJson): {}\n", e.what());
        return {};
    }
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
    if (auto ec = glz::read_beve(tp, data); ec) {
        fmt::print(stderr, "serialization error (TreeFromBeve): {}\n", glz::format_error(ec, data));
        return {};
    }
    try { return ProxiesToTree(tp.Nodes); } catch (std::exception const& e) {
        fmt::print(stderr, "serialization error (TreeFromBeve): {}\n", e.what());
        return {};
    }
}

auto IndividualFromBeve(std::string_view data) -> Individual
{
    IndividualProxy ip;
    if (auto ec = glz::read_beve(ip, data); ec) {
        fmt::print(stderr, "serialization error (IndividualFromBeve): {}\n", glz::format_error(ec, data));
        return {};
    }
    try { return ProxyToIndividual(ip); } catch (std::exception const& e) {
        fmt::print(stderr, "serialization error (IndividualFromBeve): {}\n", e.what());
        return {};
    }
}

// ---- Checkpoint ----

namespace {
auto ToProxy(Checkpoint const& cp) -> CheckpointProxy
{
    CheckpointProxy proxy;
    proxy.RngState        = cp.RngState;
    proxy.Generation      = cp.Generation;
    proxy.WorkerRngStates = cp.WorkerRngStates;
    proxy.Population.reserve(cp.Population.size());
    for (auto const& ind : cp.Population) { proxy.Population.push_back(IndividualToProxy(ind)); }
    return proxy;
}

auto FromProxy(CheckpointProxy const& proxy) -> Checkpoint
{
    if (proxy.Magic != CheckpointMagic || proxy.Version != CheckpointVersion) {
        fmt::print(stderr, "error: incompatible checkpoint format (magic={:#010x}, version={})\n",
                     proxy.Magic, proxy.Version);
        return {};
    }
    Checkpoint cp;
    cp.RngState        = proxy.RngState;
    cp.Generation      = proxy.Generation;
    cp.WorkerRngStates = proxy.WorkerRngStates;
    cp.Population.reserve(proxy.Population.size());
    for (auto const& ip : proxy.Population) { cp.Population.push_back(ProxyToIndividual(ip)); }
    return cp;
}
} // anonymous namespace

auto ToBeve(Checkpoint const& cp) -> std::string
{
    auto proxy  = ToProxy(cp);
    auto result = glz::write_beve(proxy);
    if (!result) {
        fmt::print(stderr, "serialization error (ToBeve Checkpoint): {}\n", glz::format_error(result.error()));
        return {};
    }
    return std::move(*result);
}

auto CheckpointFromBeve(std::string_view data) -> Checkpoint
{
    CheckpointProxy proxy;
    if (auto ec = glz::read_beve(proxy, data); ec) {
        fmt::print(stderr, "serialization error (CheckpointFromBeve): {}\n", glz::format_error(ec, data));
        return {};
    }
    try {
        return FromProxy(proxy);
    } catch (std::exception const& e) {
        fmt::print(stderr, "serialization error (CheckpointFromBeve): {}\n", e.what());
        return {};
    }
}

auto SaveCheckpoint(Checkpoint const& cp, std::string_view path) -> void
{
    auto const tmpPath = std::string(path) + ".tmp";
    auto proxy = ToProxy(cp);
    std::string buf;
    if (auto wec = glz::write_file_beve(proxy, tmpPath, buf); wec) {
        fmt::print(stderr, "error writing checkpoint to '{}': {}\n", tmpPath, glz::format_error(wec));
        std::filesystem::remove(tmpPath);
        return;
    }
    // On POSIX, rename() atomically replaces the destination so the old checkpoint
    // is never removed before the new one is in place.  On Windows, rename() fails
    // if the destination exists; only in that case do we remove it and retry.
    std::error_code ec;
    std::filesystem::rename(tmpPath, std::string(path), ec);
    if (ec) {
        std::filesystem::remove(std::string(path), ec);
        ec.clear();
        std::filesystem::rename(tmpPath, std::string(path), ec);
        if (ec) {
            fmt::print(stderr, "error renaming checkpoint '{}' to '{}': {}\n", tmpPath, path, ec.message());
            std::filesystem::remove(tmpPath, ec);
        }
    }
}

auto LoadCheckpoint(std::string_view path) -> Checkpoint
{
    CheckpointProxy proxy;
    std::string buf;
    if (auto ec = glz::read_file_beve(proxy, std::string(path), buf); ec) {
        fmt::print(stderr, "error loading checkpoint '{}': {}\n", path, glz::format_error(ec, buf));
        return {};
    }
    try {
        return FromProxy(proxy);
    } catch (std::exception const& e) {
        fmt::print(stderr, "error deserializing checkpoint '{}': {}\n", path, e.what());
        return {};
    }
}

} // namespace Operon::Serialization
