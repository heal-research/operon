// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/serialization.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"

#include <exception>
#include <filesystem>
#include <optional>
#include <utility>
#include <fmt/core.h>
#include <glaze/glaze.hpp>

// Ensure the enumerate below stays in sync with NodeType additions.
static_assert(Operon::NodeTypes::Count == 4,
              "NodeType count changed — update glz::meta<Operon::NodeType>");

// glz::meta specializations — all glaze details stay in this translation unit.
//
// BACKWARD-COMPATIBILITY WARNING: this enumerate shrank from 33 entries
// (one per built-in math op, plus Dynamic/Constant/Variable/Ref) to 4
// (Constant/Variable/Ref/Function) as part of the NodeType collapse. A
// built-in math op is now a Function-typed Node distinguished by HashValue
// (see BuiltinOp in node.hpp), not by a NodeType enumerator - a Function
// node's specific op is NOT reconstructible from NodeProxy::Type alone.
//
// Verified against glaze 7.8.3's BEVE codec (beve/read.hpp, beve/write.hpp):
// glaze_enum_t types are NOT written by name in BEVE (unlike JSON) - the
// raw underlying integer ordinal is written/read directly, with no
// validation on read that the value is a member of the enumerate list.
// A pre-collapse BEVE Tree/Individual export's old NodeType ordinal
// (e.g. Sin=22) would decode as `static_cast<NodeType>(22)`, a garbage
// out-of-range value - CheckpointProxy's Magic/Version fields (bumped
// below) guard checkpoint loads against this, but ToBeve(Tree)/
// ToBeve(Individual) exports have no equivalent guard, before or after
// this change. Do not attempt to load a pre-collapse .beve Tree/Individual
// export against this code; a dedicated versioned legacy reader (translating
// old ordinals to the new BuiltinOp/Function representation) would be
// needed to do so safely - not implemented here.
template <>
struct glz::meta<Operon::NodeType> {
    static constexpr auto value = glz::enumerate(
        "Constant", Operon::NodeType::Constant,
        "Variable", Operon::NodeType::Variable,
        "Ref",      Operon::NodeType::Ref,
        "Function", Operon::NodeType::Function
    );
};

// Serialization proxies — exclude fields recomputed by UpdateNodes().
// Member names are CamelCase; serialized key names (in glz::meta) are lowercase.
namespace {
struct NodeProxy {
    Operon::NodeType Type{};
    Operon::Hash     HashValue{};
    Operon::Scalar   Value{};
    // Explicit as of the NodeType collapse: Node's two-arg constructor no
    // longer infers Arity from Type's position in the enum (there is no
    // longer a position to infer from - every Function node has the same
    // Type). Without this field every deserialized Function node would get
    // Arity=0, silently corrupting the tree structure.
    uint16_t         Arity{0};
    bool             IsEnabled{true};
    bool             Optimize{false};
    uint16_t         RefTo{0};

    static auto FromNode(Operon::Node const& n) noexcept -> NodeProxy {
        return { n.Type, n.HashValue, n.Value, n.Arity, n.IsEnabled, n.Optimize, n.RefTo };
    }

    auto ToNode() const noexcept -> Operon::Node {
        Operon::Node n(Type, HashValue);
        n.Value     = Value;
        n.Arity     = Arity;
        n.Length    = Arity; // leaf-level Length; Tree::UpdateNodes() (called by ProxiesToTree) recomputes the real subtree Length
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
    uint64_t                       Rank{};
    Operon::Scalar                 Distance{};
};

// Bump Version whenever the on-disk layout changes in a backwards-incompatible way.
constexpr uint32_t CheckpointMagic   = 0x4F50434BU; // "OPCK"
// Bumped: NodeType collapse (glz::meta<NodeType> shrank from 33 to 4
// entries, NodeProxy gained an explicit Arity field) is a breaking,
// silently-misdecodable format change for BEVE - see the warning above
// glz::meta<Operon::NodeType>. A version-1 checkpoint will now be cleanly
// rejected by FromProxy's Magic/Version check below, rather than partially
// decoding into corrupted trees.
constexpr uint32_t CheckpointVersion = 2U;

struct CheckpointProxy {
    uint32_t                                 Magic{CheckpointMagic};
    uint32_t                                 Version{CheckpointVersion};
    std::array<uint64_t, 4>                  RngState{};
    uint64_t                                 Generation{0};
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
    if (proxies.empty()) { return {}; }
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
        "arity",    &T::Arity,
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

auto TreeFromJson(std::string_view json) -> std::optional<Tree>
{
    TreeProxy tp;
    if (auto ec = glz::read_json(tp, json); ec) {
        fmt::print(stderr, "serialization error (TreeFromJson): {}\n", glz::format_error(ec, json));
        return std::nullopt;
    }
    return ProxiesToTree(tp.Nodes);
}

auto IndividualFromJson(std::string_view json) -> std::optional<Individual>
{
    IndividualProxy ip;
    if (auto ec = glz::read_json(ip, json); ec) {
        fmt::print(stderr, "serialization error (IndividualFromJson): {}\n", glz::format_error(ec, json));
        return std::nullopt;
    }
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

auto TreeFromBeve(std::string_view data) -> std::optional<Tree>
{
    TreeProxy tp;
    if (auto ec = glz::read_beve(tp, data); ec) {
        fmt::print(stderr, "serialization error (TreeFromBeve): {}\n", glz::format_error(ec, data));
        return std::nullopt;
    }
    return ProxiesToTree(tp.Nodes);
}

auto IndividualFromBeve(std::string_view data) -> std::optional<Individual>
{
    IndividualProxy ip;
    if (auto ec = glz::read_beve(ip, data); ec) {
        fmt::print(stderr, "serialization error (IndividualFromBeve): {}\n", glz::format_error(ec, data));
        return std::nullopt;
    }
    return ProxyToIndividual(ip);
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

auto FromProxy(CheckpointProxy const& proxy) -> std::optional<Checkpoint>
{
    if (proxy.Magic != CheckpointMagic || proxy.Version != CheckpointVersion) {
        fmt::print(stderr, "error: incompatible checkpoint format (magic={:#010x}, version={})\n",
                     proxy.Magic, proxy.Version);
        return std::nullopt;
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

auto CheckpointFromBeve(std::string_view data) -> std::optional<Checkpoint>
{
    CheckpointProxy proxy;
    if (auto ec = glz::read_beve(proxy, data); ec) {
        fmt::print(stderr, "serialization error (CheckpointFromBeve): {}\n", glz::format_error(ec, data));
        return std::nullopt;
    }
    return FromProxy(proxy);
}

auto SaveCheckpoint(Checkpoint const& cp, std::string_view path) -> bool
{
    auto const tmpPath = std::string(path) + ".tmp";
    auto proxy = ToProxy(cp);
    std::string buf;
    if (auto wec = glz::write_file_beve(proxy, tmpPath, buf); wec) {
        fmt::print(stderr, "error writing checkpoint to '{}': {}\n", tmpPath, glz::format_error(wec));
        std::filesystem::remove(tmpPath);
        return false;
    }
    // On POSIX, rename() atomically replaces the destination — no removal needed.
    // On Windows, rename() fails when the destination already exists as a regular
    // file; only in that specific case do we remove it and retry.  For any other
    // failure reason (permissions, I/O, cross-device) we leave the old checkpoint
    // in place rather than risk destroying it.
    std::error_code ec;
    std::filesystem::rename(tmpPath, std::string(path), ec);
    if (ec) {
        std::error_code existsEc;
        if (std::filesystem::is_regular_file(std::string(path), existsEc) && !existsEc) {
            std::filesystem::remove(std::string(path), ec);
            ec.clear();
            std::filesystem::rename(tmpPath, std::string(path), ec);
        }
        if (ec) {
            fmt::print(stderr, "error renaming checkpoint '{}' to '{}': {}\n", tmpPath, path, ec.message());
            std::filesystem::remove(tmpPath, ec);
            return false;
        }
    }
    return true;
}

auto LoadCheckpoint(std::string_view path) -> std::optional<Checkpoint>
{
    CheckpointProxy proxy;
    std::string buf;
    if (auto ec = glz::read_file_beve(proxy, std::string(path), buf); ec) {
        fmt::print(stderr, "error loading checkpoint '{}': {}\n", path, glz::format_error(ec, buf));
        return std::nullopt;
    }
    return FromProxy(proxy);
}

} // namespace Operon::Serialization
