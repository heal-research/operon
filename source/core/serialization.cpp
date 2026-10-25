// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/serialization.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/hash.hpp"
#include "operon/hash/zobrist.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <glaze/glaze.hpp>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

// Ensure the enumerate below stays in sync with NodeType additions.
static_assert(Operon::NodeTypes::Count == 4, "NodeType count changed — update glz::meta<Operon::NodeType>");

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
// out-of-range value. CheckpointProxy has its own magic/version guard, and
// current Tree/Individual exports now have the TreeBeveProxy and
// IndividualBeveProxy guards below. Pre-versioning .beve Tree/Individual
// exports are intentionally rejected; a dedicated legacy reader translating
// old ordinals to the new BuiltinOp/Function representation is not implemented.
template <> struct glz::meta<Operon::NodeType> {
    static constexpr auto value = glz::enumerate("Constant", Operon::NodeType::Constant, "Variable",
        Operon::NodeType::Variable, "Ref", Operon::NodeType::Ref, "Function", Operon::NodeType::Function);
};

// Serialization proxies — exclude fields recomputed by UpdateNodes().
// Member names are CamelCase; serialized key names (in glz::meta) are lowercase.
namespace {
struct NodeProxy {
    Operon::NodeType Type {};
    Operon::Hash HashValue {};
    Operon::Scalar Value {};
    // Explicit as of the NodeType collapse: Node's two-arg constructor no
    // longer infers Arity from Type's position in the enum (there is no
    // longer a position to infer from - every Function node has the same
    // Type). Without this field every deserialized Function node would get
    // Arity=0, silently corrupting the tree structure.
    uint16_t Arity { 0 };
    bool IsEnabled { true };
    bool Optimize { false };
    uint16_t RefTo { 0 };

    static auto FromNode(Operon::Node const& n) noexcept -> NodeProxy
    {
        return { n.Type, n.HashValue, n.Value, n.Arity, n.IsEnabled, n.Optimize, n.RefTo };
    }

    auto ToNode() const noexcept -> Operon::Node
    {
        Operon::Node n(Type, HashValue);
        n.Value = Value;
        n.Arity = Arity;
        n.Length = Arity; // leaf-level Length; Tree::UpdateNodes() (called by ProxiesToTree) recomputes the real
                          // subtree Length
        n.IsEnabled = IsEnabled;
        n.Optimize = Optimize;
        n.RefTo = RefTo;
        return n;
    }
};

struct TreeProxy {
    std::vector<NodeProxy> Nodes;
};

struct IndividualProxy {
    TreeProxy Tree;
    Operon::Vector<Operon::Scalar> Fitness;
    uint64_t Rank {};
    Operon::Scalar Distance {};
};

// Bump a payload version whenever its matching on-disk layout changes.
constexpr uint32_t CheckpointMagic = 0x4F50434BU; // "OPCK"
constexpr uint32_t CheckpointVersion = 3U; // 3: the transposition cache is part of the checkpoint
constexpr uint32_t TreeMagic = 0x4F505452U; // "OPTR"
constexpr uint32_t IndividualMagic = 0x4F50494EU; // "OPIN"
constexpr uint32_t PopulationMagic = 0x4F505046U; // "OPPF"
constexpr uint32_t BevePayloadVersion = 1U;

struct TreeBeveProxy {
    uint32_t Magic { TreeMagic };
    uint32_t Version { BevePayloadVersion };
    TreeProxy Tree;
};

struct IndividualBeveProxy {
    uint32_t Magic { IndividualMagic };
    uint32_t Version { BevePayloadVersion };
    IndividualProxy Individual;
};

struct PopulationBeveProxy {
    uint32_t Magic { PopulationMagic };
    uint32_t Version { BevePayloadVersion };
    std::vector<IndividualProxy> Population;
};

// The checkpoint's own tree and individual proxies: the same layout as TreeProxy and IndividualProxy, so a checkpoint
// is encoded as before, but with the bounds of the arrays attached as BEVE-only wrappers (glz::max_length), which the
// JSON proxies cannot carry.
struct CheckpointTreeProxy {
    std::vector<NodeProxy> Nodes;
};

struct CheckpointIndividualProxy {
    CheckpointTreeProxy Tree;
    Operon::Vector<Operon::Scalar> Fitness;
    uint64_t Rank {};
    Operon::Scalar Distance {};
};

struct CheckpointProxy {
    uint32_t Magic { CheckpointMagic };
    uint32_t Version { CheckpointVersion };
    std::array<uint64_t, 4> RngState {};
    uint64_t Generation { 0 };
    std::vector<CheckpointIndividualProxy> Population;
    std::vector<std::array<uint64_t, 4>> WorkerRngStates;
    std::optional<Operon::Serialization::CacheCheckpoint> Cache;
    std::optional<Operon::Serialization::RunFingerprint> Run;
};

auto NodesToProxies(Operon::Vector<Operon::Node> const& nodes) -> std::vector<NodeProxy>
{
    std::vector<NodeProxy> proxies;
    proxies.reserve(nodes.size());
    for (auto const& n : nodes) {
        proxies.push_back(NodeProxy::FromNode(n));
    }
    return proxies;
}

// Nodes have 16-bit Length, Parent and RefTo; a tree with more nodes cannot be represented.
constexpr std::size_t MaxTreeNodes = std::numeric_limits<uint16_t>::max();
static_assert(MaxTreeNodes == Operon::Serialization::MaxCheckpointTreeNodes);

// The arity a built-in function must have whatever the primitive set, if it has a fixed one: the binary functions
// (Aq, Pow, Powabs) take two arguments and the unary ones one. The n-ary functions (Add, Mul, Sub, Div, Fmin, Fmax)
// take any number of arguments (Sub and Div with one are the negation and the inverse), and the arity of a function
// that is not built in is a property of the primitive set, which is not available here.
constexpr auto FixedBuiltinArity(Operon::Hash hash) -> std::optional<uint16_t>
{
    if (hash >= Operon::BuiltinOpCount) {
        return std::nullopt;
    }
    auto const op = static_cast<Operon::BuiltinOp>(hash);
    if (op > Operon::BuiltinOp::Powabs) {
        return uint16_t { 1 };
    }
    if (op > Operon::BuiltinOp::Fmax) {
        return uint16_t { 2 };
    }
    return std::nullopt;
}

// Empty if `nodes` is a tree that the rest of the library can work with, else the reason. Nothing is assumed about the
// nodes and nothing is indexed out of bounds: a valid node type, terminals (Constant, Variable, Ref) without
// arguments, functions with at least one argument (and the fixed arity of a binary or unary built-in function), Ref
// nodes pointing to an earlier node, and nodes in postfix order, every node finding its arguments on the stack and
// exactly one value left. An empty tree is a valid (unfilled) genotype; whether it may occur is the caller's decision.
auto TreeProblem(std::vector<NodeProxy> const& nodes) -> std::string
{
    if (nodes.size() > MaxTreeNodes) {
        return fmt::format("a tree has {} nodes, at most {} are supported", nodes.size(), MaxTreeNodes);
    }
    std::size_t depth = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto const& node = nodes[i];
        auto const type = static_cast<unsigned>(static_cast<Operon::UnderlyingNodeType>(node.Type));
        if (type >= Operon::NodeTypes::Count) {
            return fmt::format("node {} of a tree has the unknown type {}", i, type);
        }
        if (node.Type == Operon::NodeType::Function) {
            if (node.Arity == 0) {
                return fmt::format("node {} of a tree is a function without arguments", i);
            }
            if (auto const fixed = FixedBuiltinArity(node.HashValue); fixed && *fixed != node.Arity) {
                return fmt::format(
                    "node {} of a tree is a built-in function with {} arguments, it takes {}", i, node.Arity, *fixed);
            }
        } else {
            if (node.Arity != 0) {
                return fmt::format("node {} of a tree is a terminal with {} arguments", i, node.Arity);
            }
            if (node.Type == Operon::NodeType::Ref && node.RefTo >= i) {
                return fmt::format(
                    "node {} of a tree is a Ref to node {}, which does not come before it", i, node.RefTo);
            }
        }
        if (node.Arity > depth) {
            return fmt::format("node {} of a tree has more arguments than the nodes before it provide", i);
        }
        depth = depth - node.Arity + 1;
    }
    if (!nodes.empty() && depth != 1) {
        return "a tree is not a single postfix expression";
    }
    return {};
}

// The tree of `proxies`, after TreeProblem and Tree::Validate (on the tree with its derived node metadata
// recomputed, which is what the rest of the library relies on) accepted it. `problem` is empty on success.
auto ProxiesToTree(std::vector<NodeProxy> const& proxies, std::string& problem) -> Operon::Tree
{
    problem = TreeProblem(proxies);
    if (!problem.empty() || proxies.empty()) {
        return {};
    }
    Operon::Vector<Operon::Node> nodes;
    nodes.reserve(proxies.size());
    for (auto const& p : proxies) {
        nodes.push_back(p.ToNode());
    }
    auto tree = Operon::Tree(std::move(nodes)).UpdateNodes();
    if (auto const valid = tree.Validate(); !valid) {
        problem = fmt::format("a tree fails the tree validation (error {})", static_cast<int>(valid.error()));
        return {};
    }
    return tree;
}

auto IndividualToProxy(Operon::Individual const& ind) -> IndividualProxy
{
    return { { NodesToProxies(ind.Genotype.Nodes()) }, ind.Fitness, ind.Rank, ind.Distance };
}

// Individual decoding for the callers that do not move: Genotype is empty and `problem` set if the tree is not valid.
auto ProxyToIndividual(IndividualProxy const& p, std::string& problem) -> Operon::Individual
{
    Operon::Individual ind;
    ind.Genotype = ProxiesToTree(p.Tree.Nodes, problem);
    ind.Fitness = p.Fitness;
    ind.Rank = p.Rank;
    ind.Distance = p.Distance;
    return ind;
}
} // anonymous namespace

template <> struct glz::meta<NodeProxy> {
    using T = NodeProxy;
    static constexpr auto value = glz::object("type", &T::Type, "hash", &T::HashValue, "value", &T::Value, "arity",
        &T::Arity, "enabled", &T::IsEnabled, "optimize", &T::Optimize, "ref_to", &T::RefTo);
};

template <> struct glz::meta<TreeProxy> {
    using T = TreeProxy;
    static constexpr auto value = glz::object("nodes", &T::Nodes);
};

template <> struct glz::meta<IndividualProxy> {
    using T = IndividualProxy;
    static constexpr auto value
        = glz::object("tree", &T::Tree, "fitness", &T::Fitness, "rank", &T::Rank, "distance", &T::Distance);
};

template <> struct glz::meta<CheckpointTreeProxy> {
    using T = CheckpointTreeProxy;
    static constexpr auto value
        = glz::object("nodes", glz::max_length<&T::Nodes, Operon::Serialization::MaxCheckpointTreeNodes>);
};

template <> struct glz::meta<CheckpointIndividualProxy> {
    using T = CheckpointIndividualProxy;
    static constexpr auto value = glz::object("tree", &T::Tree, "fitness",
        glz::max_length<&T::Fitness, Operon::Serialization::MaxCheckpointObjectives>, "rank", &T::Rank, "distance",
        &T::Distance);
};

template <> struct glz::meta<TreeBeveProxy> {
    using T = TreeBeveProxy;
    static constexpr auto value = glz::object("magic", &T::Magic, "version", &T::Version, "tree", &T::Tree);
};

template <> struct glz::meta<IndividualBeveProxy> {
    using T = IndividualBeveProxy;
    static constexpr auto value = glz::object("magic", &T::Magic, "version", &T::Version, "individual", &T::Individual);
};

template <> struct glz::meta<PopulationBeveProxy> {
    using T = PopulationBeveProxy;
    static constexpr auto value = glz::object("magic", &T::Magic, "version", &T::Version, "population", &T::Population);
};

template <> struct glz::meta<Operon::Serialization::CacheCheckpoint> {
    using T = Operon::Serialization::CacheCheckpoint;
    // CacheCheckpoint is only ever encoded as BEVE, so the bounds of the arrays can be part of it. The arrays are typed
    // (byte-bounded: Glaze checks the declared count against the bytes left before it allocates).
    static constexpr auto value = glz::object("table_fingerprint", &T::TableFingerprint, "clock", &T::Clock, "max_age",
        &T::MaxAge, "visibility", &T::Visibility, "keys",
        glz::max_length<&T::Keys, Operon::Serialization::MaxCheckpointCacheEntries>, "insert_generations",
        glz::max_length<&T::InsertGenerations, Operon::Serialization::MaxCheckpointCacheEntries>, "lengths",
        glz::max_length<&T::Lengths, Operon::Serialization::MaxCheckpointCacheEntries>, "values",
        glz::max_length<&T::Values,
            Operon::Serialization::MaxCheckpointCacheEntries * Operon::Serialization::MaxCheckpointObjectives>);
};

template <> struct glz::meta<Operon::Serialization::RunFingerprint> {
    using T = Operon::Serialization::RunFingerprint;
    static constexpr auto value
        = glz::object("groups", glz::max_length<&T::Groups, Operon::Serialization::MaxCheckpointFingerprintEntries>,
            "hashes", glz::max_length<&T::Hashes, Operon::Serialization::MaxCheckpointFingerprintEntries>);
};

template <> struct glz::meta<CheckpointProxy> {
    using T = CheckpointProxy;
    static constexpr auto value = glz::object("magic", &T::Magic, "version", &T::Version, "rng_state", &T::RngState,
        "generation", &T::Generation, "population",
        glz::max_length<&T::Population, Operon::Serialization::MaxCheckpointPopulation>, "worker_rng_states",
        glz::max_length<&T::WorkerRngStates, Operon::Serialization::MaxCheckpointWorkerStates>, "cache", &T::Cache,
        "run", &T::Run);
};

namespace {
// Reading options of a checkpoint. Every array of the checkpoint carries its own bound (glz::max_length in the metas
// above: population, worker states, tree nodes, fitness values, cache arrays, run fingerprint), checked together with
// the bytes left in the buffer before anything is allocated. The bound here is for any array that does not (none
// today); the strings are the names of the run fingerprint groups.
struct CheckpointReadOpts : glz::opts {
    uint32_t format = glz::BEVE;
    size_t max_array_size = 4096;
    size_t max_string_length = Operon::Serialization::MaxCheckpointGroupNameBytes;
};
} // namespace

namespace Operon::Serialization {

// ---- JSON ----

auto ToJson(Tree const& tree) -> std::string
{
    TreeProxy tp { NodesToProxies(tree.Nodes()) };
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
    for (auto const& ind : front) {
        arr.push_back(IndividualToProxy(ind));
    }
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
    std::string problem;
    auto tree = ProxiesToTree(tp.Nodes, problem);
    if (!problem.empty()) {
        fmt::print(stderr, "serialization error (TreeFromJson): {}\n", problem);
        return std::nullopt;
    }
    return tree;
}

auto IndividualFromJson(std::string_view json) -> std::optional<Individual>
{
    IndividualProxy ip;
    if (auto ec = glz::read_json(ip, json); ec) {
        fmt::print(stderr, "serialization error (IndividualFromJson): {}\n", glz::format_error(ec, json));
        return std::nullopt;
    }
    std::string problem;
    auto individual = ProxyToIndividual(ip, problem);
    if (!problem.empty()) {
        fmt::print(stderr, "serialization error (IndividualFromJson): {}\n", problem);
        return std::nullopt;
    }
    return individual;
}

// ---- BEVE ----

auto ToBeve(Tree const& tree) -> std::string
{
    TreeBeveProxy proxy { .Tree = { NodesToProxies(tree.Nodes()) } };
    auto result = glz::write_beve(proxy);
    return result ? std::move(*result) : std::string {};
}

auto ToBeve(Individual const& individual) -> std::string
{
    IndividualBeveProxy proxy { .Individual = IndividualToProxy(individual) };
    auto result = glz::write_beve(proxy);
    return result ? std::move(*result) : std::string {};
}

auto ToBeve(std::span<Individual const> front) -> std::string
{
    PopulationBeveProxy proxy;
    proxy.Population.reserve(front.size());
    for (auto const& ind : front) {
        proxy.Population.push_back(IndividualToProxy(ind));
    }
    auto result = glz::write_beve(proxy);
    return result ? std::move(*result) : std::string {};
}

auto TreeFromBeve(std::string_view data) -> std::optional<Tree>
{
    TreeBeveProxy proxy;
    if (auto ec = glz::read_beve(proxy, data); ec) {
        fmt::print(stderr, "serialization error (TreeFromBeve): {}\n", glz::format_error(ec));
        return std::nullopt;
    }
    if (proxy.Magic != TreeMagic || proxy.Version != BevePayloadVersion) {
        fmt::print(
            stderr, "error: incompatible tree BEVE format (magic={:#010x}, version={})\n", proxy.Magic, proxy.Version);
        return std::nullopt;
    }
    std::string problem;
    auto tree = ProxiesToTree(proxy.Tree.Nodes, problem);
    if (!problem.empty()) {
        fmt::print(stderr, "serialization error (TreeFromBeve): {}\n", problem);
        return std::nullopt;
    }
    return tree;
}

auto IndividualFromBeve(std::string_view data) -> std::optional<Individual>
{
    IndividualBeveProxy proxy;
    if (auto ec = glz::read_beve(proxy, data); ec) {
        fmt::print(stderr, "serialization error (IndividualFromBeve): {}\n", glz::format_error(ec));
        return std::nullopt;
    }
    if (proxy.Magic != IndividualMagic || proxy.Version != BevePayloadVersion) {
        fmt::print(stderr, "error: incompatible individual BEVE format (magic={:#010x}, version={})\n", proxy.Magic,
            proxy.Version);
        return std::nullopt;
    }
    std::string problem;
    auto individual = ProxyToIndividual(proxy.Individual, problem);
    if (!problem.empty()) {
        fmt::print(stderr, "serialization error (IndividualFromBeve): {}\n", problem);
        return std::nullopt;
    }
    return individual;
}

// ---- Checkpoint ----

auto ToCacheCheckpoint(ZobristState const& state) -> CacheCheckpoint
{
    CacheCheckpoint cache;
    cache.TableFingerprint = state.TableFingerprint;
    cache.Clock = state.Clock;
    cache.MaxAge = state.MaxAge;
    cache.Visibility = static_cast<uint8_t>(state.Visibility);
    cache.Keys.reserve(state.Entries.size());
    cache.InsertGenerations.reserve(state.Entries.size());
    cache.Lengths.reserve(state.Entries.size());
    for (auto const& entry : state.Entries) {
        cache.Keys.push_back(entry.Key);
        cache.InsertGenerations.push_back(entry.InsertGeneration);
        if (entry.Value.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::length_error("a cache entry has more fitness values than a checkpoint can store");
        }
        cache.Lengths.push_back(static_cast<uint32_t>(entry.Value.size()));
        cache.Values.insert(cache.Values.end(), entry.Value.begin(), entry.Value.end());
    }
    return cache;
}

namespace {
    // Empty if the arrays of `cache` are consistent, else the reason. Checked arithmetic throughout: nothing here
    // trusts a count read from a file.
    auto CacheProblem(CacheCheckpoint const& cache) -> std::string
    {
        auto const n = cache.Keys.size();
        if (cache.InsertGenerations.size() != n || cache.Lengths.size() != n) {
            return "the cache arrays have different lengths";
        }
        if (n > MaxCheckpointCacheEntries) {
            return "the cache has more entries than a checkpoint can hold";
        }
        if (cache.Visibility > 1) {
            return "unknown cache visibility";
        }
        if (cache.MaxAge > std::numeric_limits<std::size_t>::max()) {
            return "the maximum cache entry age is not representable";
        }
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (cache.Lengths[i] == 0) {
                return "a cache entry has no fitness values";
            }
            if (cache.Lengths[i] > MaxCheckpointObjectives) {
                return "a cache entry has more fitness values than a checkpoint can hold";
            }
            if (total > std::numeric_limits<std::uint64_t>::max() - cache.Lengths[i]) {
                return "the cache entry lengths overflow";
            }
            total += cache.Lengths[i];
            if (i > 0 && cache.Keys[i] <= cache.Keys[i - 1]) {
                return "the cache keys are not strictly ascending";
            }
        }
        if (total != cache.Values.size()) {
            return "the cache entry lengths do not add up to the number of fitness values";
        }
        return {};
    }
} // anonymous namespace

auto ToZobristState(CacheCheckpoint const& cache) -> ZobristState
{
    if (auto const problem = CacheProblem(cache); !problem.empty()) {
        throw std::invalid_argument(problem);
    }
    ZobristState state;
    state.TableFingerprint = cache.TableFingerprint;
    state.Clock = cache.Clock;
    state.MaxAge = static_cast<std::size_t>(cache.MaxAge);
    state.Visibility = static_cast<CacheVisibility>(cache.Visibility);
    state.Entries.reserve(cache.Keys.size());
    std::size_t offset = 0; // within bounds: the lengths were checked to add up to Values.size()
    for (std::size_t i = 0; i < cache.Keys.size(); ++i) {
        auto const first = cache.Values.begin() + static_cast<std::ptrdiff_t>(offset);
        offset += cache.Lengths[i];
        state.Entries.push_back({ .Key = static_cast<Operon::Hash>(cache.Keys[i]),
            .InsertGeneration = cache.InsertGenerations[i],
            .Value
            = Operon::Vector<Operon::Scalar>(first, cache.Values.begin() + static_cast<std::ptrdiff_t>(offset)) });
    }
    return state;
}

namespace {
    auto ToProxy(Checkpoint const& cp) -> CheckpointProxy
    {
        CheckpointProxy proxy;
        proxy.RngState = cp.RngState;
        proxy.Generation = cp.Generation;
        proxy.WorkerRngStates = cp.WorkerRngStates;
        proxy.Cache = cp.Cache;
        proxy.Run = cp.Run;
        proxy.Population.reserve(cp.Population.size());
        for (auto const& ind : cp.Population) {
            proxy.Population.push_back(
                { { NodesToProxies(ind.Genotype.Nodes()) }, ind.Fitness, ind.Rank, ind.Distance });
        }
        return proxy;
    }

    // `proxy` is consumed: the arrays that need no conversion are moved, not copied (a cache can be most of a
    // gigabyte).
    auto FromProxy(CheckpointProxy&& proxy) -> std::optional<Checkpoint>
    {
        if (proxy.Magic != CheckpointMagic) {
            fmt::print(stderr, "error: not a checkpoint file (magic={:#010x})\n", proxy.Magic);
            return std::nullopt;
        }
        if (proxy.Version != CheckpointVersion) {
            fmt::print(stderr,
                "error: checkpoint format version {} is {} (this build reads version {}); a checkpoint written by "
                "another version cannot be resumed\n",
                proxy.Version, proxy.Version < CheckpointVersion ? "no longer supported" : "newer than supported",
                CheckpointVersion);
            return std::nullopt;
        }
        if (proxy.Cache) {
            if (auto const problem = CacheProblem(*proxy.Cache); !problem.empty()) {
                fmt::print(stderr, "error: corrupt checkpoint: {}\n", problem);
                return std::nullopt;
            }
        }
        if (proxy.Run && proxy.Run->Groups.size() != proxy.Run->Hashes.size()) {
            fmt::print(stderr, "error: corrupt checkpoint: the run fingerprint arrays have different lengths\n");
            return std::nullopt;
        }
        Checkpoint cp;
        cp.Population.reserve(proxy.Population.size());
        for (std::size_t i = 0; i < proxy.Population.size(); ++i) {
            auto& ip = proxy.Population[i];
            // a parent without a genotype cannot be evaluated, selected or recombined
            if (ip.Tree.Nodes.empty()) {
                fmt::print(stderr, "error: corrupt checkpoint: individual {} of the population has no tree\n", i);
                return std::nullopt;
            }
            std::string problem;
            Individual ind;
            ind.Genotype = ProxiesToTree(ip.Tree.Nodes, problem);
            if (!problem.empty()) {
                fmt::print(stderr, "error: corrupt checkpoint: individual {} of the population: {}\n", i, problem);
                return std::nullopt;
            }
            ind.Fitness = std::move(ip.Fitness);
            ind.Rank = ip.Rank;
            ind.Distance = ip.Distance;
            cp.Population.push_back(std::move(ind));
            ip = {}; // the proxy of a large population is released as it is converted
        }
        cp.RngState = proxy.RngState;
        cp.Generation = proxy.Generation;
        cp.WorkerRngStates = std::move(proxy.WorkerRngStates);
        cp.Cache = std::move(proxy.Cache);
        cp.Run = std::move(proxy.Run);
        return cp;
    }

    // Empty if the reader accepts the number of elements of every array of `cp`, else the first array that is too big.
    auto LimitProblem(Checkpoint const& cp) -> std::string
    {
        if (cp.Population.size() > MaxCheckpointPopulation) {
            return fmt::format("the population has {} individuals, at most {} can be read back", cp.Population.size(),
                MaxCheckpointPopulation);
        }
        if (cp.WorkerRngStates.size() > MaxCheckpointWorkerStates) {
            return fmt::format("there are {} worker random states, at most {} can be read back",
                cp.WorkerRngStates.size(), MaxCheckpointWorkerStates);
        }
        for (auto const& ind : cp.Population) {
            if (ind.Genotype.Length() > MaxCheckpointTreeNodes) {
                return fmt::format(
                    "a tree has {} nodes, at most {} can be read back", ind.Genotype.Length(), MaxCheckpointTreeNodes);
            }
            if (ind.Fitness.size() > MaxCheckpointObjectives) {
                return fmt::format("an individual has {} fitness values, at most {} can be read back",
                    ind.Fitness.size(), MaxCheckpointObjectives);
            }
        }
        if (cp.Cache) {
            if (auto const problem = CacheProblem(*cp.Cache); !problem.empty()) {
                return problem;
            }
        }
        if (cp.Run
            && (cp.Run->Groups.size() > MaxCheckpointFingerprintEntries
                || cp.Run->Hashes.size() > MaxCheckpointFingerprintEntries)) {
            return fmt::format("the run fingerprint has more than {} groups", MaxCheckpointFingerprintEntries);
        }
        if (cp.Run) {
            for (auto const& group : cp.Run->Groups) {
                if (group.size() > MaxCheckpointGroupNameBytes) {
                    return fmt::format("a run fingerprint group name has {} bytes, at most {} can be read back",
                        group.size(), MaxCheckpointGroupNameBytes);
                }
            }
        }
        return {};
    }
} // anonymous namespace

namespace {
    void PutLittleEndian(std::string& out, std::uint64_t value, std::size_t bytes)
    {
        for (std::size_t i = 0; i < bytes; ++i) {
            out.push_back(static_cast<char>((value >> (8U * i)) & 0xFFU));
        }
    }

    auto GetLittleEndian(std::string_view in, std::size_t offset, std::size_t bytes) -> std::uint64_t
    {
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < bytes; ++i) {
            value |= static_cast<std::uint64_t>(static_cast<unsigned char>(in[offset + i])) << (8U * i);
        }
        return value;
    }

    auto PayloadChecksum(std::string_view payload) -> std::uint64_t
    {
        return Operon::Hasher {}(reinterpret_cast<std::uint8_t const*>(payload.data()), payload.size()); // NOLINT
    }
} // namespace

auto ToBeve(Checkpoint const& cp) -> std::string
{
    try {
        auto proxy = ToProxy(cp);
        auto result = glz::write_beve(proxy);
        if (!result) {
            fmt::print(stderr, "serialization error (ToBeve Checkpoint): {}\n", glz::format_error(result.error()));
            return {};
        }
        // envelope: magic, format version, checksum of the payload (all little endian), then the payload
        std::string out;
        out.reserve(CheckpointEnvelopeBytes + result->size());
        PutLittleEndian(out, CheckpointMagic, 4);
        PutLittleEndian(out, CheckpointVersion, 4);
        PutLittleEndian(out, PayloadChecksum(*result), 8);
        out += *result;
        return out;
    } catch (std::exception const& e) {
        fmt::print(stderr, "serialization error (ToBeve Checkpoint): {}\n", e.what());
        return {};
    }
}

auto CheckpointFromBeve(std::string_view data) -> std::optional<Checkpoint>
{
    if (data.size() > MaxCheckpointFileBytes) {
        fmt::print(stderr, "error: checkpoint data of {} bytes exceeds the limit of {} bytes\n", data.size(),
            MaxCheckpointFileBytes);
        return std::nullopt;
    }
    if (data.size() < CheckpointEnvelopeBytes || GetLittleEndian(data, 0, 4) != CheckpointMagic) {
        fmt::print(stderr,
            "error: not a checkpoint file of this format (this build reads format version {}; checkpoints of "
            "other versions cannot be resumed)\n",
            CheckpointVersion);
        return std::nullopt;
    }
    if (auto const version = GetLittleEndian(data, 4, 4); version != CheckpointVersion) {
        fmt::print(stderr,
            "error: checkpoint format version {} is {} (this build reads version {}); a checkpoint written by "
            "another version cannot be resumed\n",
            version, version < CheckpointVersion ? "no longer supported" : "newer than supported", CheckpointVersion);
        return std::nullopt;
    }
    // Before anything is decoded: a damaged payload (a flipped bit, a truncated or overwritten file) is rejected as a
    // whole instead of being interpreted.
    auto const payload = data.substr(CheckpointEnvelopeBytes);
    if (GetLittleEndian(data, 8, 8) != PayloadChecksum(payload)) {
        fmt::print(stderr, "error: checkpoint is corrupt (checksum mismatch)\n");
        return std::nullopt;
    }
    try {
        CheckpointProxy proxy;
        // the reason is the error code only: the buffer is binary and is not echoed
        if (auto ec = glz::read<CheckpointReadOpts {}>(proxy, payload); ec) {
            fmt::print(stderr, "serialization error (CheckpointFromBeve): {}\n", glz::format_error(ec));
            return std::nullopt;
        }
        return FromProxy(std::move(proxy));
    } catch (std::bad_alloc const&) {
        fmt::print(stderr, "error: corrupt checkpoint: not enough memory to read it\n");
    } catch (std::length_error const&) {
        fmt::print(stderr, "error: corrupt checkpoint: it declares an array that is too large\n");
    } catch (std::exception const& e) {
        fmt::print(stderr, "error: corrupt checkpoint: {}\n", e.what());
    }
    return std::nullopt;
}

auto SaveCheckpoint(Checkpoint const& cp, std::string_view path, std::uint64_t maxFileBytes) -> bool
{
    auto const limit = std::min(maxFileBytes, MaxCheckpointFileBytes);
    auto const tmpPath = std::string(path) + ".tmp";
    std::error_code ec;
    try {
        if (auto const problem = LimitProblem(cp); !problem.empty()) {
            fmt::print(stderr, "error: checkpoint not written to '{}': {}\n", path, problem);
            return false;
        }
        auto const bytes = ToBeve(cp);
        if (bytes.empty()) {
            return false;
        }
        if (bytes.size() > limit) {
            fmt::print(stderr,
                "error: checkpoint not written to '{}': it would take {} bytes, a checkpoint of more than {} bytes "
                "cannot be read back\n",
                path, bytes.size(), limit);
            return false;
        }
        if (auto const wec = glz::buffer_to_file(bytes, tmpPath); bool(wec)) {
            fmt::print(stderr, "error writing checkpoint to '{}': {}\n", tmpPath, glz::format_error(wec));
            std::filesystem::remove(tmpPath, ec);
            return false;
        }
    } catch (std::exception const& e) {
        fmt::print(stderr, "error: checkpoint not written to '{}': {}\n", path, e.what());
        std::filesystem::remove(tmpPath, ec);
        return false;
    }
    // On POSIX, rename() atomically replaces the destination — no removal needed.
    // On Windows, rename() fails when the destination already exists as a regular
    // file; only in that specific case do we remove it and retry.  For any other
    // failure reason (permissions, I/O, cross-device) we leave the old checkpoint
    // in place rather than risk destroying it.
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
    std::error_code ec;
    auto const size = std::filesystem::file_size(std::string(path), ec);
    if (ec) {
        fmt::print(stderr, "error loading checkpoint '{}': {}\n", path, ec.message());
        return std::nullopt;
    }
    if (size > MaxCheckpointFileBytes) {
        fmt::print(stderr, "error loading checkpoint '{}': the file of {} bytes exceeds the limit of {} bytes\n", path,
            size, MaxCheckpointFileBytes);
        return std::nullopt;
    }
    std::string buf(static_cast<std::size_t>(size), '\0');
    {
        std::ifstream in(std::string(path), std::ios::binary);
        if (!in || !in.read(buf.data(), static_cast<std::streamsize>(buf.size()))
            || static_cast<std::uintmax_t>(in.gcount()) != size) {
            fmt::print(stderr, "error loading checkpoint '{}': the file could not be read\n", path);
            return std::nullopt;
        }
    }
    return CheckpointFromBeve(buf);
}

} // namespace Operon::Serialization
