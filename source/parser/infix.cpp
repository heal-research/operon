// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <array>
#include <fmt/format.h>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "operon/parser/infix.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace {

// Build a compile-time lookup table from infix_parser::node_type to
// Operon::BuiltinOp. Entries without an Operon equivalent are set to the
// NoBuiltinOp sentinel.
constexpr auto MakeBuiltinOpMap()
{
    constexpr auto count = static_cast<std::size_t>(infix_parser::node_type::count);
    std::array<Operon::BuiltinOp, count> map{};
    map.fill(Operon::NoBuiltinOp);

    map[static_cast<std::size_t>(infix_parser::node_type::add)]      = Operon::BuiltinOp::Add;
    map[static_cast<std::size_t>(infix_parser::node_type::sub)]      = Operon::BuiltinOp::Sub;
    map[static_cast<std::size_t>(infix_parser::node_type::mul)]      = Operon::BuiltinOp::Mul;
    map[static_cast<std::size_t>(infix_parser::node_type::div)]      = Operon::BuiltinOp::Div;
    map[static_cast<std::size_t>(infix_parser::node_type::pow)]      = Operon::BuiltinOp::Pow;
    map[static_cast<std::size_t>(infix_parser::node_type::abs)]      = Operon::BuiltinOp::Abs;
    map[static_cast<std::size_t>(infix_parser::node_type::square)]   = Operon::BuiltinOp::Square;
    map[static_cast<std::size_t>(infix_parser::node_type::exp)]      = Operon::BuiltinOp::Exp;
    map[static_cast<std::size_t>(infix_parser::node_type::log)]      = Operon::BuiltinOp::Log;
    map[static_cast<std::size_t>(infix_parser::node_type::sin)]      = Operon::BuiltinOp::Sin;
    map[static_cast<std::size_t>(infix_parser::node_type::cos)]      = Operon::BuiltinOp::Cos;
    map[static_cast<std::size_t>(infix_parser::node_type::tan)]      = Operon::BuiltinOp::Tan;
    map[static_cast<std::size_t>(infix_parser::node_type::asin)]     = Operon::BuiltinOp::Asin;
    map[static_cast<std::size_t>(infix_parser::node_type::acos)]     = Operon::BuiltinOp::Acos;
    map[static_cast<std::size_t>(infix_parser::node_type::atan)]     = Operon::BuiltinOp::Atan;
    map[static_cast<std::size_t>(infix_parser::node_type::sinh)]     = Operon::BuiltinOp::Sinh;
    map[static_cast<std::size_t>(infix_parser::node_type::cosh)]     = Operon::BuiltinOp::Cosh;
    map[static_cast<std::size_t>(infix_parser::node_type::tanh)]     = Operon::BuiltinOp::Tanh;
    map[static_cast<std::size_t>(infix_parser::node_type::sqrt)]     = Operon::BuiltinOp::Sqrt;
    map[static_cast<std::size_t>(infix_parser::node_type::cbrt)]     = Operon::BuiltinOp::Cbrt;
    map[static_cast<std::size_t>(infix_parser::node_type::log1p)]    = Operon::BuiltinOp::Log1p;
    map[static_cast<std::size_t>(infix_parser::node_type::logabs)]   = Operon::BuiltinOp::Logabs;
    map[static_cast<std::size_t>(infix_parser::node_type::sqrtabs)]  = Operon::BuiltinOp::Sqrtabs;
    map[static_cast<std::size_t>(infix_parser::node_type::aq)]       = Operon::BuiltinOp::Aq;
    map[static_cast<std::size_t>(infix_parser::node_type::fmin)]     = Operon::BuiltinOp::Fmin;
    map[static_cast<std::size_t>(infix_parser::node_type::fmax)]     = Operon::BuiltinOp::Fmax;
    map[static_cast<std::size_t>(infix_parser::node_type::powabs)]   = Operon::BuiltinOp::Powabs;

    return map;
}

constexpr auto node_type_map = MakeBuiltinOpMap();
struct ParsedSubtree {
    infix_parser::expression Nodes;
    std::optional<Operon::Scalar> VariableWeight;
};

auto ToOperonNode(infix_parser::node const& a, std::optional<Operon::Scalar> variableWeight = {})
    -> tl::expected<Operon::Node, Operon::InfixParseError>
{
    if (a.type == infix_parser::node_type::constant) {
        return Operon::Node::Constant(a.value);
    }
    if (a.type == infix_parser::node_type::variable) {
        auto node = Operon::Node(Operon::NodeType::Variable, Operon::Hasher{}(a.name));
        node.Value = variableWeight.value_or(Operon::Scalar{1});
        return node;
    }
    auto const op = node_type_map.at(static_cast<std::size_t>(a.type));
    if (op == Operon::NoBuiltinOp) {
        return tl::unexpected(Operon::InfixParseError{
            fmt::format("unsupported expression node type: {}", static_cast<int>(a.type))});
    }
    return Operon::Node::Function(static_cast<Operon::Hash>(op), a.arity);
}
auto MaterializeWeight(ParsedSubtree& subtree) -> void
{
    if (!subtree.VariableWeight) { return; }
    subtree.Nodes.push_back(infix_parser::node::constant(static_cast<double>(*subtree.VariableWeight)));
    subtree.Nodes.push_back(infix_parser::node::function(infix_parser::node_type::mul, 2));
    subtree.VariableWeight.reset();
}

auto FoldWeightedProducts(infix_parser::expression const& expr, bool enabled) -> std::vector<ParsedSubtree>
{
    using infix_parser::node_type;
    std::vector<ParsedSubtree> stack;
    stack.reserve(expr.size());
    for (auto const& item : expr) {
        if (item.arity == 0) {
            stack.push_back({{item}, std::nullopt});
            continue;
        }
        if (item.type == node_type::mul && item.arity == 2 && stack.size() >= 2) {
            auto lhs = std::move(stack.back()); stack.pop_back();
            auto rhs = std::move(stack.back()); stack.pop_back();
            auto fold = [&](ParsedSubtree& target, ParsedSubtree const& coefficient) {
                if (!enabled || coefficient.Nodes.size() != 1 || coefficient.Nodes[0].type != node_type::constant
                    || target.Nodes.size() != 1 || target.Nodes[0].type != node_type::variable) { return false; }
                target.VariableWeight = target.VariableWeight.value_or(Operon::Scalar{1})
                    * static_cast<Operon::Scalar>(coefficient.Nodes[0].value);
                stack.push_back(std::move(target));
                return true;
            };
            if (fold(lhs, rhs) || fold(rhs, lhs)) { continue; }
            MaterializeWeight(rhs);
            MaterializeWeight(lhs);
            rhs.Nodes.insert(rhs.Nodes.end(), std::make_move_iterator(lhs.Nodes.begin()), std::make_move_iterator(lhs.Nodes.end()));
            rhs.Nodes.push_back(item);
            stack.push_back(std::move(rhs));
            continue;
        }
        std::vector<ParsedSubtree> children;
        children.reserve(item.arity);
        for (std::size_t child = 0; child < item.arity; ++child) {
            children.push_back(std::move(stack.back()));
            stack.pop_back();
        }
        auto merged = ParsedSubtree{};
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
            MaterializeWeight(*it);
            merged.Nodes.insert(merged.Nodes.end(), std::make_move_iterator(it->Nodes.begin()), std::make_move_iterator(it->Nodes.end()));
        }
        merged.Nodes.push_back(item);
        stack.push_back(std::move(merged));
    }
    return stack;
}


} // anonymous namespace
namespace Operon {

auto InfixParser::Parse(std::string_view infix, InfixParseOptions options) -> tl::expected<Tree, InfixParseError>
{
    auto result = infix_parser::parse(infix);
    if (auto const* err = std::get_if<infix_parser::parse_error>(&result)) {
        return tl::unexpected(InfixParseError{
            fmt::format("parse error at position {}: {}", err->position, err->message)});
    }
    auto subtrees = FoldWeightedProducts(std::get<infix_parser::expression>(result), options.FoldVariableWeights);
    Operon::Vector<Operon::Node> nodes;
    for (auto& subtree : subtrees) {
        for (std::size_t i = 0; i < subtree.Nodes.size(); ++i) {
            auto weight = i + 1 == subtree.Nodes.size() ? subtree.VariableWeight : std::optional<Operon::Scalar>{};
            auto node = ToOperonNode(subtree.Nodes[i], weight);
            if (!node) { return tl::unexpected(std::move(node.error())); }
            nodes.push_back(std::move(*node));
        }
    }
    Operon::Tree tree{nodes};
    tree.UpdateNodes();
    if (options.Reduce) { tree.Reduce(); }
    return tree;
}

auto InfixParser::Parse(std::string_view infix, Dataset const& dataset, InfixParseOptions options) -> tl::expected<Tree, InfixParseError>
{
    auto tree = Parse(infix, options);
    if (!tree) { return tl::unexpected(tree.error()); }
    for (auto const& node : tree->Nodes()) {
        if (node.IsVariable() && !dataset.GetVariable(node.HashValue).has_value()) {
            return tl::unexpected(InfixParseError{
                fmt::format("variable with hash {} not found in dataset", node.HashValue)});
        }
    }
    return tree;
}

auto InfixParser::ParseOrThrow(std::string_view infix, InfixParseOptions options) -> Tree
{
    auto tree = Parse(infix, options);
    if (!tree) { throw std::invalid_argument(tree.error().Message); }
    return std::move(*tree);
}

auto InfixParser::ParseOrThrow(std::string_view infix, Dataset const& dataset, InfixParseOptions options) -> Tree
{
    auto tree = Parse(infix, dataset, options);
    if (!tree) { throw std::invalid_argument(tree.error().Message); }
    return std::move(*tree);
}

auto InfixParser::ParseFunctionBody(std::string_view infix, std::span<std::string const> params,
                                    InfixParseOptions options) -> tl::expected<Tree, InfixParseError>
{
    if (params.size() > Operon::kMaxComposedFunctionArity) {
        return tl::unexpected(InfixParseError{fmt::format(
            "composed function has {} parameters, exceeding the v1 cap of {}",
            params.size(), Operon::kMaxComposedFunctionArity)});
    }

    auto treeResult = Parse(infix, options);
    if (!treeResult) { return tl::unexpected(std::move(treeResult.error())); }
    auto tree = std::move(*treeResult);

    Operon::Vector<Operon::Hash> paramHashes(params.size());
    std::ranges::transform(params, paramHashes.begin(), [](auto const& name) { return Operon::Hasher{}(name); });
    std::vector<bool> used(params.size(), false);

    for (auto& node : tree.Nodes()) {
        if (node.Type == Operon::NodeType::Constant) {
            node.Optimize = false;
            continue;
        }
        if (!node.IsVariable()) { continue; }
        auto it = std::ranges::find(paramHashes, node.HashValue);
        if (it == paramHashes.end()) {
            return tl::unexpected(InfixParseError{fmt::format("undeclared identifier in function body (hash {})", node.HashValue)});
        }
        auto const idx = static_cast<std::size_t>(std::distance(paramHashes.begin(), it));
        node.HashValue = node.CalculatedHashValue = Operon::ParamHash(idx);
        used[idx] = true;
    }

    for (std::size_t i = 0; i < params.size(); ++i) {
        if (!used[i]) {
            return tl::unexpected(InfixParseError{fmt::format("unused parameter '{}' in composed function body", params[i])});
        }
    }
    tree.UpdateNodes();
    return tree;
}

} // namespace Operon
