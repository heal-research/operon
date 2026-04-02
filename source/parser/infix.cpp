// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <array>
#include <fmt/core.h>
#include <iterator>
#include <stdexcept>
#include <string>

#include "operon/parser/infix.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace {

// Build a compile-time lookup table from infix_parser::node_type to Operon::NodeType.
// Entries without an Operon equivalent are set to the NoType sentinel.
constexpr auto MakeNodeTypeMap()
{
    constexpr auto count = static_cast<std::size_t>(infix_parser::node_type::count);
    std::array<Operon::NodeType, count> map{};
    map.fill(Operon::NodeTypes::NoType);

    map[static_cast<std::size_t>(infix_parser::node_type::add)]      = Operon::NodeType::Add;
    map[static_cast<std::size_t>(infix_parser::node_type::sub)]      = Operon::NodeType::Sub;
    map[static_cast<std::size_t>(infix_parser::node_type::mul)]      = Operon::NodeType::Mul;
    map[static_cast<std::size_t>(infix_parser::node_type::div)]      = Operon::NodeType::Div;
    map[static_cast<std::size_t>(infix_parser::node_type::pow)]      = Operon::NodeType::Pow;
    map[static_cast<std::size_t>(infix_parser::node_type::abs)]      = Operon::NodeType::Abs;
    map[static_cast<std::size_t>(infix_parser::node_type::square)]   = Operon::NodeType::Square;
    map[static_cast<std::size_t>(infix_parser::node_type::exp)]      = Operon::NodeType::Exp;
    map[static_cast<std::size_t>(infix_parser::node_type::log)]      = Operon::NodeType::Log;
    map[static_cast<std::size_t>(infix_parser::node_type::sin)]      = Operon::NodeType::Sin;
    map[static_cast<std::size_t>(infix_parser::node_type::cos)]      = Operon::NodeType::Cos;
    map[static_cast<std::size_t>(infix_parser::node_type::tan)]      = Operon::NodeType::Tan;
    map[static_cast<std::size_t>(infix_parser::node_type::asin)]     = Operon::NodeType::Asin;
    map[static_cast<std::size_t>(infix_parser::node_type::acos)]     = Operon::NodeType::Acos;
    map[static_cast<std::size_t>(infix_parser::node_type::atan)]     = Operon::NodeType::Atan;
    map[static_cast<std::size_t>(infix_parser::node_type::sinh)]     = Operon::NodeType::Sinh;
    map[static_cast<std::size_t>(infix_parser::node_type::cosh)]     = Operon::NodeType::Cosh;
    map[static_cast<std::size_t>(infix_parser::node_type::tanh)]     = Operon::NodeType::Tanh;
    map[static_cast<std::size_t>(infix_parser::node_type::sqrt)]     = Operon::NodeType::Sqrt;
    map[static_cast<std::size_t>(infix_parser::node_type::cbrt)]     = Operon::NodeType::Cbrt;
    map[static_cast<std::size_t>(infix_parser::node_type::log1p)]    = Operon::NodeType::Log1p;
    map[static_cast<std::size_t>(infix_parser::node_type::logabs)]   = Operon::NodeType::Logabs;
    map[static_cast<std::size_t>(infix_parser::node_type::sqrtabs)]  = Operon::NodeType::Sqrtabs;
    map[static_cast<std::size_t>(infix_parser::node_type::aq)]       = Operon::NodeType::Aq;
    map[static_cast<std::size_t>(infix_parser::node_type::fmin)]     = Operon::NodeType::Fmin;
    map[static_cast<std::size_t>(infix_parser::node_type::fmax)]     = Operon::NodeType::Fmax;
    map[static_cast<std::size_t>(infix_parser::node_type::powabs)]   = Operon::NodeType::Powabs;

    return map;
}

constexpr auto node_type_map = MakeNodeTypeMap();

auto ToOperonNode(infix_parser::node const& a) -> Operon::Node
{
    if (a.type == infix_parser::node_type::constant) {
        return Operon::Node::Constant(a.value);
    }
    if (a.type == infix_parser::node_type::variable) {
        return Operon::Node(Operon::NodeType::Variable, Operon::Hasher{}(a.name));
    }
    auto const operonType = node_type_map.at(static_cast<std::size_t>(a.type));
    if (operonType == Operon::NodeTypes::NoType) {
        throw std::runtime_error(fmt::format("unsupported node type: {}", static_cast<int>(a.type)));
    }
    Operon::Node node(operonType);
    node.Arity = a.arity;
    node.Length = a.arity;
    return node;
}

} // anonymous namespace

namespace Operon {

auto InfixParser::Parse(std::string_view infix, bool reduce) -> Tree
{
    auto result = infix_parser::parse(infix);
    if (auto const* err = std::get_if<infix_parser::parse_error>(&result)) {
        throw std::invalid_argument(fmt::format("parse error at position {}: {}", err->position, err->message));
    }
    auto const& expr = std::get<infix_parser::expression>(result);

    Operon::Vector<Operon::Node> nodes;
    nodes.reserve(expr.size());

    for (auto const& a : expr) {
        nodes.push_back(ToOperonNode(a));
    }

    Operon::Tree tree{nodes};
    tree.UpdateNodes();
    if (reduce) { tree.Reduce(); }
    return tree;
}

auto InfixParser::Parse(std::string_view infix, Dataset const& dataset, bool reduce) -> Tree
{
    auto tree = Parse(infix, reduce);
    for (auto const& node : tree.Nodes()) {
        if (node.IsVariable() && !dataset.GetVariable(node.HashValue).has_value()) {
            throw std::invalid_argument(fmt::format("variable with hash {} not found in dataset", node.HashValue));
        }
    }
    return tree;
}

} // namespace Operon
