// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <array>
#include <fmt/format.h>
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

auto ToOperonNode(infix_parser::node const& a) -> Operon::Node
{
    if (a.type == infix_parser::node_type::constant) {
        return Operon::Node::Constant(a.value);
    }
    if (a.type == infix_parser::node_type::variable) {
        return Operon::Node(Operon::NodeType::Variable, Operon::Hasher{}(a.name));
    }
    auto const op = node_type_map.at(static_cast<std::size_t>(a.type));
    if (op == Operon::NoBuiltinOp) {
        throw std::runtime_error(fmt::format("unsupported node type: {}", static_cast<int>(a.type)));
    }
    return Operon::Node::Function(static_cast<Operon::Hash>(op), a.arity);
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
