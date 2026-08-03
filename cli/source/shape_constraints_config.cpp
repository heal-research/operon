// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "shape_constraints_config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <fmt/format.h>
#include <glaze/glaze.hpp>

namespace Operon {

namespace {

using Json = glz::generic_i64;

// generic_i64 parses a bare-integer JSON literal (e.g. `-1`, no decimal
// point) as int64_t, not double (that's the whole reason probes_config.cpp
// and this file use generic_i64 over plain generic) — calling .get<double>()
// on such a node throws (std::get on the wrong variant alternative), so
// every numeric read here must check holds<std::int64_t>() first, exactly
// like probes_config.cpp's own ToParamValue/ToCount already do.
auto ToNumber(Json const& v) -> double
{
    return v.holds<std::int64_t>() ? static_cast<double>(v.get<std::int64_t>()) : v.get<double>();
}

auto ToDomainBound(Json const& arr, char const* context) -> std::pair<Operon::Scalar, Operon::Scalar>
{
    if (!arr.is_array()) {
        throw std::runtime_error(fmt::format("shape-constraints config: {} must be a [lo, hi] number pair", context));
    }
    auto const items = arr.get_array();
    if (items.size() != 2 || !items[0].is_number() || !items[1].is_number()) {
        throw std::runtime_error(fmt::format("shape-constraints config: {} must be a [lo, hi] number pair", context));
    }
    return { static_cast<Operon::Scalar>(ToNumber(items[0])), static_cast<Operon::Scalar>(ToNumber(items[1])) };
}

auto RequireString(Json const& obj, char const* field, char const* context) -> std::string
{
    if (!obj.contains(field) || !obj.at(field).is_string()) {
        throw std::runtime_error(fmt::format("shape-constraints config: {} requires a string '{}'", context, field));
    }
    return obj.at(field).get<std::string>();
}

auto ParseOp(Json const& entry, std::string const& opStr) -> std::pair<ShapeConstraintOp, std::string>
{
    if (opStr == "id") { return {ShapeConstraintOp::Identity, ""}; }
    if (opStr != "derivative") {
        throw std::runtime_error(fmt::format(
            "shape-constraints config: unrecognized op '{}' (expected 'id' or 'derivative')", opStr));
    }

    auto const variable = RequireString(entry, "variable", "derivative constraint");
    if (!entry.contains("order") || !entry.at("order").is_number()) {
        throw std::runtime_error("shape-constraints config: derivative constraint requires an integer 'order'");
    }
    auto const raw = ToNumber(entry.at("order"));
    auto const order = static_cast<int>(raw);
    if (static_cast<double>(order) != raw || (order != 1 && order != 2)) {
        throw std::runtime_error(fmt::format(
            "shape-constraints config: derivative constraint has 'order' {} (must be exactly 1 or 2)", raw));
    }
    return {order == 1 ? ShapeConstraintOp::FirstDerivative : ShapeConstraintOp::SecondDerivative, variable};
}

} // namespace

auto LoadShapeConstraints(std::string const& path) -> std::optional<ShapeConstraintSet>
{
    if (path.empty()) { return std::nullopt; }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(fmt::format("shape-constraints config: could not open '{}'", path));
    }
    std::stringstream buf;
    buf << in.rdbuf();
    auto const text = buf.str();

    Json doc;
    if (auto ec = glz::read_json(doc, text); ec) {
        throw std::runtime_error(fmt::format("shape-constraints config '{}': {}", path, glz::format_error(ec, text)));
    }

    if (!doc.is_object()) {
        throw std::runtime_error("shape-constraints config: top-level JSON value must be an object");
    }

    ShapeConstraintSet set;

    if (doc.contains("domains")) {
        auto const& domains = doc.at("domains");
        if (!domains.is_object()) {
            throw std::runtime_error("shape-constraints config: 'domains' must be an object mapping variable names to [lo, hi] number pairs");
        }
        for (auto const& [name, bound] : domains.get_object()) {
            set.Domains.insert_or_assign(name, ToDomainBound(bound, fmt::format("domain '{}'", name).c_str()));
        }
    }

    if (doc.contains("constraints")) {
        auto const& constraints = doc.at("constraints");
        if (!constraints.is_array()) {
            throw std::runtime_error("shape-constraints config: 'constraints' must be an array of constraint entries");
        }
        for (auto const& entry : constraints.get_array()) {
            if (!entry.is_object()) {
                throw std::runtime_error("shape-constraints config: each constraint entry must be an object");
            }
            auto const opStr = RequireString(entry, "op", "each constraint entry");
            auto [op, variable] = ParseOp(entry, opStr);

            ShapeConstraint c;
            c.Op = op;
            c.Variable = std::move(variable);

            bool const hasSign = entry.contains("sign");
            bool const hasBound = entry.contains("bound");
            if (hasSign == hasBound) {
                throw std::runtime_error(fmt::format(
                    "shape-constraints config: constraint '{}' must set exactly one of 'sign' or 'bound'", opStr));
            }
            if (hasSign) {
                if (!entry.at("sign").is_number()) {
                    throw std::runtime_error(fmt::format(
                        "shape-constraints config: constraint '{}' has non-numeric 'sign' (must be exactly 1 or -1)", opStr));
                }
                auto const raw = ToNumber(entry.at("sign"));
                auto const s = static_cast<int>(raw);
                // Reject non-integral values (e.g. 1.9) rather than
                // silently truncating them to a valid-looking 1 or -1.
                if (static_cast<double>(s) != raw || (s != 1 && s != -1)) {
                    throw std::runtime_error(fmt::format(
                        "shape-constraints config: constraint '{}' has 'sign' {} (must be exactly 1 or -1)", opStr, raw));
                }
                c.Sign = s;
            } else {
                c.Bound = ToDomainBound(entry.at("bound"), fmt::format("constraint '{}' 'bound'", opStr).c_str());
            }

            set.Constraints.push_back(std::move(c));
        }
    }

    return set;
}

} // namespace Operon
