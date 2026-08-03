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

// Parses an "op" string into (ShapeConstraintOp, variable name). "id" has
// no variable; "d/d<var>" and "d2/d<var>2" carry <var> literally between
// the fixed prefix/suffix (no attempt to disambiguate a variable name
// that itself ends in a digit against the "d2/...2" marker -- none of
// this codebase's problem set needs one).
auto ParseOp(std::string const& op) -> std::pair<ShapeConstraintOp, std::string>
{
    if (op == "id") { return {ShapeConstraintOp::Identity, ""}; }
    if (op.starts_with("d/d") && op.size() > 3) {
        return {ShapeConstraintOp::FirstDerivative, op.substr(3)};
    }
    if (op.starts_with("d2/d") && op.ends_with('2') && op.size() > 5) {
        return {ShapeConstraintOp::SecondDerivative, op.substr(4, op.size() - 5)};
    }
    throw std::runtime_error(fmt::format(
        "shape-constraints config: unrecognized op '{}' (expected 'id', 'd/d<var>', or 'd2/d<var>2')", op));
}

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

    ShapeConstraintSet set;

    if (doc.contains("domains")) {
        for (auto const& [name, bound] : doc.at("domains").get_object()) {
            set.Domains.insert_or_assign(name, ToDomainBound(bound, fmt::format("domain '{}'", name).c_str()));
        }
    }

    if (doc.contains("constraints")) {
        for (auto const& entry : doc.at("constraints").get_array()) {
            auto const opStr = RequireString(entry, "op", "each constraint entry");
            auto [op, variable] = ParseOp(opStr);

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
