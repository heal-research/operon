// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "shape_constraints_config.hpp"

#include <fstream>
#include <sstream>
#include <utility>

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
auto ConfigurationError(std::string message) -> Cli::Error
{
    return {Cli::ErrorCode::Configuration, "shape-constraints config", std::move(message)};
}

auto InputError(std::string message) -> Cli::Error
{
    return {Cli::ErrorCode::Input, "shape-constraints config", std::move(message)};
}

auto ToNumber(Json const& v) -> double
{
    return v.holds<std::int64_t>() ? static_cast<double>(v.get<std::int64_t>()) : v.get<double>();
}

auto ToDomainBound(Json const& arr, char const* context) -> Cli::Result<std::pair<Operon::Scalar, Operon::Scalar>>
{
    if (!arr.is_array()) {
        return tl::unexpected(ConfigurationError(fmt::format("{} must be a [lo, hi] number pair", context)));
    }
    auto const items = arr.get_array();
    if (items.size() != 2 || !items[0].is_number() || !items[1].is_number()) {
        return tl::unexpected(ConfigurationError(fmt::format("{} must be a [lo, hi] number pair", context)));
    }
    return std::pair{
        static_cast<Operon::Scalar>(ToNumber(items[0])),
        static_cast<Operon::Scalar>(ToNumber(items[1]))
    };
}

auto RequireString(Json const& obj, char const* field, char const* context) -> Cli::Result<std::string>
{
    if (!obj.contains(field) || !obj.at(field).is_string()) {
        return tl::unexpected(ConfigurationError(fmt::format("{} requires a string '{}'", context, field)));
    }
    return obj.at(field).get<std::string>();
}

auto ParseOp(Json const& entry, std::string const& opStr) -> Cli::Result<std::pair<ShapeConstraintOp, std::string>>
{
    if (opStr == "id") { return std::pair{ShapeConstraintOp::Identity, std::string{}}; }
    if (opStr != "derivative") {
        return tl::unexpected(ConfigurationError(fmt::format(
            "unrecognized op '{}' (expected 'id' or 'derivative')", opStr)));
    }

    auto variable = RequireString(entry, "variable", "derivative constraint");
    if (!variable) { return tl::unexpected(std::move(variable.error())); }
    if (!entry.contains("order") || !entry.at("order").is_number()) {
        return tl::unexpected(ConfigurationError("derivative constraint requires an integer 'order'"));
    }
    auto const raw = ToNumber(entry.at("order"));
    auto const order = static_cast<int>(raw);
    if (static_cast<double>(order) != raw || (order != 1 && order != 2)) {
        return tl::unexpected(ConfigurationError(fmt::format(
            "derivative constraint has 'order' {} (must be exactly 1 or 2)", raw)));
    }
    return std::pair{order == 1 ? ShapeConstraintOp::FirstDerivative : ShapeConstraintOp::SecondDerivative, std::move(*variable)};
}

} // namespace

auto LoadShapeConstraints(std::string const& path) -> Cli::Result<std::optional<ShapeConstraintSet>>
{
    if (path.empty()) { return std::optional<ShapeConstraintSet>{}; }

    std::ifstream in(path);
    if (!in) {
        return tl::unexpected(InputError(fmt::format("could not open '{}'", path)));
    }
    std::stringstream buf;
    buf << in.rdbuf();
    auto const text = buf.str();
    if (in.bad()) {
        return tl::unexpected(InputError(fmt::format("could not read '{}'", path)));
    }

    Json doc;
    if (auto ec = glz::read_json(doc, text); ec) {
        return tl::unexpected(ConfigurationError(fmt::format("'{}': {}", path, glz::format_error(ec, text))));
    }

    if (!doc.is_object()) {
        return tl::unexpected(ConfigurationError("top-level JSON value must be an object"));
    }

    ShapeConstraintSet set;

    if (doc.contains("domains")) {
        auto const& domains = doc.at("domains");
        if (!domains.is_object()) {
            return tl::unexpected(ConfigurationError("'domains' must be an object mapping variable names to [lo, hi] number pairs"));
        }
        for (auto const& [name, bound] : domains.get_object()) {
            auto domain = ToDomainBound(bound, fmt::format("domain '{}'", name).c_str());
            if (!domain) { return tl::unexpected(std::move(domain.error())); }
            set.Domains.insert_or_assign(name, std::move(*domain));
        }
    }

    if (doc.contains("constraints")) {
        auto const& constraints = doc.at("constraints");
        if (!constraints.is_array()) {
            return tl::unexpected(ConfigurationError("'constraints' must be an array of constraint entries"));
        }
        for (auto const& entry : constraints.get_array()) {
            if (!entry.is_object()) {
                return tl::unexpected(ConfigurationError("each constraint entry must be an object"));
            }
            auto opStr = RequireString(entry, "op", "each constraint entry");
            if (!opStr) { return tl::unexpected(std::move(opStr.error())); }
            auto parsedOp = ParseOp(entry, *opStr);
            if (!parsedOp) { return tl::unexpected(std::move(parsedOp.error())); }
            auto [op, variable] = std::move(*parsedOp);

            ShapeConstraint c;
            c.Op = op;
            c.Variable = std::move(variable);

            bool const hasSign = entry.contains("sign");
            bool const hasBound = entry.contains("bound");
            if (hasSign == hasBound) {
                return tl::unexpected(ConfigurationError(fmt::format(
                    "constraint '{}' must set exactly one of 'sign' or 'bound'", *opStr)));
            }
            if (hasSign) {
                if (!entry.at("sign").is_number()) {
                    return tl::unexpected(ConfigurationError(fmt::format(
                        "constraint '{}' has non-numeric 'sign' (must be exactly 1 or -1)", *opStr)));
                }
                auto const raw = ToNumber(entry.at("sign"));
                auto const s = static_cast<int>(raw);
                // Reject non-integral values (e.g. 1.9) rather than
                // silently truncating them to a valid-looking 1 or -1.
                if (static_cast<double>(s) != raw || (s != 1 && s != -1)) {
                    return tl::unexpected(ConfigurationError(fmt::format(
                        "constraint '{}' has 'sign' {} (must be exactly 1 or -1)", *opStr, raw)));
                }
                c.Sign = s;
            } else {
                auto bound = ToDomainBound(entry.at("bound"), fmt::format("constraint '{}' 'bound'", *opStr).c_str());
                if (!bound) { return tl::unexpected(std::move(bound.error())); }
                c.Bound = std::move(*bound);
            }

            set.Constraints.push_back(std::move(c));
        }
    }

    return std::optional<ShapeConstraintSet>{std::move(set)};
}

} // namespace Operon
