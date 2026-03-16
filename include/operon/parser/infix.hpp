// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_PARSER_HPP
#define OPERON_PARSER_HPP

#include <infix-parser/parser.hpp>
#include <string_view>
#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon {

struct OPERON_EXPORT InfixParser {
    static auto Parse(std::string_view infix, bool reduce = false) -> Tree;
    static auto Parse(std::string_view infix, Dataset const& dataset, bool reduce = false) -> Tree;
};
} // namespace Operon

#endif
