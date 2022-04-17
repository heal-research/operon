// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_HASH_HPP
#define OPERON_HASH_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include "operon/core/constants.hpp"
#include "operon/operon_export.hpp"

namespace Operon {
    struct OPERON_EXPORT Hasher {
        auto operator()(uint8_t const* key, size_t len) noexcept -> uint64_t;
        auto operator()(std::string const& key) noexcept -> uint64_t;
    };
} // namespace Operon
#endif

