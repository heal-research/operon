// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_VARIABLE_HPP
#define OPERON_VARIABLE_HPP

#include "types.hpp"

namespace Operon {
struct Variable {
    std::string Name;
    Operon::Hash Hash{0};
    int64_t Index{0};

    constexpr auto operator==(Variable const& rhs) const noexcept -> bool {
        return std::tie(Name, Hash, Index) == std::tie(rhs.Name, rhs.Hash, rhs.Index);
    }

    constexpr auto operator!=(Variable const& rhs) const noexcept -> bool {
        return !(*this == rhs);
    }
};
} // namespace Operon

#endif
