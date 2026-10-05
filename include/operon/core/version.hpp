// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_VERSION_HPP
#define OPERON_VERSION_HPP

#include "operon/operon_export.hpp"
#include <chrono>
#include <fmt/format.h>
#include <string>

namespace Operon {
// NOLINTBEGIN
auto OPERON_EXPORT Version() -> std::string;
} // namespace Operon

#endif
