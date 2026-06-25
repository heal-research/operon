// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#pragma once

#include "operon/core/types.hpp"
#include "operon/interpreter/backend/mad_eve/functions.hpp"
#include "operon/interpreter/backend/mad_eve/derivatives.hpp"

static_assert(std::is_same_v<Operon::Scalar, float>,
    "MadEve backend requires single-precision float. Build with -DUSE_SINGLE_PRECISION=ON.");
