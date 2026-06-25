// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_AUTODIFF_FORWARD_DUAL_HPP
#define OPERON_AUTODIFF_FORWARD_DUAL_HPP

#include "operon/core/types.hpp"

#include "operon/ceres/jet.h"

namespace Operon {
using Dual = ceres::Jet<Operon::Scalar, 4 * sizeof(double) / sizeof(Scalar)>;
} // namespace Operon

#endif
