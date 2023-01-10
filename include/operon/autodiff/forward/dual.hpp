// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_AUTODIFF_FORWARD_DUAL_HPP
#define OPERON_AUTODIFF_FORWARD_DUAL_HPP

#include "operon/core/types.hpp"

#if defined(HAVE_CERES)
#include <ceres/jet.h>
#else
#include "operon/ceres/jet.h"
#endif

namespace Operon {
using Dual = ceres::Jet<Operon::Scalar, 4 * sizeof(double) / sizeof(Scalar)>;
} // namespace Operon

#endif
