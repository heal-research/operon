// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include "core/operator.hpp"

namespace Operon {
// wraps a creator and generates trees from a given size distribution
    using UniformInitializer = InitializerBase<std::uniform_int_distribution<size_t>>;
    using NormalInitializer = InitializerBase<std::normal_distribution<>>;
}
#endif
