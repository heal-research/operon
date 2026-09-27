// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_MEMORY_VIEW_HPP
#define OPERON_MEMORY_VIEW_HPP

#include <cstddef>
#include <span>

#include "operon/core/types.hpp"

namespace Operon {

using MemoryIndex = std::size_t;
using AccumulationScalar = double;

using ScalarSpan = std::span<Scalar>;
using ConstScalarSpan = std::span<Scalar const>;

using ScalarMatrixView = std::mdspan<Scalar,
    std::dextents<MemoryIndex, 2>, std::layout_stride>;
using ConstScalarMatrixView = std::mdspan<Scalar const,
    std::dextents<MemoryIndex, 2>, std::layout_stride>;

/**
 * Canonical logical matrix contract for public numerical APIs.
 *
 * Access is always (row, column); extents are logical dimensions and strides
 * are expressed in Scalar elements. Callers own the referenced storage and
 * must keep it alive for the duration of the call. Implementations must not
 * assume row-major or column-major storage unless a narrower API says so.
 * Output contents after an error are indeterminate unless documented
 * otherwise by the consuming API.
 */
struct MatrixViewContract {
    static constexpr MemoryIndex Rank {2};
};

} // namespace Operon

#endif
