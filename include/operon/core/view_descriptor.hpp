// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_VIEW_DESCRIPTOR_HPP
#define OPERON_VIEW_DESCRIPTOR_HPP

#include <array>
#include <cstdint>

#include <gsl/pointers>
#include <tl/expected.hpp>

#include "operon/core/memory_view.hpp"
#include "operon/core/view_descriptor.h"

namespace Operon {

/**
 * Constructs `Operon::ConstScalarMatrixView` from a borrowed
 * `OperonViewDescriptor`. This is the C++-only counterpart to
 * `operon_view_validate`: the descriptor itself carries no C++ types, but
 * building an `mdspan` over it requires `Operon::Scalar`, `std::mdspan`, and
 * `std::layout_stride`, none of which the plain-C header may depend on.
 *
 * `desc.scalar_code`/`desc.element_size` must match `Operon::Scalar`
 * (`OPERON_SCALAR_F32` when built with `USE_SINGLE_PRECISION`, otherwise
 * `OPERON_SCALAR_F64`) -- a descriptor for the other precision is a scalar
 * mismatch, not merely a differently-typed view, and is rejected the same
 * way as any other invalid descriptor.
 *
 * `desc.rank` must be exactly 2: `operon_view_validate` alone accepts rank 1
 * (a general C descriptor concern), but a rank-1 descriptor leaves
 * `extents[1]`/`byte_strides[1]` outside its own validated range -- this
 * matrix-specific constructor must not read them.
 *
 * Negative `byte_strides` (reversed axes) are structurally valid per
 * `operon_view_validate`, but this constructor does not yet support them:
 * `Operon::ScalarMatrixView`'s index type is unsigned, and casting a
 * negative byte stride into it would silently wrap into a huge positive
 * stride rather than a reversed one. Until a signed-offset view type exists,
 * a negative stride here is rejected as malformed input
 * (`OPERON_VIEW_ERR_STRIDE`).
 */
[[nodiscard]] auto MakeConstMatrixView(OperonViewDescriptor const& desc) -> tl::expected<ConstScalarMatrixView, OperonViewStatus>;

/**
 * Same as `MakeConstMatrixView`, for a caller that holds a genuinely mutable
 * pointer to the same storage `desc.data` borrows.
 *
 * `desc.data` is `void const*` so the plain-C descriptor can equally
 * describe a read-only view; that alone is never sufficient evidence that
 * the underlying object is not `const` in the caller's own type system, so
 * this function does not derive a mutable pointer from `desc.data` via
 * `const_cast`. Instead the caller supplies `data` directly -- the same
 * pointer it already legitimately owns as non-const -- and `desc` is used
 * only for shape validation (rank/extents/strides/scalar code) and to
 * reject a descriptor additionally carrying `OPERON_VIEW_READONLY`.
 */
[[nodiscard]] auto MakeMatrixView(OperonViewDescriptor const& desc, gsl::not_null<Scalar*> data) -> tl::expected<ScalarMatrixView, OperonViewStatus>;

namespace detail {
    [[nodiscard]] constexpr auto ScalarCodeFor() -> std::uint32_t
    {
        // Operon::Scalar is float or double (types.hpp); anything else is a
        // build configuration this header does not need to defend against.
        return sizeof(Scalar) == sizeof(float) ? OPERON_SCALAR_F32 : OPERON_SCALAR_F64;
    }

    [[nodiscard]] constexpr auto HasNegativeStride(OperonViewDescriptor const& desc) -> bool
    {
        return desc.byte_strides[0] < 0 || desc.byte_strides[1] < 0;
    }

    template <typename T>
    [[nodiscard]] auto MakeStridedView(OperonViewDescriptor const& desc, T* data) -> std::mdspan<T, std::dextents<MemoryIndex, 2>, std::layout_stride>
    {
        using Extents = std::dextents<MemoryIndex, 2>;
        using Mapping = std::layout_stride::mapping<Extents>;
        auto const elementStride = static_cast<MemoryIndex>(desc.element_size);
        std::array<MemoryIndex, 2> strides {
            static_cast<MemoryIndex>(desc.byte_strides[0]) / elementStride,
            static_cast<MemoryIndex>(desc.byte_strides[1]) / elementStride,
        };
        return { data, Mapping { Extents { desc.extents[0], desc.extents[1] }, strides } };
    }
} // namespace detail

inline auto MakeConstMatrixView(OperonViewDescriptor const& desc) -> tl::expected<ConstScalarMatrixView, OperonViewStatus>
{
    if (desc.rank != 2U) {
        return tl::unexpected(OPERON_VIEW_ERR_RANK);
    }
    if (desc.scalar_code != detail::ScalarCodeFor()) {
        return tl::unexpected(OPERON_VIEW_ERR_SCALAR);
    }
    if (detail::HasNegativeStride(desc)) {
        return tl::unexpected(OPERON_VIEW_ERR_STRIDE);
    }
    if (auto const status = operon_view_validate(&desc, /*require_writable=*/0); status != OPERON_VIEW_OK) {
        return tl::unexpected(status);
    }
    return detail::MakeStridedView<Scalar const>(desc, static_cast<Scalar const*>(desc.data));
}

inline auto MakeMatrixView(OperonViewDescriptor const& desc, gsl::not_null<Scalar*> data) -> tl::expected<ScalarMatrixView, OperonViewStatus>
{
    if (desc.rank != 2U) {
        return tl::unexpected(OPERON_VIEW_ERR_RANK);
    }
    if (desc.scalar_code != detail::ScalarCodeFor()) {
        return tl::unexpected(OPERON_VIEW_ERR_SCALAR);
    }
    if (detail::HasNegativeStride(desc)) {
        return tl::unexpected(OPERON_VIEW_ERR_STRIDE);
    }
    if (auto const status = operon_view_validate(&desc, /*require_writable=*/1); status != OPERON_VIEW_OK) {
        return tl::unexpected(status);
    }
    return detail::MakeStridedView<Scalar>(desc, data.get());
}

} // namespace Operon

#endif
