// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_VIEW_DESCRIPTOR_HPP
#define OPERON_VIEW_DESCRIPTOR_HPP

#include <array>
#include <cstdint>

#include <tl/expected.hpp>

#include "operon/core/memory_view.hpp"
#include "operon/core/view_descriptor.h"

namespace Operon {

/**
 * Constructs `Operon::ConstScalarMatrixView`/`Operon::ScalarMatrixView` from a
 * borrowed `OperonViewDescriptor`. This is the C++-only counterpart to
 * `operon_view_validate`: the descriptor itself carries no C++ types, but
 * building an `mdspan` over it requires `Operon::Scalar`, `std::mdspan`, and
 * `std::layout_stride`, none of which the plain-C header may depend on.
 *
 * `desc.scalar_code`/`desc.element_size` must match `Operon::Scalar`
 * (`OPERON_SCALAR_F32` when built with `USE_SINGLE_PRECISION`, otherwise
 * `OPERON_SCALAR_F64`) -- a descriptor for the other precision is a scalar
 * mismatch, not merely a differently-typed view, and is rejected the same
 * way as any other invalid descriptor.
 */
[[nodiscard]] auto MakeConstMatrixView(OperonViewDescriptor const& desc) -> tl::expected<ConstScalarMatrixView, OperonViewStatus>;

/// Same as `MakeConstMatrixView`, additionally rejecting a descriptor
/// carrying `OPERON_VIEW_READONLY`.
[[nodiscard]] auto MakeMatrixView(OperonViewDescriptor& desc) -> tl::expected<ScalarMatrixView, OperonViewStatus>;

namespace detail {
    [[nodiscard]] constexpr auto ScalarCodeFor() -> std::uint32_t
    {
        // Operon::Scalar is float or double (types.hpp); anything else is a
        // build configuration this header does not need to defend against.
        return sizeof(Scalar) == sizeof(float) ? OPERON_SCALAR_F32 : OPERON_SCALAR_F64;
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
    if (desc.scalar_code != detail::ScalarCodeFor()) {
        return tl::unexpected(OPERON_VIEW_ERR_SCALAR);
    }
    if (auto const status = operon_view_validate(&desc, /*require_writable=*/0); status != OPERON_VIEW_OK) {
        return tl::unexpected(status);
    }
    return detail::MakeStridedView<Scalar const>(desc, static_cast<Scalar const*>(desc.data));
}

inline auto MakeMatrixView(OperonViewDescriptor& desc) -> tl::expected<ScalarMatrixView, OperonViewStatus>
{
    if (desc.scalar_code != detail::ScalarCodeFor()) {
        return tl::unexpected(OPERON_VIEW_ERR_SCALAR);
    }
    if (auto const status = operon_view_validate(&desc, /*require_writable=*/1); status != OPERON_VIEW_OK) {
        return tl::unexpected(status);
    }
    // Validated above as non-readonly and non-null; borrowing back the
    // mutable pointer the caller owns is safe. `desc.data` is `void const*`
    // purely so the plain-C struct can describe read-only views too.
    return detail::MakeStridedView<Scalar>(desc, const_cast<Scalar*>(static_cast<Scalar const*>(desc.data))); // NOLINT(cppcoreguidelines-pro-type-const-cast)
}

} // namespace Operon

#endif
