/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors */

#ifndef OPERON_VIEW_DESCRIPTOR_H
#define OPERON_VIEW_DESCRIPTOR_H

/*
 * C-compatible borrowed matrix-view descriptor.
 *
 * This header is valid, self-contained C11 (and valid C++): it declares no
 * ownership, callbacks, exceptions, or C++ types. It exists so a foreign
 * caller (Python buffer protocol glue, another language's FFI, a C library)
 * can hand Operon a borrowed view of its own storage without linking against
 * any C++ type. `operon_view_validate` is the only operation defined here;
 * constructing an `Operon::ScalarMatrixView`/`Operon::ConstScalarMatrixView`
 * from a validated descriptor is a C++-only concern (view_descriptor.hpp).
 *
 * Ownership: `data` is borrowed. The descriptor never outlives, frees, or
 * mutates lifetime of the pointee; the caller keeps the referenced storage
 * alive for as long as any view built from this descriptor is in use.
 *
 * Layout is frozen for 64-bit targets (matching Operon's x86-64-v3 build
 * floor): widening it is an ABI break and requires bumping
 * OPERON_VIEW_DESCRIPTOR_VERSION.
 */

#include <stddef.h> // NOLINT(modernize-deprecated-headers)
#include <stdint.h> // NOLINT(modernize-deprecated-headers)

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(*)

#define OPERON_VIEW_DESCRIPTOR_VERSION 1U
#define OPERON_VIEW_MAX_RANK 2U

enum OperonScalarCode {
    OPERON_SCALAR_F32 = 1,
    OPERON_SCALAR_F64 = 2
};

enum OperonViewFlags {
    OPERON_VIEW_READONLY = 1U << 0
};

enum OperonViewStatus {
    OPERON_VIEW_OK = 0,
    OPERON_VIEW_ERR_VERSION = 1,     /* unrecognized descriptor version */
    OPERON_VIEW_ERR_STRUCT_SIZE = 2, /* struct_size does not match sizeof at this version */
    OPERON_VIEW_ERR_RANK = 3,        /* rank is zero or exceeds OPERON_VIEW_MAX_RANK */
    OPERON_VIEW_ERR_SCALAR = 4,      /* scalar_code/element_size mismatch or unknown */
    OPERON_VIEW_ERR_NULL_DATA = 5,   /* data is null with a nonzero logical extent */
    OPERON_VIEW_ERR_EXTENT = 6,      /* an extent is zero in a rank slot below `rank` -- must be expressed as a valid empty view, not a garbage trailing dimension */
    OPERON_VIEW_ERR_STRIDE = 7,      /* a byte stride is not a multiple of element_size */
    OPERON_VIEW_ERR_ALIGNMENT = 8,   /* data pointer is not aligned to element_size */
    OPERON_VIEW_ERR_OVERFLOW = 9,    /* extents/strides overflow size_t/ptrdiff_t bounds */
    OPERON_VIEW_ERR_WRITABLE = 10    /* a writable view was requested against a read-only descriptor */
};

/*
 * Borrowed, logically-indexed, arbitrary-stride matrix descriptor.
 *
 * `extents[k]`/`byte_strides[k]` beyond `rank - 1` are ignored by
 * `operon_view_validate` and must be zeroed by the producer for forward
 * compatibility with a future higher-rank version.
 *
 * `byte_strides` are signed to permit reversed/negative-stride views; they
 * are always expressed in bytes, not elements, so callers never need to
 * know `element_size` to compute an offset.
 */
struct OperonViewDescriptor {
    uint32_t version;
    uint32_t struct_size;
    uint32_t rank;
    uint32_t scalar_code;
    uint32_t element_size;
    uint32_t flags;
    uint32_t reserved; /* zero; reserved for future flag bits */
    void const* data;  /* borrowed; never owned or freed by the consumer */
    size_t extents[OPERON_VIEW_MAX_RANK];
    ptrdiff_t byte_strides[OPERON_VIEW_MAX_RANK];
};

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(void*) == 8, "OperonViewDescriptor layout is frozen for 64-bit targets only");
_Static_assert(sizeof(struct OperonViewDescriptor) == 72, "OperonViewDescriptor layout changed; bump OPERON_VIEW_DESCRIPTOR_VERSION");
#elif defined(__cplusplus)
static_assert(sizeof(void*) == 8, "OperonViewDescriptor layout is frozen for 64-bit targets only");
static_assert(sizeof(struct OperonViewDescriptor) == 72, "OperonViewDescriptor layout changed; bump OPERON_VIEW_DESCRIPTOR_VERSION");
#endif

/*
 * Validates `desc` against version, struct size, rank, scalar type,
 * null/zero rules, stride divisibility, alignment, and overflow. Does not
 * dereference `data` beyond a null check; bounds beyond the descriptor's own
 * fields are the caller's responsibility once a view is constructed.
 *
 * `require_writable` is nonzero when the caller intends to construct a
 * mutable view; validation then rejects a descriptor carrying
 * OPERON_VIEW_READONLY.
 *
 * Returns OPERON_VIEW_OK on success, otherwise the first violated rule.
 */
static inline enum OperonViewStatus operon_view_validate(struct OperonViewDescriptor const* desc, int require_writable)
{
    size_t k;
    size_t logical_extent;

    if (desc == NULL) {
        return OPERON_VIEW_ERR_NULL_DATA;
    }
    if (desc->version != OPERON_VIEW_DESCRIPTOR_VERSION) {
        return OPERON_VIEW_ERR_VERSION;
    }
    if (desc->struct_size != sizeof(struct OperonViewDescriptor)) {
        return OPERON_VIEW_ERR_STRUCT_SIZE;
    }
    if (desc->rank == 0U || desc->rank > OPERON_VIEW_MAX_RANK) {
        return OPERON_VIEW_ERR_RANK;
    }

    switch (desc->scalar_code) {
    case OPERON_SCALAR_F32:
        if (desc->element_size != sizeof(float)) {
            return OPERON_VIEW_ERR_SCALAR;
        }
        break;
    case OPERON_SCALAR_F64:
        if (desc->element_size != sizeof(double)) {
            return OPERON_VIEW_ERR_SCALAR;
        }
        break;
    default:
        return OPERON_VIEW_ERR_SCALAR;
    }

    if (require_writable && (desc->flags & (uint32_t)OPERON_VIEW_READONLY) != 0U) {
        return OPERON_VIEW_ERR_WRITABLE;
    }

    logical_extent = 1;
    for (k = 0; k < desc->rank; ++k) {
        size_t extent = desc->extents[k];
        ptrdiff_t stride = desc->byte_strides[k];
        ptrdiff_t abs_stride = stride < 0 ? -stride : stride;

        if (extent == 0U) {
            return OPERON_VIEW_ERR_EXTENT;
        }
        if ((size_t)abs_stride % desc->element_size != 0U) {
            return OPERON_VIEW_ERR_STRIDE;
        }
        /* overflow check: logical_extent * extent must not wrap size_t */
        if (extent != 0U && logical_extent > (SIZE_MAX / extent)) {
            return OPERON_VIEW_ERR_OVERFLOW;
        }
        logical_extent *= extent;
    }

    if (desc->data == NULL) {
        return OPERON_VIEW_ERR_NULL_DATA;
    }
    if ((size_t)(uintptr_t)desc->data % desc->element_size != 0U) {
        return OPERON_VIEW_ERR_ALIGNMENT;
    }

    return OPERON_VIEW_OK;
}

// NOLINTEND(*)

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPERON_VIEW_DESCRIPTOR_H */
