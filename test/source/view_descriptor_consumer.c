/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors */

/*
 * Standalone C11 consumer of view_descriptor.h. Proves the header is valid,
 * self-contained C11 with no C++ dependency -- a foreign caller (Python
 * buffer-protocol glue, another language's FFI) can include only this header
 * and validate a descriptor without linking against any Operon C++ symbol.
 * Deliberately not linked against operon::operon.
 */

#include <operon/core/view_descriptor.h>

// NOLINTBEGIN(*)

#include <stdio.h>

static int check(char const* name, enum OperonViewStatus got, enum OperonViewStatus want)
{
    if (got != want) {
        fprintf(stderr, "%s: expected status %d, got %d\n", name, (int)want, (int)got);
        return 1;
    }
    return 0;
}

int main(void)
{
    float data[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    struct OperonViewDescriptor desc;
    int failures = 0;

    desc.version = OPERON_VIEW_DESCRIPTOR_VERSION;
    desc.struct_size = sizeof(desc);
    desc.rank = 2;
    desc.scalar_code = OPERON_SCALAR_F32;
    desc.element_size = sizeof(float);
    desc.flags = 0;
    desc.reserved = 0;
    desc.data = data;
    desc.extents[0] = 2;
    desc.extents[1] = 2;
    desc.byte_strides[0] = 2 * (ptrdiff_t)sizeof(float);
    desc.byte_strides[1] = (ptrdiff_t)sizeof(float);

    failures += check("valid descriptor", operon_view_validate(&desc, 0), OPERON_VIEW_OK);

    desc.flags = OPERON_VIEW_READONLY;
    failures += check("readonly rejects writable request", operon_view_validate(&desc, 1), OPERON_VIEW_ERR_WRITABLE);
    failures += check("readonly still permits read-only request", operon_view_validate(&desc, 0), OPERON_VIEW_OK);
    desc.flags = 0;

    desc.extents[1] = 0;
    failures += check("zero extent rejected", operon_view_validate(&desc, 0), OPERON_VIEW_ERR_EXTENT);
    desc.extents[1] = 2;

    desc.data = NULL;
    failures += check("null data rejected", operon_view_validate(&desc, 0), OPERON_VIEW_ERR_NULL_DATA);
    desc.data = data;

    if (failures == 0) {
        printf("view_descriptor_consumer: OK\n");
        return 0;
    }

    return 1;
}

// NOLINTEND(*)
