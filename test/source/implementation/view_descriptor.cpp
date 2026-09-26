// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <array>
#include <cstddef>
#include <cstdint>
#include <catch2/catch_test_macros.hpp>

#include "operon/core/view_descriptor.hpp"

namespace {

auto ScalarCodeForBuild() -> std::uint32_t
{
    return sizeof(Operon::Scalar) == sizeof(float) ? OPERON_SCALAR_F32 : OPERON_SCALAR_F64;
}

auto MakeValidDescriptor(std::array<Operon::Scalar, 6>& storage) -> OperonViewDescriptor
{
    OperonViewDescriptor desc {};
    desc.version = OPERON_VIEW_DESCRIPTOR_VERSION;
    desc.struct_size = sizeof(desc);
    desc.rank = 2;
    desc.scalar_code = ScalarCodeForBuild();
    desc.element_size = sizeof(Operon::Scalar);
    desc.data = storage.data();
    desc.extents[0] = 2;
    desc.extents[1] = 3;
    desc.byte_strides[0] = 3 * sizeof(Operon::Scalar);
    desc.byte_strides[1] = sizeof(Operon::Scalar);
    return desc;
}

} // namespace

TEST_CASE("view_descriptor: valid row-major descriptor yields a matching mdspan", "[view-descriptor]")
{
    std::array<Operon::Scalar, 6> storage {1, 2, 3, 4, 5, 6};
    auto desc = MakeValidDescriptor(storage);

    auto view = Operon::MakeConstMatrixView(desc);
    REQUIRE(view.has_value());
    CHECK(view->extent(0) == 2);
    CHECK(view->extent(1) == 3);
    auto const at = [&](std::size_t r, std::size_t c) {
        return view->accessor().access(view->data_handle(), view->mapping()(r, c));
    };
    CHECK(at(0, 0) == Operon::Scalar {1});
    CHECK(at(1, 2) == Operon::Scalar {6});
}

TEST_CASE("view_descriptor: column-major strides describe the same logical values", "[view-descriptor]")
{
    std::array<Operon::Scalar, 6> storage {1, 4, 2, 5, 3, 6}; // column-major for the 2x3 [1..6] matrix
    OperonViewDescriptor desc {};
    desc.version = OPERON_VIEW_DESCRIPTOR_VERSION;
    desc.struct_size = sizeof(desc);
    desc.rank = 2;
    desc.scalar_code = ScalarCodeForBuild();
    desc.element_size = sizeof(Operon::Scalar);
    desc.data = storage.data();
    desc.extents[0] = 2;
    desc.extents[1] = 3;
    desc.byte_strides[0] = sizeof(Operon::Scalar);
    desc.byte_strides[1] = 2 * sizeof(Operon::Scalar);

    auto view = Operon::MakeConstMatrixView(desc);
    REQUIRE(view.has_value());
    auto const at = [&](std::size_t r, std::size_t c) {
        return view->accessor().access(view->data_handle(), view->mapping()(r, c));
    };
    CHECK(at(0, 0) == Operon::Scalar {1});
    CHECK(at(1, 2) == Operon::Scalar {6});
}

TEST_CASE("view_descriptor: MakeMatrixView writes back through the caller-supplied mutable pointer", "[view-descriptor]")
{
    std::array<Operon::Scalar, 4> storage {0, 0, 0, 0};
    OperonViewDescriptor desc {};
    desc.version = OPERON_VIEW_DESCRIPTOR_VERSION;
    desc.struct_size = sizeof(desc);
    desc.rank = 2;
    desc.scalar_code = ScalarCodeForBuild();
    desc.element_size = sizeof(Operon::Scalar);
    desc.data = storage.data();
    desc.extents[0] = 2;
    desc.extents[1] = 2;
    desc.byte_strides[0] = 2 * sizeof(Operon::Scalar);
    desc.byte_strides[1] = sizeof(Operon::Scalar);

    auto view = Operon::MakeMatrixView(desc, storage.data());
    REQUIRE(view.has_value());
    view->accessor().access(view->data_handle(), view->mapping()(1, 1)) = Operon::Scalar {42};
    CHECK(storage[3] == Operon::Scalar {42});
}

TEST_CASE("view_descriptor: MakeMatrixView rejects a read-only descriptor", "[view-descriptor]")
{
    std::array<Operon::Scalar, 6> storage {1, 2, 3, 4, 5, 6};
    auto desc = MakeValidDescriptor(storage);
    desc.flags = OPERON_VIEW_READONLY;

    auto mutableView = Operon::MakeMatrixView(desc, storage.data());
    REQUIRE_FALSE(mutableView.has_value());
    CHECK(mutableView.error() == OPERON_VIEW_ERR_WRITABLE);

    auto constView = Operon::MakeConstMatrixView(desc);
    CHECK(constView.has_value());
}

TEST_CASE("view_descriptor: rejects the wrong scalar precision", "[view-descriptor]")
{
    std::array<Operon::Scalar, 6> storage {1, 2, 3, 4, 5, 6};
    auto desc = MakeValidDescriptor(storage);
    desc.scalar_code = ScalarCodeForBuild() == OPERON_SCALAR_F32 ? OPERON_SCALAR_F64 : OPERON_SCALAR_F32;

    auto view = Operon::MakeConstMatrixView(desc);
    REQUIRE_FALSE(view.has_value());
    CHECK(view.error() == OPERON_VIEW_ERR_SCALAR);
}

TEST_CASE("view_descriptor: MakeConstMatrixView/MakeMatrixView reject a rank-1 descriptor", "[view-descriptor]")
{
    // A rank-1 descriptor is structurally valid per operon_view_validate --
    // this API is specifically the rank-2 matrix constructor and must not
    // read extents[1]/byte_strides[1], which sit outside a rank-1
    // descriptor's own validated range.
    std::array<Operon::Scalar, 4> storage {1, 2, 3, 4};
    OperonViewDescriptor desc {};
    desc.version = OPERON_VIEW_DESCRIPTOR_VERSION;
    desc.struct_size = sizeof(desc);
    desc.rank = 1;
    desc.scalar_code = ScalarCodeForBuild();
    desc.element_size = sizeof(Operon::Scalar);
    desc.data = storage.data();
    desc.extents[0] = 4;
    desc.byte_strides[0] = sizeof(Operon::Scalar);
    // Deliberately garbage in the rank-1-ignored slot, to prove the matrix
    // constructors never read it.
    desc.extents[1] = 0xDEADBEEFU;
    desc.byte_strides[1] = 12345;

    CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_OK);

    auto constView = Operon::MakeConstMatrixView(desc);
    REQUIRE_FALSE(constView.has_value());
    CHECK(constView.error() == OPERON_VIEW_ERR_RANK);

    auto mutableView = Operon::MakeMatrixView(desc, storage.data());
    REQUIRE_FALSE(mutableView.has_value());
    CHECK(mutableView.error() == OPERON_VIEW_ERR_RANK);
}

TEST_CASE("view_descriptor: negative byte strides are structurally valid but rejected by the matrix constructors", "[view-descriptor]")
{
    std::array<Operon::Scalar, 6> storage {1, 2, 3, 4, 5, 6};
    auto desc = MakeValidDescriptor(storage);
    desc.byte_strides[0] = -static_cast<std::ptrdiff_t>(3 * sizeof(Operon::Scalar));

    // operon_view_validate only checks divisibility/overflow/representability
    // of a stride, not its sign -- a negative stride is a general, valid
    // reversed-axis descriptor as far as the C layer is concerned.
    CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_OK);

    // Neither matrix constructor yet supports building an mdspan (unsigned
    // index type) from a negative stride, so both reject it explicitly
    // rather than silently wrapping it into a huge positive stride.
    auto constView = Operon::MakeConstMatrixView(desc);
    REQUIRE_FALSE(constView.has_value());
    CHECK(constView.error() == OPERON_VIEW_ERR_STRIDE);

    auto mutableView = Operon::MakeMatrixView(desc, storage.data());
    REQUIRE_FALSE(mutableView.has_value());
    CHECK(mutableView.error() == OPERON_VIEW_ERR_STRIDE);
}

TEST_CASE("view_descriptor: operon_view_validate rejects malformed descriptors", "[view-descriptor]")
{
    std::array<Operon::Scalar, 6> storage {1, 2, 3, 4, 5, 6};

    SECTION("wrong version")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.version = OPERON_VIEW_DESCRIPTOR_VERSION + 1;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_VERSION);
    }
    SECTION("wrong struct size")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.struct_size = sizeof(desc) - 1;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_STRUCT_SIZE);
    }
    SECTION("rank zero")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.rank = 0;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_RANK);
    }
    SECTION("rank exceeds max")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.rank = OPERON_VIEW_MAX_RANK + 1;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_RANK);
    }
    SECTION("misaligned stride")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.byte_strides[1] = static_cast<std::ptrdiff_t>(desc.element_size) - 1;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_STRIDE);
    }
    SECTION("stride of PTRDIFF_MIN is rejected, not negated into UB")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.byte_strides[1] = PTRDIFF_MIN;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_OVERFLOW);
    }
    SECTION("null data")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.data = nullptr;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_NULL_DATA);
    }
    SECTION("extent overflow (element count)")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.extents[0] = SIZE_MAX;
        desc.extents[1] = 2;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_OVERFLOW);
    }
    SECTION("stride-span overflow (large extent times large stride)")
    {
        // Individually representable extent and stride whose product (the
        // maximum byte offset reached along this axis) is not.
        auto desc = MakeValidDescriptor(storage);
        desc.rank = 1;
        desc.extents[0] = SIZE_MAX / 4;
        desc.byte_strides[0] = static_cast<std::ptrdiff_t>(desc.element_size) * 4;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_OVERFLOW);
    }
    SECTION("readonly rejected when writable required")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.flags = OPERON_VIEW_READONLY;
        CHECK(operon_view_validate(&desc, 1) == OPERON_VIEW_ERR_WRITABLE);
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_OK);
    }
}

TEST_CASE("view_descriptor: a zero extent describes a valid, empty view", "[view-descriptor]")
{
    std::array<Operon::Scalar, 6> storage {1, 2, 3, 4, 5, 6};

    SECTION("zero extent with real backing storage")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.extents[1] = 0;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_OK);

        auto view = Operon::MakeConstMatrixView(desc);
        REQUIRE(view.has_value());
        CHECK(view->extent(0) == 2);
        CHECK(view->extent(1) == 0);
    }

    SECTION("zero extent with null data is still valid: nothing is ever dereferenced")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.extents[0] = 0;
        desc.data = nullptr;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_OK);

        auto view = Operon::MakeConstMatrixView(desc);
        REQUIRE(view.has_value());
        CHECK(view->extent(0) == 0);
    }

    SECTION("zero extent still enforces stride divisibility on the empty axis")
    {
        auto desc = MakeValidDescriptor(storage);
        desc.extents[0] = 0;
        desc.byte_strides[0] = static_cast<std::ptrdiff_t>(desc.element_size) - 1;
        CHECK(operon_view_validate(&desc, 0) == OPERON_VIEW_ERR_STRIDE);
    }
}
