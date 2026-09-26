// SPDX-License-Identifier: MIT

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "operon/core/memory_view.hpp"

TEST_CASE("canonical matrix view preserves logical indexing across layouts", "[memory-view]")
{
    std::array<Operon::Scalar, 6> row_major {1, 2, 3, 4, 5, 6};
    std::array<Operon::Scalar, 6> column_major {1, 4, 2, 5, 3, 6};
    using Extents = std::dextents<std::size_t, 2>;
    using Mapping = std::layout_stride::mapping<Extents>;

    auto at = [](Operon::ScalarMatrixView view, std::size_t row, std::size_t column) -> Operon::Scalar {
        return view.accessor().access(view.data_handle(), view.mapping()(row, column));
    };
    Operon::ScalarMatrixView rows {
        row_major.data(), Mapping {Extents {2, 3}, std::array<std::size_t, 2> {3, 1}}};
    Operon::ScalarMatrixView columns {
        column_major.data(), Mapping {Extents {2, 3}, std::array<std::size_t, 2> {1, 2}}};

    REQUIRE(at(rows, 0, 0) == 1);
    REQUIRE(at(rows, 1, 2) == 6);
    REQUIRE(at(columns, 0, 0) == 1);
    REQUIRE(at(columns, 1, 2) == 6);
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            REQUIRE(at(rows, row, column) == at(columns, row, column));
        }
    }
}

TEST_CASE("canonical matrix view supports padded strides", "[memory-view]")
{
    std::array<Operon::Scalar, 8> storage {1, 2, 3, 0, 4, 5, 6, 0};
    using Extents = std::dextents<std::size_t, 2>;
    using Mapping = std::layout_stride::mapping<Extents>;
    Operon::ScalarMatrixView view {
        storage.data(), Mapping {Extents {2, 3}, std::array<std::size_t, 2> {4, 1}}};
    auto at = [](Operon::ScalarMatrixView value, std::size_t row, std::size_t column) -> Operon::Scalar {
        return value.accessor().access(value.data_handle(), value.mapping()(row, column));
    };

    REQUIRE(view.extent(0) == 2);
    REQUIRE(view.extent(1) == 3);
    REQUIRE(at(view, 0, 2) == 3);
    REQUIRE(at(view, 1, 0) == 4);
    REQUIRE(at(view, 1, 2) == 6);
}
