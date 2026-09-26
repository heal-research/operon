#include <array>

#include "operon/core/memory_view.hpp"

int main()
{
    std::array<Operon::Scalar, 4> storage {1, 2, 3, 4};
    using Extents = std::dextents<std::size_t, 2>;
    using Mapping = std::layout_stride::mapping<Extents>;
    Operon::ScalarMatrixView view {
        storage.data(), Mapping {Extents {2, 2}, std::array<std::size_t, 2> {2, 1}}};
    auto const value = view.accessor().access(view.data_handle(), view.mapping()(1, 1));
    return value == Operon::Scalar {4} ? 0 : 1;
}
