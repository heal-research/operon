#ifndef OPERON_BACKEND_HPP
#define OPERON_BACKEND_HPP

#include "operon/mdspan/mdspan.hpp"
#include "operon/core/types.hpp"

namespace Operon::Backend {
    template<typename T>
    static auto constexpr BatchSize = 512UL / sizeof(T);

    static auto constexpr DefaultAlignment = 32UL;

    template<typename T, std::size_t S = BatchSize<T>>
    using View = std::mdspan<T, std::extents<int, S, std::dynamic_extent>, std::layout_left>;

    template<typename T, std::size_t S>
    auto Ptr(View<T, S> view, std::integral auto col) -> Backend::View<T, S>::element_type* {
        return view.data_handle() + col * S;
    }
} // namespace Operon::Backend

#endif
