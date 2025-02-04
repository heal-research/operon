#ifndef OPERON_MDSPAN_HPP
#define OPERON_MDSPAN_HPP

#define MDSPAN_IMPL_STANDARD_NAMESPACE std
#define MDSPAN_IMPL_PROPOSED_NAMESPACE experimental
#include <mdspan/mdspan.hpp>
#include <mdspan/mdarray.hpp>

template<typename Extents>
constexpr auto extents_size = []<auto... Idx>(std::index_sequence<Idx...>) {
    return (Extents::static_extent(Idx) * ...);
}(std::make_index_sequence<Extents::rank()>{});

#endif