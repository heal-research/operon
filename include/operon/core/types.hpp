// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_TYPES_HPP
#define OPERON_TYPES_HPP

#include <ankerl/unordered_dense.h>
#include <cstdint>
#include <fluky/fluky.hpp>
#include <gch/small_vector.hpp>
#include <span>
#include <utility>

#include "constants.hpp"
#include "operon/mdspan/mdspan.hpp"
#include "aligned_allocator.hpp"

namespace Operon {
using Hash = uint64_t;
constexpr HashFunction HashFunc = HashFunction::XXHash;

using RandomGenerator = fluky::xoshiro256ss;

template <typename T, typename Allocator = std::allocator<T>>
using Vector = std::vector<T, Allocator>;

template <typename T>
using Span = std::span<T>;

template<typename T, typename Extents, typename LayoutPolicy = std::layout_left, typename AccessorPolicy = std::default_accessor<T>>
using MDSpan = std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>;

template<typename T, typename Extents, typename LayoutPolicy = std::layout_left, typename Container = std::vector<T>>
using MDArray = std::experimental::mdarray<T, Extents, LayoutPolicy, Container>;

template <class Key,
          class T,
          class Hash = ankerl::unordered_dense::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
          class Bucket = ankerl::unordered_dense::bucket_type::standard,
          class BucketContainer = ankerl::unordered_dense::detail::default_container_t>
using Map = ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, BucketContainer, false>;

template <class Key,
          class T,
          class Hash = ankerl::unordered_dense::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
          class Bucket = ankerl::unordered_dense::bucket_type::standard,
          class BucketContainer = ankerl::unordered_dense::detail::default_container_t>
using SegmentedMap = ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, BucketContainer, true>;

template <class Key,
          class Hash = ankerl::unordered_dense::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<Key>,
          class Bucket = ankerl::unordered_dense::bucket_type::standard,
          class BucketContainer = ankerl::unordered_dense::detail::default_container_t>
using Set = ankerl::unordered_dense::detail::table<Key, void, Hash, KeyEqual, AllocatorOrContainer, Bucket, BucketContainer, false>;

template <class Key,
          class Hash = ankerl::unordered_dense::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<Key>,
          class Bucket = ankerl::unordered_dense::bucket_type::standard,
          class BucketContainer = ankerl::unordered_dense::detail::default_container_t>
using SegmentedSet = ankerl::unordered_dense::detail::table<Key, void, Hash, KeyEqual, AllocatorOrContainer, Bucket, BucketContainer, true>;

template<size_t... Ints>
using Seq = std::integer_sequence<size_t, Ints...>;

#if defined(USE_SINGLE_PRECISION)
using Scalar = float;
#else
using Scalar = double;
#endif

} // namespace Operon

#endif
