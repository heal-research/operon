// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_TYPES_HPP
#define OPERON_TYPES_HPP

#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <cstdint>
#include <nonstd/span.hpp>

#include "constants.hpp"
#include "operon/random/random.hpp"

namespace Operon {
using Hash = uint64_t;
constexpr HashFunction HashFunc = HashFunction::XXHash;

using RandomGenerator = Random::RomuTrio;

template <typename T>
using Vector = std::vector<T>;

template <typename T>
using Span = nonstd::span<T>;

template <class Key,
          class T,
          class Hash = ankerl::unordered_dense::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
          class Bucket = ankerl::unordered_dense::bucket_type::standard>
using Map = ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket>;

template <class Key,
          class Hash = ankerl::unordered_dense::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<Key>,
          class Bucket = ankerl::unordered_dense::bucket_type::standard>
using Set = ankerl::unordered_dense::detail::table<Key, void, Hash, KeyEqual, AllocatorOrContainer, Bucket>;


#if defined(USE_SINGLE_PRECISION)
using Scalar = float;
#else
using Scalar = double;
#endif
} // namespace Operon

#endif
