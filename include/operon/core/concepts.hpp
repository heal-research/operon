// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CONCEPTS_HPP
#define OPERON_CONCEPTS_HPP

#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>

#include "types.hpp"

namespace Operon {
class Tree;
struct Individual;
} // namespace Operon

namespace Operon::Concepts {

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Callable with two (or three) value spans, returns a scalar error measure.
template<typename T>
concept ErrorMetricCallable = requires(T t,
    Operon::Span<Operon::Scalar const> x,
    Operon::Span<Operon::Scalar const> y,
    Operon::Span<Operon::Scalar const> w) {
    { t(x, y) } -> std::convertible_to<double>;
    { t(x, y, w) } -> std::convertible_to<double>;
};

// String-to-hash callable with transparent lookup support.
template<typename T>
concept Hasher = requires(T t, std::string_view sv) {
    typename T::is_transparent;
    { t(sv) } -> std::convertible_to<Operon::Hash>;
};

// Builds a new Tree given RNG + (targetLength, minDepth, maxDepth).
template<typename T>
concept Creator = requires(T const& t, Operon::RandomGenerator& rng,
    std::size_t a, std::size_t b, std::size_t c) {
    { t(rng, a, b, c) } -> std::same_as<Operon::Tree>;
};

// Mutates a Tree and returns the result.
template<typename T>
concept Mutator = requires(T const& t, Operon::RandomGenerator& rng, Operon::Tree tree) {
    { t(rng, tree) } -> std::same_as<Operon::Tree>;
};

// Recombines two parent Trees into a child Tree.
template<typename T>
concept Crossover = requires(T const& t, Operon::RandomGenerator& rng,
    Operon::Tree const& a, Operon::Tree const& b) {
    { t(rng, a, b) } -> std::same_as<Operon::Tree>;
};

// Selects an individual index from a pre-set population.
template<typename T>
concept Selector = requires(T const& t, Operon::RandomGenerator& rng) {
    { t(rng) } -> std::convertible_to<std::size_t>;
};

// Merges offspring into the parent population in-place.
template<typename T>
concept Reinserter = requires(T const& t, Operon::RandomGenerator& rng,
    Operon::Span<Operon::Individual> parents,
    Operon::Span<Operon::Individual> offspring) {
    { t(rng, parents, offspring) } -> std::same_as<void>;
};

// Evaluates an Individual and returns a fitness vector.
// Both the buffered and unbuffered call forms are required.
template<typename T>
concept EvaluatorCallable = requires(T const& t, Operon::RandomGenerator& rng,
    Operon::Individual const& ind,
    Operon::Span<Operon::Scalar> buf) {
    { t(rng, ind)      } -> std::same_as<Operon::Vector<Operon::Scalar>>;
    { t(rng, ind, buf) } -> std::same_as<Operon::Vector<Operon::Scalar>>;
};

// Static struct providing a scalar negative log-likelihood over three value spans.
// The third span is overloaded: sigma (Gaussian) or weights (Poisson); may be empty.
template<typename T>
concept Likelihood = requires(Operon::Span<Operon::Scalar const> x,
    Operon::Span<Operon::Scalar const> y,
    Operon::Span<Operon::Scalar const> z) {
    { T::ComputeLikelihood(x, y, z) } -> std::same_as<Operon::Scalar>;
};

} // namespace Operon::Concepts

#endif
