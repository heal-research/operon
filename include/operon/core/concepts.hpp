#ifndef OPERON_CONCEPTS_HPP
#define OPERON_CONCEPTS_HPP

#include <concepts>
#include <type_traits>

namespace Operon::Concepts {
    // T is an arithmetic number
    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;
} // namespace Operon::Concepts

#endif


