#ifndef OPERON_MDSPAN_HPP
#define OPERON_MDSPAN_HPP

#define MDSPAN_IMPL_STANDARD_NAMESPACE std
#define MDSPAN_IMPL_PROPOSED_NAMESPACE experimental
#include <mdspan/mdspan.hpp>
#include <mdspan/mdarray.hpp>

#include <memory>

template<typename Extents>
constexpr auto extents_size = []<auto... Idx>(std::index_sequence<Idx...>) {
    return (Extents::static_extent(Idx) * ...);
}(std::make_index_sequence<Extents::rank()>{});

#if defined(MDSPAN_IMPL_COMPILER_MSVC) || defined(__INTEL_COMPILER)
#  define _MDSPAN_RESTRICT_KEYWORD __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define _MDSPAN_RESTRICT_KEYWORD __restrict__
#else
#  define _MDSPAN_RESTRICT_KEYWORD
#endif

#define _MDSPAN_RESTRICT_POINTER( ELEMENT_TYPE ) ELEMENT_TYPE * _MDSPAN_RESTRICT_KEYWORD

// https://en.cppreference.com/w/c/language/restrict gives examples
// of the kinds of optimizations that may apply to restrict.  For instance,
// "[r]estricted pointers can be assigned to unrestricted pointers freely,
// the optimization opportunities remain in place
// as long as the compiler is able to analyze the code:"
//
// void f(int n, float * restrict r, float * restrict s) {
//   float * p = r, * q = s; // OK
//   while(n-- > 0) *p++ = *q++; // almost certainly optimized just like *r++ = *s++
// }
//
// This is relevant because restrict_accessor<ElementType>::reference is _not_ restrict.
// (It's not formally correct to apply C restrict wording to C++ references.
// However, GCC defines this extension:
//
// https://gcc.gnu.org/onlinedocs/gcc/Restricted-Pointers.html
//
// In what follows, I'll assume that this has a reasonable definition.)
// The idea is that even though p[i] has type ElementType& and not ElementType& restrict,
// the compiler can figure out that the reference comes from a pointer based on p,
// which is marked restrict.
//
// Note that any performance improvements can only be determined by experiment.
// Compilers are not required to do anything with restrict.
// Any use of this keyword is not Standard C++,
// so you'll have to refer to the compiler's documentation,
// look at the assembler output, and do performance experiments.
//
// NOLINTBEGIN(*)
template<class ElementType>
struct restrict_accessor {
    using offset_policy = std::default_accessor<ElementType>;
    using element_type = ElementType;
    using reference = ElementType&;
    using data_handle_type = _MDSPAN_RESTRICT_POINTER( ElementType );

    constexpr restrict_accessor() noexcept = default;

    MDSPAN_TEMPLATE_REQUIRES(
            class OtherElementType,
            /* requires */ (std::is_convertible<OtherElementType(*)[], element_type(*)[]>::value)
            )
    constexpr explicit restrict_accessor(restrict_accessor<OtherElementType>/*noname*/) noexcept {}
    constexpr auto access(data_handle_type p, size_t i) const noexcept -> reference {
        return p[i];
    }
    constexpr auto
        offset(data_handle_type p, size_t i) const noexcept -> typename offset_policy::data_handle_type {
            return p + i;
        }
};

// NOTE (mfh 2022/08/08) BYTE_ALIGNMENT must be unsigned and a power of 2.
#if defined(__cpp_lib_assume_aligned)
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) (std::assume_aligned< BYTE_ALIGNMENT >( POINTER ))
  constexpr char assume_aligned_method[] = "std::assume_aligned";
#elif defined(__ICL)
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__ICC)
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__clang__)
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__GNUC__)
  // __builtin_assume_aligned returns void*
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) reinterpret_cast< ELEMENT_TYPE* >(__builtin_assume_aligned( POINTER, BYTE_ALIGNMENT ))
  constexpr char assume_aligned_method[] = "__builtin_assume_aligned";
#else
#  define _MDSPAN_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#endif

// Some compilers other than Clang or GCC like to define __clang__ or __GNUC__.
// Thus, we order the tests from most to least specific.
#if defined(__ICL)
#  define _MDSPAN_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __declspec(align_value( BYTE_ALIGNMENT ))
  constexpr char align_attribute_method[] = "__declspec(align_value(BYTE_ALIGNMENT))";
#elif defined(__ICC)
#  define _MDSPAN_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __attribute__((align_value( BYTE_ALIGNMENT )))
  constexpr char align_attribute_method[] = "__attribute__((align_value(BYTE_ALIGNMENT)))";
#elif defined(__clang__)
#  define _MDSPAN_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __attribute__((align_value( BYTE_ALIGNMENT )))
  constexpr char align_attribute_method[] = "__attribute__((align_value(BYTE_ALIGNMENT)))";
#else
#  define _MDSPAN_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT )
  constexpr char align_attribute_method[] = "(none)";
#endif

constexpr auto
is_nonzero_power_of_two(const std::size_t x) -> bool
{
// Just checking __cpp_lib_int_pow2 isn't enough for some GCC versions.
// The <bit> header exists, but std::has_single_bit does not.
#if defined(__cpp_lib_int_pow2) && __cplusplus >= 202002L
  return std::has_single_bit(x);
#else
  return x != 0 && (x & (x - 1)) == 0;
#endif
}

template<class ElementType>
constexpr auto
valid_byte_alignment(const std::size_t byte_alignment) -> bool
{
  return is_nonzero_power_of_two(byte_alignment) && byte_alignment >= alignof(ElementType);
}

// We define aligned_pointer_t through a struct
// so we can check whether the byte alignment is valid.
// This makes it impossible to use the alias
// with an invalid byte alignment.
template<class ElementType, std::size_t byte_alignment>
struct aligned_pointer {
  static_assert(valid_byte_alignment<ElementType>(byte_alignment),
		"byte_alignment must be a power of two no less than "
		"the minimum required alignment of ElementType.");

#if defined(__ICC)
  // x86-64 ICC 2021.5.0 emits warning #3186 ("expected typedef declaration") here.
  // No other compiler (including Clang, which has a similar type attribute) has this issue.
#  pragma warning push
#  pragma warning disable 3186
#endif

  using type = ElementType* _MDSPAN_ALIGN_VALUE_ATTRIBUTE( byte_alignment );

#if defined(__ICC)
#  pragma warning pop
#endif
};

template<class ElementType, std::size_t byte_alignment>
using aligned_pointer_t = typename aligned_pointer<ElementType, byte_alignment>::type;

template<class ElementType, std::size_t byte_alignment>
auto
bless(ElementType* ptr, std::integral_constant<std::size_t, byte_alignment> /* ba */ ) -> aligned_pointer_t<ElementType, byte_alignment>
{
  return _MDSPAN_ASSUME_ALIGNED( ElementType, ptr, byte_alignment );
}

template<class ElementType, std::size_t ByteAlignment>
struct aligned_accessor {
  using offset_policy = std::default_accessor<ElementType>;
  using element_type = ElementType;
  using reference = ElementType&;
  using data_handle_type = aligned_pointer_t<ElementType, ByteAlignment>;

  constexpr aligned_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    std::size_t other_byte_alignment,
    /* requires */ (std::is_convertible<OtherElementType(*)[], element_type(*)[]>::value && other_byte_alignment == ByteAlignment)
    )
  constexpr explicit aligned_accessor(aligned_accessor<OtherElementType, other_byte_alignment>) noexcept {}

  constexpr auto access(data_handle_type p, size_t i) const noexcept -> reference {
    // This may declare alignment twice, depending on
    // if we have an attribute for marking pointer types.
    return _MDSPAN_ASSUME_ALIGNED( ElementType, p, ByteAlignment )[i];
  }

  constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept {
    return p + i;
  }
};
// NOLINTEND(*)

#endif
