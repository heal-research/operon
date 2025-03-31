#ifndef OPERON_ALIGNED_ALLOCATOR_HPP
#define OPERON_ALIGNED_ALLOCATOR_HPP

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <memory>

template<typename T, std::size_t Alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__>
requires (Alignment >= alignof(T))
class AlignedAllocator {

public:
    // NOLINTBEGIN(readability-identifier-naming)
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    template<class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    constexpr AlignedAllocator() noexcept = default;
    constexpr AlignedAllocator(AlignedAllocator const& other) noexcept = default;

    template<typename U>
    explicit constexpr AlignedAllocator(AlignedAllocator<U, Alignment> const&) noexcept { }

    auto operator=(AlignedAllocator const&) noexcept -> AlignedAllocator& = default;
    auto operator=(AlignedAllocator&&) noexcept -> AlignedAllocator& = default;

    constexpr ~AlignedAllocator() = default;

    template<typename U>
    auto operator==(AlignedAllocator<U> const&) -> bool { return true; }

    template<typename U>
    auto operator!=(AlignedAllocator<U> const&) -> bool { return false; }

    // NOLINTBEGIN(readability-identifier-naming)
    [[nodiscard]] auto allocate(std::size_t n) -> T* {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(::operator new[](n * sizeof(T), std::align_val_t{Alignment}));
    }

#if defined(__cpp_lib_allocate_at_least)
    [[nodiscard]] auto allocate_at_least(std::size_t num_elements) -> std::allocation_result<T*, std::size_t> {
        auto* p = allocate(num_elements);
        return std::allocation_result<T*>(p, num_elements);
    }
#endif

    auto deallocate(T* p, [[maybe_unused]] std::size_t n) -> void {
        ::operator delete[](p, std::align_val_t{Alignment});
    }
    // NOLINTEND(readability-identifier-naming)
};


#endif
