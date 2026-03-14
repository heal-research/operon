#ifndef OPERON_BITSET_HPP
#define OPERON_BITSET_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace Operon {

template <std::size_t N>
class Bitset {
    static constexpr std::size_t bits_per_word = 64;
    static constexpr std::size_t num_words = (N + bits_per_word - 1) / bits_per_word;

    uint64_t words_[num_words]{};

    [[nodiscard]] constexpr auto WordOf(std::size_t i) const -> std::size_t { return i / bits_per_word; }
    [[nodiscard]] constexpr auto BitOf(std::size_t i)  const -> std::size_t { return i % bits_per_word; }

public:
    constexpr Bitset() = default;

    // Set / clear / test
    constexpr void Set(std::size_t i)   { words_[WordOf(i)] |=  (uint64_t{1} << BitOf(i)); }
    constexpr void Clear(std::size_t i) { words_[WordOf(i)] &= ~(uint64_t{1} << BitOf(i)); }
    [[nodiscard]] constexpr auto Test(std::size_t i) const -> bool {
        return (words_[WordOf(i)] >> BitOf(i)) & 1;
    }

    // Bitwise operations
    constexpr auto operator|(Bitset other) const -> Bitset {
        Bitset r;
        for (std::size_t i = 0; i < num_words; ++i)
            r.words_[i] = words_[i] | other.words_[i];
        return r;
    }
    constexpr auto operator|=(Bitset other) -> Bitset& {
        for (std::size_t i = 0; i < num_words; ++i) {
            words_[i] |= other.words_[i];
        }
        return *this;
    }
    constexpr auto operator&(Bitset other) const -> Bitset {
        Bitset r;
        for (std::size_t i = 0; i < num_words; ++i)
            r.words_[i] = words_[i] & other.words_[i];
        return r;
    }
    constexpr auto operator&=(Bitset other) -> Bitset& {
        for (std::size_t i = 0; i < num_words; ++i) {
            words_[i] &= other.words_[i];
        }
        return *this;
    }
    constexpr auto operator~() const -> Bitset {
        Bitset r;
        for (std::size_t i = 0; i < num_words; ++i) {
            r.words_[i] = ~words_[i];
        }
        // Mask off unused bits in the last word so Count()/ForEach() stay correct.
        if constexpr (N % bits_per_word != 0) {
            r.words_[num_words - 1] &= (uint64_t{1} << (N % bits_per_word)) - 1;
        }
        return r;
    }

    [[nodiscard]] constexpr auto Any() const -> bool {
        for (std::size_t i = 0; i < num_words; ++i) {
            if (words_[i]) { return true; }
        }
        return false;
    }
    [[nodiscard]] constexpr auto None() const -> bool { return !Any(); }

    [[nodiscard]] constexpr auto Count() const -> std::size_t {
        std::size_t c = 0;
        for (std::size_t i = 0; i < num_words; ++i) {
            c += static_cast<std::size_t>(std::popcount(words_[i]));
        }
        return c;
    }

    // Iterate over set bits efficiently
    template <typename F>
    constexpr void ForEach(F&& f) const {
        for (std::size_t w = 0; w < num_words; ++w) {
            auto bits = words_[w];
            while (bits) {
                auto idx = static_cast<std::size_t>(std::countr_zero(bits));
                std::forward<F>(f)((w * bits_per_word) + idx);
                bits &= bits - 1;  // clear lowest set bit
            }
        }
    }

    constexpr auto operator==(Bitset const&) const -> bool = default;
};
}  // namespace Operon

#endif