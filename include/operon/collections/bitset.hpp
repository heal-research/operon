// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_COLLECTIONS_BITSET_HPP
#define OPERON_COLLECTIONS_BITSET_HPP

#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <ostream>
#include <vector>

#if defined(_M_IX86) || defined(_M_ARM) || defined(_M_X64) || defined(_M_ARM64)
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
#endif
#if (defined(_M_X64) || defined(_M_ARM64))
#pragma intrinsic(_BitScanForward64)
#endif

namespace Operon {
    template<typename T = uint64_t /* block type */, size_t S = std::numeric_limits<T>::digits /* block size in bits */>
    class Bitset {
        std::vector<T> blocks_;
        size_t numBits_{};

        [[nodiscard]] inline auto BlockIndex(size_t i) const { return i / S; }
        [[nodiscard]] inline auto BitIndex(size_t i) const { return BlockIndex(i) + i % S; }

        public:
        static constexpr T ZeroBlock = T{0}; // a block with all the bits set to zero
        static constexpr T OneBlock = ~ZeroBlock; // a block with all the bits set to one
        static constexpr size_t BlockSize = S;
        using Block = T;

        Bitset() = default;

        explicit Bitset(size_t n, T blockInit)
        {
            Resize(n, blockInit);
        }

        inline void Fill(T value) { std::fill(blocks_.begin(), blocks_.end(), value); }

        inline void Resize(size_t n, T blockInit = T{0}) {
            numBits_ = n;
            size_t const nb = (n / S) + (n % S != 0); // NOLINT
            blocks_.resize(nb, blockInit);
            blocks_.back() >>= S * nb - n; // zero the bits in the last block that are over n
        }

        inline void Set(size_t i) {
            assert(i < numBits_);
            blocks_[i / S] |= (T{1} << (i % S));
        }

        inline void Reset(size_t i) {
            assert(i < numBits_);
            blocks_[i / S] &= ~(T{1} << (i % S));
        }

        [[nodiscard]] auto PopCount() const -> size_t {
            return std::transform_reduce(blocks_.begin(), blocks_.end(), size_t{0}, std::plus<>{}, [](auto b) { return __builtin_popcountl(b); });
        }

        [[nodiscard]] auto Size() const -> size_t { return numBits_; }
        [[nodiscard]] auto Capacity() const -> size_t { return BlockSize * blocks_.size(); }

        auto Data() -> T* { return blocks_.data(); } 
        [[nodiscard]] auto Data() const -> T const* { return blocks_.data(); }

        auto Blocks() const -> std::vector<T> const& { return blocks_; }
        auto Blocks() -> std::vector<T>& { return blocks_; }

        [[nodiscard]] inline auto NumBlocks() const -> size_t { return blocks_.size(); }
        [[nodiscard]] inline auto Empty() const -> bool { return blocks_.empty(); }

        inline auto operator[](size_t i) const -> bool {
            return static_cast<bool>(blocks_[i / S] & (T{1} << (i % S))); 
        }

        template<typename BinaryOp>
        auto Apply(Bitset const& other) -> Bitset {
            assert(blocks_.size() == other.blocks_.size());
            assert(numBits_ == other.numBits_);
            Bitset<T> result(numBits_);
            BinaryOp op{};
            auto const* p = blocks_.data();
            auto const* q = other.blocks_.data();
            auto* r = result.blocks_.data();
            for (size_t i = 0UL; i < blocks_.size(); ++i) {
                r[i] = op(p[i], q[i]);
            }
            return result;
        }

        template<typename BinaryOp>
        auto Apply(Bitset const& other) -> Bitset& {
            assert(blocks_.size() == other.blocks_.size());
            assert(numBits_ == other.numBits_);
            Bitset<T> result(numBits_);
            BinaryOp op{};
            auto * p = blocks_.data();
            auto const* q = other.blocks_.data();
            for (size_t i = 0UL; i < blocks_.size(); ++i) {
                p[i] = op(p[i], q[i]);
            }
            return *this;
        }

        friend auto operator&(Bitset const& lhs, Bitset const& rhs) -> Bitset {
            auto result = lhs;
            result &= rhs;
            return result;
        }

        friend auto operator&=(Bitset& lhs, Bitset const& rhs) -> Bitset& {
            assert(lhs.blocks_.size() == rhs.blocks_.size());
            assert(lhs.numBits_ == rhs.numBits_);
            for (size_t i = 0; i < lhs.blocks_.size(); ++i) {
                lhs[i] |= rhs[i];
            }
            return lhs;
        };

        friend auto operator|(Bitset const& lhs, Bitset const& rhs) -> Bitset {
            auto result = lhs;
            result |= rhs;
            return result;
        }

        friend auto operator|=(Bitset& lhs, Bitset const& rhs) -> Bitset& {
            assert(lhs.blocks_.size() == rhs.blocks_.size());
            assert(lhs.numBits_ == rhs.numBits_);
            for (size_t i = 0; i < lhs.blocks_.size(); ++i) {
                lhs[i] |= rhs[i];
            }
            return lhs;
        };

        friend auto operator^(Bitset const& lhs, Bitset const& rhs) -> Bitset {
            auto result = lhs;
            result ^= rhs;
            return result;
        }

        friend auto operator^=(Bitset& lhs, Bitset const& rhs) -> Bitset& {
            assert(lhs.blocks_.size() == rhs.blocks_.size());
            assert(lhs.numBits_ == rhs.numBits_);
            for (size_t i = 0; i < lhs.blocks_.size(); ++i) {
                lhs[i] ^= rhs[i];
            }
            return lhs;
        };

        friend auto operator~(Bitset const& lhs) -> Bitset {
            auto result = lhs;
            for (size_t i = 0; i < lhs.blocks_.size(); ++i) {
                result[i] = ~result[i];
            }
            return result;
        };

        friend auto operator<<(std::ostream& os, Bitset const& bs) -> std::ostream& {
            os << "{ ";
            for (auto i = 0UL; i < bs.blocks_.size(); ++i) {
                auto b = bs.blocks_[i];
                auto o = BlockSize * i;

                while (b) {
                    auto x = o + __builtin_ctzl(b);
                    b &= (b-1);
                    os << x;
                    if (b || (i < bs.blocks_.size() - 1 && bs.blocks_[i+1] != 0)) {
                        os << ", ";
                    }
                }
            }
            os << " }";
            return os;
        };

        [[nodiscard]] auto ToVec() const -> std::vector<size_t> {
            std::vector<size_t> result;
            result.reserve(PopCount());

            for (auto i = 0UL; i < blocks_.size(); ++i) {
                auto b = blocks_[i];
                auto o = BlockSize * i;

                while (b) {
                    auto x = o + __builtin_ctzl(b);
                    result.push_back(x);
                    b &= (b-1);
                }
            }

            return result;
        }

        template <typename U, std::enable_if_t<std::is_integral_v<U> && std::is_unsigned_v<U>, bool> = true>
        static inline auto CountTrailingZeros(U block) -> size_t // NOLINU
        {
            assert(block != U{0}); // output is undefined for 0!

            constexpr size_t u_bits_number = std::numeric_limits<unsigned>::digits;
            constexpr size_t ul_bits_number = std::numeric_limits<unsigned long>::digits;
            constexpr size_t ull_bits_number = std::numeric_limits<unsigned long long>::digits;
            const size_t bits_per_block = sizeof(U) * CHAR_BIT;

#if defined(__clang__) || defined(__GNUC__)
            if constexpr (bits_per_block <= u_bits_number) {
                return static_cast<size_t>(__builtin_ctz(static_cast<unsigned int>(block)));
            } else if constexpr (bits_per_block <= ul_bits_number) {
                return static_cast<size_t>(__builtin_ctzl(static_cast<unsigned long>(block)));
            } else if constexpr (bits_per_block <= ull_bits_number) {
                return static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(block)));
            }
#elif defined(_MSC_VER)
            constexpr size_t ui64_bits_number = std::numeric_limits<unsigned __int64>::digits;
            if constexpr (bits_per_block <= ul_bits_number) {
                unsigned long index = std::numeric_limits<unsigned long>::max();
                _BitScanForward(&index, static_cast<unsigned long>(block));
                return static_cast<size_t>(index);
            } else if constexpr (bits_per_block <= ui64_bits_number) {
#if (defined(_M_X64) || defined(_M_ARM64))
                unsigned long index = std::numeric_limits<unsigned long>::max();
                _BitScanForward64(&index, static_cast<unsigned __int64>(block));
                return static_cast<size_t>(index);
#else
                constexpr unsigned long max_ul = std::numeric_limits<unsigned long>::max();
                unsigned long low = block & max_ul;
                if (low != 0) {
                    unsigned long index = std::numeric_limits<unsigned long>::max();
                    _BitScanForward(&index, low);
                    return static_cast<size_t>(index);
                }
                unsigned long high = block >> ul_bits_number;
                unsigned long index = std::numeric_limits<unsigned long>::max();
                _BitScanForward(&index, high);
                return static_cast<size_t>(ul_bits_number + index);
#endif
            }
#endif
            U mask = U { 1 };
            for (size_t i = 0; i < bits_per_block; ++i) {
                if ((block & mask) != 0) {
                    return i;
                }
                mask <<= 1;
            }
        }
    };
} // namespace Operon

#endif
