/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef OPERON_COMMON_HPP
#define OPERON_COMMON_HPP

#include "constants.h"
#include "gsl/gsl"
#include "random/jsf.hpp"
#include "random/sfc64.hpp"
#include "xxhash/xxhash.hpp"

#include <random>

namespace Operon {
// we always use 64 bit hash values
namespace operon {
    constexpr uint8_t hash_bits = 64; // can be 32 or 64
    using hash_t                = xxh::hash_t<hash_bits>;
    using rand_t                = Random::Sfc64;
    using scalar_t              = double;

    namespace scalar {
        static inline scalar_t max() 
        { 
            return std::numeric_limits<scalar_t>::max(); 
        }
        static inline scalar_t min() 
        {
            if constexpr (std::is_floating_point_v<scalar_t>) return std::numeric_limits<scalar_t>::lowest(); 
            else return std::numeric_limits<scalar_t>::min(); 
        }
    }
}

class Range {
public:
    inline size_t Start() const noexcept { return range_.first; }
    inline size_t End() const noexcept { return range_.second; }
    inline size_t Size() const noexcept { return range_.second - range_.first; }
    std::pair<size_t, size_t> Bounds() const noexcept { return range_; }

    Range() {}
    Range(size_t start, size_t end)
        : range_(CheckRange(start, end))
    {
    }
    Range(std::pair<size_t, size_t> range)
        : range_(CheckRange(range.first, range.second))
    {
    }

private:
    static std::pair<size_t, size_t> CheckRange(size_t start, size_t end)
    {
        Expects(start <= end);
        return { start, end };
    }
    std::pair<size_t, size_t> range_;
};

// a dataset variable described by: name, hash value (for hashing), data column index
struct Variable {
    std::string Name;
    operon::hash_t Hash;
    gsl::index Index;
};
};

#endif
