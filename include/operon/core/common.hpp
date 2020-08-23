/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
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

#include "constants.hpp"
#include "contracts.hpp"
#include "types.hpp"

namespace Operon {

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
        EXPECT(start <= end);
        return { start, end };
    }
    std::pair<size_t, size_t> range_;
};

// a dataset variable described by: name, hash value (for hashing), data column index
struct Variable {
    std::string Name;
    Operon::Hash Hash;
    size_t Index;
};
}

#endif
