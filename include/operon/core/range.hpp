// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_RANGE_HPP
#define OPERON_RANGE_HPP

#include "core/contracts.hpp"

#include <tuple>

namespace Operon {
class Range {
    public:
        inline std::size_t Start() const noexcept { return range_.first; }
        inline std::size_t End() const noexcept { return range_.second; }
        inline std::size_t Size() const noexcept { return range_.second - range_.first; }
        std::pair<std::size_t, std::size_t> Bounds() const noexcept { return range_; }

        Range() {}
        Range(std::size_t start, std::size_t end)
            : range_(CheckRange(start, end))
        {
        }
        Range(std::pair<std::size_t, std::size_t> range)
            : range_(CheckRange(range.first, range.second))
        {
        }

    private:
        static std::pair<std::size_t, std::size_t> CheckRange(std::size_t start, std::size_t end)
        {
            EXPECT(start <= end);
            return { start, end };
        }
        std::pair<std::size_t, std::size_t> range_;
};
}

#endif
