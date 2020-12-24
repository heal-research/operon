#ifndef OPERON_RANGE_HPP
#define OPERON_RANGE_HPP

#include "common.hpp"

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
}

#endif
