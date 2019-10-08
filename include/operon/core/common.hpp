#ifndef OPERON_COMMON_HPP
#define OPERON_COMMON_HPP

#include "gsl/gsl"
#include "jsf.hpp"
#include "xxhash/xxhash.hpp"

namespace Operon {
// we always use 64 bit hash values
namespace operon {
    using hash_t   = xxh::hash64_t;
    using rand_t   = Random::JsfRand<64>;
    using scalar_t = double;
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
