#ifndef OPERON_COMMON_HPP
#define OPERON_COMMON_HPP

#include "gsl/gsl"
#include "xxhash/xxhash.hpp"

namespace Operon
{
    // we always use 64 bit hash values
    namespace operon 
    {
        using hash_t = xxh::hash64_t;
    }
};

#endif
