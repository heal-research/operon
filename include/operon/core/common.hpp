#ifndef OPERON_COMMON_HPP
#define OPERON_COMMON_HPP

#include "gsl/gsl"
#include "xxhash/xxhash.hpp"
#include "jsf.hpp"

namespace Operon
{
    // we always use 64 bit hash values
    namespace operon 
    {
        using hash_t = xxh::hash64_t;
        using rand_t = Random::JsfRand<64>;
    }

    struct Range 
    {
        size_t Start;
        size_t End;

        inline int Size() const { return End - Start; }
    };

    // a dataset variable described by: name, hash value (for hashing), data column index
    struct Variable 
    {
        std::string    Name;
        operon::hash_t Hash;
        gsl::index     Index;
    }; 
};

#endif
