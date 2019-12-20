#ifndef OPERON_TYPES_HPP
#define OPERON_TYPES_HPP

#include <cstdint>
#include "xxhash/xxhash.hpp"
#include "random/sfc64.hpp"
#include "random/jsf.hpp"

#include <Eigen/Core>

namespace Operon {
    constexpr uint8_t HashBits = 64; // can be 32 or 64
    using Hash                 = xxh::hash_t<HashBits>;
    using Random               = RandomGenerator::Sfc64;
    using Scalar               = double;

    // Operon::Vector is just an aligned std::vector 
    // alignment can be controlled with the EIGEN_MAX_ALIGN_BYTES macro
    // https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html#TopicPreprocessorDirectivesPerformance
    template<typename T>
    using Vector = std::vector<T, Eigen::aligned_allocator<T>>;

    namespace Numeric {
        template<typename T> 
        static inline T Max() 
        {
            return std::numeric_limits<T>::max();
        }
        template<typename T>
        static inline T Min() 
        {
            if constexpr (std::is_floating_point_v<T>) return std::numeric_limits<T>::lowest(); 
            else return std::numeric_limits<T>::min(); 
        }
    }
} // namespace operon

#endif

