#ifndef OPERON_TYPES_HPP
#define OPERON_TYPES_HPP

#include <cstdint>
#include "xxhash/xxhash.hpp"
#include "random/random.hpp"

#include <ceres/jet.h>
#include <Eigen/Core>

namespace Operon {
    constexpr std::uint8_t HashBits = 64; // can be 32 or 64
    using Hash                 = xxh::hash_t<HashBits>;
    using Random               = RandomGenerator::RomuTrio;
    //using Random               = RandomGenerator::Sfc64;
#ifdef USE_SINGLE_PRECISION
    using Scalar               = float;
#else 
    using Scalar               = double;
#endif
    using Dual                 = ceres::Jet<Scalar, 4>;

    // Operon::Vector is just an aligned std::vector 
    // alignment can be controlled with the EIGEN_MAX_ALIGN_BYTES macro
    // https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html#TopicPreprocessorDirectivesPerformance
    template<typename T>
    using Vector               = std::vector<T, Eigen::aligned_allocator<T>>;

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
            else if constexpr(std::is_same_v<T, Dual>) return T{std::numeric_limits<Scalar>::lowest()};
            else return std::numeric_limits<T>::min(); 
        }
    }
} // namespace operon

#endif

