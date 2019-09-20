#ifndef CROSSOVER_HPP
#define CROSSOVER_HPP

#include <vector>

#include "core/operator.hpp"

namespace Operon {
class SubtreeCrossover : public CrossoverBase {
public:
    SubtreeCrossover(double p, size_t d, size_t l)
        : internalProbability(p)
        , maxDepth(d)
        , maxLength(l)
    {
    }
    auto operator()(operon::rand_t& random, const Tree& lhs, const Tree& rhs) const -> Tree override;

private:
    double internalProbability;
    long maxDepth;
    long maxLength;
};
}
#endif
