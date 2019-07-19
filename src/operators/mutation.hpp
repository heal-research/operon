#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "operator.hpp"

namespace Operon 
{
    struct OnePointMutation : public MutatorBase 
    {
        Tree operator()(RandomDevice& random, const Tree& tree) const override;
    };
}
#endif

