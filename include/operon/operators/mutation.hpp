#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "core/operator.hpp"

namespace Operon 
{
    struct OnePointMutation : public MutatorBase 
    {
        void operator()(operon::rand_t& random, Tree& tree) const override;
    };

    struct MultiPointMutation : public MutatorBase 
    {
        void operator()(operon::rand_t& random, Tree& tree) const override;
    };
}

#endif

