#ifndef INITIALIZATION_HPP
#define INITIALIZATION_HPP

#include "core/operator.hpp"

namespace Operon 
{
    class Grammar; // forward declaration

    class GrowTreeCreator : public CreatorBase 
    {
        public:
            GrowTreeCreator(size_t depth, size_t length) : maxDepth(depth), maxLength(length) { }
            Tree operator()(operon::rand_t& random, const Grammar& grammar, const gsl::span<const Variable> variables) const; 

        private:
            const size_t maxDepth;
            const size_t maxLength;
    };

}
#endif
