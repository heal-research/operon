#ifndef INITIALIZATION_HPP
#define INITIALIZATION_HPP

#include "operator.hpp"

namespace Operon 
{
    class Grammar; // forward declaration

    class GrowTreeCreator : public CreatorBase 
    {
        public:
            GrowTreeCreator(size_t depth, size_t length) : maxDepth(depth), maxLength(length) { }
            Tree operator()(RandomDevice& random, const Grammar& grammar, const std::vector<Variable>& variables) const; 

        private:
            const size_t maxDepth;
            const size_t maxLength;
    };

}
#endif
