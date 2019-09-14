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
            size_t maxDepth;
            size_t maxLength;
    };

    class FullTreeCreator : public CreatorBase
    {
        public:
            FullTreeCreator(size_t depth, size_t length) : maxDepth(depth), maxLength(length) { } 
            Tree operator()(operon::rand_t& random, const Grammar& grammar, const gsl::span<const Variable> variables) const; 

        private:
            size_t maxDepth;
            size_t maxLength;
    };

    class RampedHalfAndHalfCreator : public CreatorBase
    {
        public:
            RampedHalfAndHalfCreator(size_t depth, size_t length) : 
                grow(depth, length), 
                full(static_cast<size_t>(std::log2(length)), length) 
            { }

            Tree operator()(operon::rand_t& random, const Grammar& grammar, const gsl::span<const Variable> variables) const
            {
                std::uniform_real_distribution<double> uniformReal(0, 1);

                if (uniformReal(random) < 0.5)
                {
                    return grow(random, grammar, variables);
                }
                return full(random, grammar, variables);
            }

        private:
            GrowTreeCreator grow;
            FullTreeCreator full;
    };
}
#endif
