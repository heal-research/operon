#include "operator.hpp"

using namespace std;

namespace Operon
{
    class OnePointMutation : public MutatorBase 
    {
        Tree operator()(RandomDevice& random, const Tree& tree) const override;
    };
}

