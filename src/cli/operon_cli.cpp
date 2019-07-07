#include <fmt/core.h>

#include "operators.hpp"
#include "grammar.hpp"
#include "format.hpp"


int main(int argc, char* argv[])
{
    size_t maxDepth = 5;
    Operon::Grammar grammar;
    Operon::Random::JsfRand<64> jsf;

    const size_t nTrees = 5;

    fmt::print("{}\n", Operon::NodeType::Add < Operon::NodeType::Sub);
    auto trees = grammar.Initialize(jsf, nTrees, 100, maxDepth);
    
    for (size_t i = 0; i < nTrees; ++i)
    {
        trees[i].UpdateNodes();
        for(auto it = trees[i].Nodes().rbegin(); it != trees[i].Nodes().rend(); ++it)
        {
            fmt::print("{}\n", *it);
        }
        fmt::print("\n");
    }

    return 0;
}

