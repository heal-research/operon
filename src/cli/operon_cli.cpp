#include <fmt/core.h>

#include "operators.hpp"
#include "grammar.hpp"
#include "format.hpp"


int main(int argc, char* argv[])
{
    size_t maxDepth = 5;
    Operon::Grammar grammar;
    Operon::Random::JsfRand<64> jsf;

    fmt::print("{}\n", Operon::NodeType::Add < Operon::NodeType::Sub);
    auto trees = grammar.Initialize(jsf, 10, 100, maxDepth);
    
    for(auto& node : trees[2].Nodes())
    {
        fmt::print("{}\n", node);
    }

    return 0;
}

