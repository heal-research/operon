#ifndef FORMATTERS_HPP
#define FORMATTERS_HPP

#include "dataset.hpp"
#include "tree.hpp"

namespace Operon 
{
    class TreeFormatter 
    {
        static void FormatNode(const Tree& tree, const Dataset& dataset, size_t i, std::string& current, std::string indent, bool isLast, bool initialMarker, int decimalPrecision)
        {
            current += indent;

            if (initialMarker)
            {
                current += isLast ? "└── " : "├── "; 
            }

            auto& s = tree[i];
            if (s.IsConstant()) 
            {
                auto formatString = fmt::format("{{:.{}f}}\n", decimalPrecision);
                fmt::format_to(std::back_inserter(current), formatString, s.Value);
            } 
            else if (s.IsVariable())
            {
                fmt::format_to(std::back_inserter(current), "{} [{}]\n", dataset.GetName(s.CalculatedHashValue), s.CalculatedHashValue);
            } 
            else
            {
                fmt::format_to(std::back_inserter(current), "{}\n", s.Name());
            }

            if (s.IsLeaf) { return; }

            if (i != tree.Length() - 1)
            {
                indent += isLast ? "    " : "|   "; 
            }

            for (auto it = tree.Children(i); it.HasNext(); ++it)            
            {
                FormatNode(tree, dataset, it.Index(), current, indent, it.Count() + 1 == s.Arity, true, decimalPrecision);
            }
        }

        public:
        static std::string Format(const Tree& tree, const Dataset& dataset, int decimalPrecision = 2)
        {
            std::string result;
            FormatNode(tree, dataset, tree.Length() - 1, result, "", true, false, decimalPrecision);
            return result;
        }
    };

    class InfixFormatter 
    {
        static void FormatNode(const Tree& tree, const Dataset& dataset, size_t i, std::string& current, int decimalPrecision)
        {
            auto& s = tree[i];
            if (s.IsConstant()) 
            {
                auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}})" : "{{:.{}f}}", decimalPrecision);
                fmt::format_to(std::back_inserter(current), formatString, s.Value);
            } 
            else if (s.IsVariable())
            {
                auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}}) * {{}}" : "{{:.{}f}} * {{}}", decimalPrecision);
                fmt::format_to(std::back_inserter(current), formatString, s.Value, dataset.GetName(s.CalculatedHashValue));
            } 
            else
            {
                if (s.Type < NodeType::Log) // add, sub, mul, div 
                {
                    fmt::format_to(std::back_inserter(current), "(");
                    for (auto it = tree.Children(i); it.HasNext(); ++it)
                    {
                        FormatNode(tree, dataset, it.Index(), current, decimalPrecision);
                        if (it.Count() + 1 < s.Arity)
                        {
                            fmt::format_to(std::back_inserter(current), " {} ", s.Name());
                        }
                    }
                    fmt::format_to(std::back_inserter(current), ")");
                }
                else // unary operators log, exp, sin, etc. 
                {
                    auto name = s.Name();
                    fmt::format_to(std::back_inserter(current), "{}", name);
                    if (tree[i-1].IsLeaf)
                    {
                        // surround a single leaf argument with parantheses
                        fmt::format_to(std::back_inserter(current), "(");
                        FormatNode(tree, dataset, i - 1, current, decimalPrecision);
                        fmt::format_to(std::back_inserter(current), ")");
                    } 
                    else 
                    {
                        FormatNode(tree, dataset, i - 1, current, decimalPrecision);
                    }
                }
            }
        }

        public:
        static std::string Format(const Tree& tree, const Dataset& dataset, int decimalPrecision = 2)
        {
            std::string result;
            FormatNode(tree, dataset, tree.Length() - 1, result, decimalPrecision);
            return result;
        }
    };
}

#endif

