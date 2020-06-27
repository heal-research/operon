#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "core/common.hpp"
#include "core/constants.hpp"
#include "core/eval.hpp"
#include "core/grammar.hpp"
#include "algorithms/config.hpp"

#include "operators/creator.hpp"
#include "operators/generator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(operon, m) {
    m.doc() = "Operon Python Module";
    m.attr("__version__") = 0.1;

    // algorithm configuration is being held in a struct on the C++ side
    py::class_<Operon::GeneticAlgorithmConfig>(m, "GeneticAlgorithmConfig")
        .def_readwrite("Generations",          &Operon::GeneticAlgorithmConfig::Generations)
        .def_readwrite("Evaluations",          &Operon::GeneticAlgorithmConfig::Evaluations)
        .def_readwrite("Iterations",           &Operon::GeneticAlgorithmConfig::Iterations)
        .def_readwrite("PopulationSize",       &Operon::GeneticAlgorithmConfig::PopulationSize)
        .def_readwrite("PoolSize",             &Operon::GeneticAlgorithmConfig::PoolSize)
        .def_readwrite("CrossoverProbability", &Operon::GeneticAlgorithmConfig::CrossoverProbability)
        .def_readwrite("MutationProbability",  &Operon::GeneticAlgorithmConfig::MutationProbability)
        .def_readwrite("Seed",                 &Operon::GeneticAlgorithmConfig::Seed)
        ;

    // node type
    py::enum_<Operon::NodeType>(m, "NodeType")
        .value("Add", Operon::NodeType::Add)
        .value("Mul", Operon::NodeType::Mul)
        .value("Sub", Operon::NodeType::Sub)
        .value("Div", Operon::NodeType::Div)
        .value("Log", Operon::NodeType::Log)
        .value("Exp", Operon::NodeType::Exp)
        .value("Sin", Operon::NodeType::Sin)
        .value("Cos", Operon::NodeType::Cos)
        .value("Tab", Operon::NodeType::Tan)
        .value("Sqrt", Operon::NodeType::Sqrt)
        .value("Cbrt", Operon::NodeType::Cbrt)
        .value("Square", Operon::NodeType::Square)
        .value("Constant", Operon::NodeType::Constant)
        .value("Variable", Operon::NodeType::Variable)
        // expose overloaded operators
        .def(py::self & py::self)
        .def(py::self &= py::self)
        .def(py::self | py::self)
        .def(py::self |= py::self)
        .def(py::self ^ py::self)
        .def(py::self ^= py::self)
        .def(~py::self)
        ;

    // node
    py::class_<Operon::Node>(m, "Node")
        .def(py::init<Operon::NodeType>())
        .def(py::init<Operon::NodeType, Operon::Hash>())
        .def("Name",  &Operon::Node::Name)
        .def("IsLeaf",  &Operon::Node::IsLeaf)
        .def("IsCommutative",  &Operon::Node::IsCommutative)
        .def_readwrite("Value", &Operon::Node::Value)
        .def_readwrite("HashValue", &Operon::Node::HashValue)
        .def_readwrite("CalculatedHashValue", &Operon::Node::CalculatedHashValue)
        .def_readwrite("Arity", &Operon::Node::Arity)
        .def_readwrite("Length", &Operon::Node::Length)
        .def_readwrite("Depth", &Operon::Node::Depth)
        .def_readwrite("Parent", &Operon::Node::Parent)
        .def_readwrite("Type", &Operon::Node::Type)
        .def_readwrite("IsEnabled", &Operon::Node::IsEnabled)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self >= py::self)
        ;

    // tree
    py::class_<Operon::Tree>(m, "Tree") 
        .def(py::init<std::initializer_list<Operon::Node>>())
        .def(py::init<std::vector<Operon::Node>>())
        .def(py::init<const Operon::Tree&>())
        .def("UpdateNodes", &Operon::Tree::UpdateNodes)
        .def("UpdateNodeDepth", &Operon::Tree::UpdateNodeDepth)
        .def("Sort", &Operon::Tree::Sort)
        .def("Reduce", &Operon::Tree::Reduce)
        .def("Simplify", &Operon::Tree::Simplify)
        .def("ChildIndices", &Operon::Tree::ChildIndices)
        .def("SetEnabled", &Operon::Tree::SetEnabled)
        .def("Nodes", py::overload_cast<>(&Operon::Tree::Nodes))
        .def("NodesConst", py::overload_cast<>(&Operon::Tree::Nodes, py::const_))
        .def("SetCoefficients", &Operon::Tree::SetCoefficients)
        .def("GetCoefficients", &Operon::Tree::GetCoefficients)
        .def("CoefficientsCount", &Operon::Tree::CoefficientsCount)
        .def("Length", &Operon::Tree::Length)
        .def("VisitationLength", &Operon::Tree::VisitationLength)
        .def("Depth", py::overload_cast<>(&Operon::Tree::Depth, py::const_))
        .def("DepthIndex", py::overload_cast<gsl::index>(&Operon::Tree::Depth, py::const_))
        .def("Level", &Operon::Tree::Level)
        .def("Empty", &Operon::Tree::Empty)
        .def("HashValue", &Operon::Tree::HashValue)
        ;

    // grammar
    py::class_<Operon::Grammar>(m, "Grammar")
        .def("IsEnabled", &Operon::Grammar::IsEnabled)
        .def("Enable", &Operon::Grammar::Enable)
        .def("Disable", &Operon::Grammar::Disable)
        .def("GetConfig", &Operon::Grammar::GetConfig)
        .def("SetConfig", &Operon::Grammar::SetConfig)
        .def("GetFrequency", &Operon::Grammar::GetFrequency)
        .def("EnabledSymbols", &Operon::Grammar::EnabledSymbols)
        .def("FunctionArityLimits", &Operon::Grammar::FunctionArityLimits)
        .def("SampleRandomSymbol", &Operon::Grammar::SampleRandomSymbol)
        ;

    // dataset
    py::class_<Operon::Dataset>(m, "Dataset")
        .def(py::init<const std::string&, bool>())
        .def(py::init<const Operon::Dataset&>())
        .def(py::init<const std::vector<Operon::Variable>&, const std::vector<std::vector<Operon::Scalar>>&>()) 
        .def("Rows", &Operon::Dataset::Rows)
        .def("Cols", &Operon::Dataset::Cols)
        .def("Values", &Operon::Dataset::Values)
        .def("VariableNames", &Operon::Dataset::VariableNames)
        .def("GetValues", py::overload_cast<const std::string&>(&Operon::Dataset::GetValues, py::const_))
        .def("GetValues", py::overload_cast<Operon::Hash>(&Operon::Dataset::GetValues, py::const_))
        .def("GetValues", py::overload_cast<gsl::index>(&Operon::Dataset::GetValues, py::const_))
        .def("GetName", py::overload_cast<Operon::Hash>(&Operon::Dataset::GetName, py::const_))
        .def("GetName", py::overload_cast<gsl::index>(&Operon::Dataset::GetName, py::const_))
        .def("GetHashValue", &Operon::Dataset::GetHashValue)
        .def("GetIndex", &Operon::Dataset::GetIndex)
        .def("Shuffle", &Operon::Dataset::Shuffle)
        .def("Normalize", &Operon::Dataset::Normalize)
        .def("Standardize", &Operon::Dataset::Standardize)
        ;

    // tree creator 
    py::class_<Operon::BalancedTreeCreator>(m, "BalancedTreeCreator")
        .def(py::init<const Operon::Grammar&, const gsl::span<const Operon::Variable>, double>())
        .def("__call__", &Operon::BalancedTreeCreator::operator())
        ;

    py::class_<Operon::ProbabilisticTreeCreator>(m, "ProbabilisticTreeCreator")
        .def(py::init<const Operon::Grammar&, const gsl::span<const Operon::Variable>>())
        .def("__call__", &Operon::ProbabilisticTreeCreator::operator())
        ;

    py::class_<Operon::GrowTreeCreator>(m, "GrowTreeCreator")
        .def(py::init<const Operon::Grammar&, const gsl::span<const Operon::Variable>>())
        .def("__call__", &Operon::GrowTreeCreator::operator())
        ;
}
