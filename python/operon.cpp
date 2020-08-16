#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "algorithms/config.hpp"
#include "core/common.hpp"
#include "core/constants.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/grammar.hpp"
#include "core/metrics.hpp"
#include "core/operator.hpp"

#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/generator.hpp"
#include "operators/mutation.hpp"
#include "operators/selection.hpp"

#include "stat/pearson.hpp"

namespace py = pybind11;

// enable pass-by-reference semantics for this vector type
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Variable>);

PYBIND11_MODULE(pyoperon, m)
{
    m.doc() = "Operon Python Module";
    m.attr("__version__") = 0.1;

    // binding code
    py::bind_vector<std::vector<Operon::Variable>>(m, "VariableCollection");

    // free functions
    // we use a lambda to avoid defining a fourth arg for the defaulted C++ function arg
    m.def("Evaluate", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
                auto result = py::array_t<Operon::Scalar>(r.Size());
                auto buf = result.request();
                Operon::Evaluate(t, d, r, (Operon::Scalar *)nullptr, gsl::span<Operon::Scalar>((Operon::Scalar*)buf.ptr, buf.size));
                return result;
                }, py::arg("tree"), py::arg("dataset"), py::arg("range"));

    // we want to call this from the python side
    m.def("RSquared", [](py::array_t<Operon::Scalar> lhs, py::array_t<Operon::Scalar> rhs) { 
            py::buffer_info x = lhs.request();
            py::buffer_info y = rhs.request();

            auto r = Operon::RSquared(
                    gsl::span<const Operon::Scalar>((Operon::Scalar *)x.ptr, x.size), 
                    gsl::span<const Operon::Scalar>((Operon::Scalar *)y.ptr, y.size)
            );
            return std::isnan(r) ? 0.0 : r * r;
    });

    // classes
    py::class_<Operon::Variable>(m, "Variable")
        .def_readwrite("Name", &Operon::Variable::Name)
        .def_readwrite("Hash", &Operon::Variable::Hash)
        .def_readwrite("Index", &Operon::Variable::Index);

    py::class_<Operon::Range>(m, "Range")
        .def(py::init<size_t, size_t>())
        .def(py::init<std::pair<size_t, size_t>>())
        .def_property_readonly("Start", &Operon::Range::Start)
        .def_property_readonly("End", &Operon::Range::End)
        .def_property_readonly("Size", &Operon::Range::Size);

    // algorithm configuration is being held in a struct on the C++ side
    py::class_<Operon::GeneticAlgorithmConfig>(m, "GeneticAlgorithmConfig")
        .def_readwrite("Generations", &Operon::GeneticAlgorithmConfig::Generations)
        .def_readwrite("Evaluations", &Operon::GeneticAlgorithmConfig::Evaluations)
        .def_readwrite("Iterations", &Operon::GeneticAlgorithmConfig::Iterations)
        .def_readwrite("PopulationSize", &Operon::GeneticAlgorithmConfig::PopulationSize)
        .def_readwrite("PoolSize", &Operon::GeneticAlgorithmConfig::PoolSize)
        .def_readwrite("CrossoverProbability", &Operon::GeneticAlgorithmConfig::CrossoverProbability)
        .def_readwrite("MutationProbability", &Operon::GeneticAlgorithmConfig::MutationProbability)
        .def_readwrite("Seed", &Operon::GeneticAlgorithmConfig::Seed);

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
        .def(~py::self);

    // node
    py::class_<Operon::Node>(m, "Node")
        .def(py::init<Operon::NodeType>())
        .def(py::init<Operon::NodeType, Operon::Hash>())
        .def_property_readonly("Name", &Operon::Node::Name)
        .def_property_readonly("IsLeaf", &Operon::Node::IsLeaf)
        .def_property_readonly("IsConstant", &Operon::Node::IsConstant)
        .def_property_readonly("IsVariable", &Operon::Node::IsVariable)
        .def_property_readonly("IsCommutative", &Operon::Node::IsCommutative)
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
        .def(py::self >= py::self);

    // tree
    py::class_<Operon::Tree>(m, "Tree")
        .def(py::init<std::initializer_list<Operon::Node>>())
        .def(py::init<Operon::Vector<Operon::Node>>())
        .def(py::init<const Operon::Tree&>())
        //.def(py::init<Operon::Tree&&>())
        .def("UpdateNodes", &Operon::Tree::UpdateNodes)
        .def("Sort", &Operon::Tree::Sort)
        .def("Hash", static_cast<Operon::Tree& (Operon::Tree::*)(Operon::HashFunction, Operon::HashMode)>(&Operon::Tree::Hash))
        .def("Reduce", &Operon::Tree::Reduce)
        //.def("Simplify", &Operon::Tree::Simplify) // not yet implemented
        .def("ChildIndices", &Operon::Tree::ChildIndices)
        .def("SetEnabled", &Operon::Tree::SetEnabled)
        .def("SetCoefficients", &Operon::Tree::SetCoefficients)
        .def("GetCoefficients", &Operon::Tree::GetCoefficients)
        .def("CoefficientsCount", &Operon::Tree::CoefficientsCount)
        .def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node>& (Operon::Tree::*)() &>(&Operon::Tree::Nodes))
        .def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node> const& (Operon::Tree::*)() const&>(&Operon::Tree::Nodes))
        //.def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node>&& (Operon::Tree::*)() &&>(&Operon::Tree::Nodes))
        .def_property_readonly("Length", &Operon::Tree::Length)
        .def_property_readonly("VisitationLength", &Operon::Tree::VisitationLength)
        .def_property_readonly("Depth", py::overload_cast<>(&Operon::Tree::Depth, py::const_))
        .def_property_readonly("Depth", py::overload_cast<gsl::index>(&Operon::Tree::Depth, py::const_))
        .def_property_readonly("Level", &Operon::Tree::Level)
        .def_property_readonly("Empty", &Operon::Tree::Empty)
        .def_property_readonly("HashValue", &Operon::Tree::HashValue)
        .def("__getitem__", py::overload_cast<gsl::index>(&Operon::Tree::operator[]))
        .def("__getitem__", py::overload_cast<gsl::index>(&Operon::Tree::operator[], py::const_));

    // grammar
    py::class_<Operon::Grammar>(m, "Grammar")
        .def(py::init<>())
        .def_property_readonly_static("Arithmetic", [](py::object /* self */) { return Operon::Grammar::Arithmetic; })
        .def("IsEnabled", &Operon::Grammar::IsEnabled)
        .def("Enable", &Operon::Grammar::Enable)
        .def("Disable", &Operon::Grammar::Disable)
        .def("GetConfig", &Operon::Grammar::GetConfig)
        .def("SetConfig", &Operon::Grammar::SetConfig)
        .def("GetFrequency", &Operon::Grammar::GetFrequency)
        .def("GetMinimumArity", &Operon::Grammar::GetMinimumArity)
        .def("GetMaximumArity", &Operon::Grammar::GetMaximumArity)
        .def_property_readonly("EnabledSymbols", &Operon::Grammar::EnabledSymbols)
        .def("FunctionArityLimits", &Operon::Grammar::FunctionArityLimits)
        .def("SampleRandomSymbol", &Operon::Grammar::SampleRandomSymbol);

    // dataset
    py::class_<Operon::Dataset>(m, "Dataset")
        .def(py::init<const std::string&, bool>(), py::arg("filename"), py::arg("has_header"))
        .def(py::init<const Operon::Dataset&>())
        .def(py::init<const std::vector<Operon::Variable>&, const std::vector<std::vector<Operon::Scalar>>&>())
        .def_property_readonly("Rows", &Operon::Dataset::Rows)
        .def_property_readonly("Cols", &Operon::Dataset::Cols)
        .def_property_readonly("Values", &Operon::Dataset::Values)
        .def_property_readonly("VariableNames", &Operon::Dataset::VariableNames)
        .def("GetValues", py::overload_cast<const std::string&>(&Operon::Dataset::GetValues, py::const_))
        .def("GetValues", py::overload_cast<Operon::Hash>(&Operon::Dataset::GetValues, py::const_))
        .def("GetValues", py::overload_cast<gsl::index>(&Operon::Dataset::GetValues, py::const_))
        .def("GetVariable", py::overload_cast<const std::string&>(&Operon::Dataset::GetVariable, py::const_))
        .def("GetVariable", py::overload_cast<Operon::Hash>(&Operon::Dataset::GetVariable, py::const_))
        .def_property_readonly("Variables", [](const Operon::Dataset& self) {
            auto vars = self.Variables();
            return std::vector<Operon::Variable>(vars.begin(), vars.end());
                })
        .def("Shuffle", &Operon::Dataset::Shuffle)
        .def("Normalize", &Operon::Dataset::Normalize)
        .def("Standardize", &Operon::Dataset::Standardize);

    // tree creator
    py::class_<Operon::BalancedTreeCreator>(m, "BalancedTreeCreator")
        .def(py::init([](Operon::Grammar const& grammar, std::vector<Operon::Variable> const& variables, double bias){
            return Operon::BalancedTreeCreator(grammar, gsl::span<const Operon::Variable>(variables.data(), variables.size()), bias); 
        }), py::arg("grammar"), py::arg("variables"), py::arg("bias"))
        .def("__call__", &Operon::BalancedTreeCreator::operator())
        .def_property("IrregularityBias", &Operon::BalancedTreeCreator::GetBias, &Operon::BalancedTreeCreator::SetBias);

    py::class_<Operon::ProbabilisticTreeCreator>(m, "ProbabilisticTreeCreator")
        .def(py::init<const Operon::Grammar&, const std::vector<Operon::Variable>>())
        .def("__call__", &Operon::ProbabilisticTreeCreator::operator());

    py::class_<Operon::GrowTreeCreator>(m, "GrowTreeCreator")
        .def(py::init<const Operon::Grammar&, const std::vector<Operon::Variable>>())
        .def("__call__", &Operon::GrowTreeCreator::operator());

    // crossover
    py::class_<Operon::SubtreeCrossover>(m, "SubtreeCrossover")
        .def(py::init<double, size_t, size_t>())
        .def("__call__", &Operon::SubtreeCrossover::operator());

    // mutation
    py::class_<Operon::OnePointMutation>(m, "OnePointMutation")
        .def(py::init<>())
        .def("__call__", &Operon::OnePointMutation::operator());

    py::class_<Operon::ChangeVariableMutation>(m, "ChangeVariableMutation")
        .def(py::init<const gsl::span<const Operon::Variable>>())
        .def("__call__", &Operon::ChangeVariableMutation::operator());

    py::class_<Operon::ChangeFunctionMutation>(m, "ChangeFunctionMutation")
        .def(py::init<Operon::Grammar>())
        .def("__call__", &Operon::ChangeFunctionMutation::operator());

    py::class_<Operon::ReplaceSubtreeMutation>(m, "ReplaceSubtreeMutation")
        .def(py::init<Operon::CreatorBase&, size_t, size_t>())
        .def("__call__", &Operon::ReplaceSubtreeMutation::operator());

    // selection
    py::class_<Operon::TournamentSelector>(m, "TournamentSelector")
        .def(py::init<Operon::ComparisonCallback>())
        .def("__call__", &Operon::TournamentSelector::operator())
        .def_property("TournamentSize", &Operon::TournamentSelector::GetTournamentSize, &Operon::TournamentSelector::SetTournamentSize);

    py::class_<Operon::RankTournamentSelector>(m, "RankTournamentSelector")
        .def(py::init<Operon::ComparisonCallback>())
        .def("__call__", &Operon::RankTournamentSelector::operator())
        .def("Prepare", &Operon::RankTournamentSelector::Prepare)
        .def_property("TournamentSize", &Operon::RankTournamentSelector::GetTournamentSize, &Operon::RankTournamentSelector::SetTournamentSize);

    py::class_<Operon::ProportionalSelector>(m, "ProportionalSelector")
        .def(py::init<Operon::ComparisonCallback>())
        .def("__call__", &Operon::ProportionalSelector::operator())
        .def("Prepare", py::overload_cast<const gsl::span<const Operon::Individual>>(&Operon::ProportionalSelector::Prepare, py::const_))
        .def("SetObjIndex", &Operon::ProportionalSelector::SetObjIndex);

    // random generators
    py::class_<Operon::RandomGenerator::RomuTrio>(m, "RomuTrio")
        .def(py::init<uint64_t>())
        .def("__call__", &Operon::RandomGenerator::RomuTrio::operator());

    // tree format
    py::class_<Operon::TreeFormatter>(m, "TreeFormatter")
        .def_static("Format", &Operon::TreeFormatter::Format);

    py::class_<Operon::InfixFormatter>(m, "InfixFormatter")
        .def_static("Format", &Operon::InfixFormatter::Format);
}
