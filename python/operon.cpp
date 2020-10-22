/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <type_traits>

#include "algorithms/gp.hpp"
#include "core/common.hpp"
#include "core/constants.hpp"
#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/pset.hpp"
#include "core/metrics.hpp"
#include "core/operator.hpp"

#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/evaluator.hpp"
#include "operators/generator.hpp"
#include "operators/mutation.hpp"
#include "operators/selection.hpp"
#include "operators/reinserter/replaceworst.hpp"

#include "stat/pearson.hpp"

namespace py = pybind11;

// enable pass-by-reference semantics for this vector type
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Variable>)
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Individual>)

template<typename T>
Operon::Dataset MakeDataset(py::array_t<T> array)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");

    // sanity check
    if (array.ndim() != 2) {
        throw std::runtime_error("The input array must have exactly two dimensions.");
    }

    // check if the array satisfies our data storage requirements (contiguous, column-major order)
    if (std::is_same_v<T, Operon::Scalar> && (array.flags() & py::array::f_style)) {
        auto ref = array.template cast<Eigen::Ref<Operon::Dataset::Matrix const>>();
        return Operon::Dataset(ref);
    } else {
        fmt::print("Warning: array does not satisfy contiguity or storage-order requirements. a copy will be made.");
        auto m = array.template cast<Operon::Dataset::Matrix>();
        return Operon::Dataset(std::move(m));
    }
}

template<typename T>
py::array_t<T const> MakeView(gsl::span<T const> view)
{
    auto sz = static_cast<pybind11::ssize_t>(view.size());
    py::array_t<T const> arr(sz, view.data(), py::capsule(view.data()));
    ENSURE(arr.owndata() == false);
    ENSURE(arr.data() == view.data());
    return arr;
}

PYBIND11_MODULE(pyoperon, m)
{
    m.doc() = "Operon Python Module";
    m.attr("__version__") = 0.1;

    // binding code
    py::bind_vector<std::vector<Operon::Variable>>(m, "VariableCollection");
    py::bind_vector<std::vector<Operon::Individual>>(m, "IndividualCollection");

    // free functions
    // we use a lambda to avoid defining a fourth arg for the defaulted C++ function arg
    m.def("Evaluate", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
        auto result = py::array_t<Operon::Scalar>(static_cast<pybind11::ssize_t>(r.Size()));
        auto buf = result.request();
        auto res = gsl::span<Operon::Scalar>((Operon::Scalar*)buf.ptr, r.Size());
        Operon::Evaluate(t, d, r, res, static_cast<Operon::Scalar*>(nullptr));
        return result;
        }, py::arg("tree"), py::arg("dataset"), py::arg("range"));

    m.def("CalculateFitness", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        auto estimated = Operon::Evaluate(t, d, r, (Operon::Scalar*)nullptr);
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        if (metric == "rsquared") return Operon::RSquared(estimated, values); 
        if (metric == "mse")      return Operon::MeanSquaredError(estimated, values);
        if (metric == "rmse")     return Operon::RootMeanSquaredError(estimated, values);
        if (metric == "nmse")     return Operon::NormalizedMeanSquaredError(estimated, values);
        throw std::runtime_error("Invalid fitness metric"); 

    }, py::arg("tree"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");

    m.def("CalculateFitness", [](std::vector<Operon::Tree> const& trees, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        std::add_pointer<double(gsl::span<const Operon::Scalar>, gsl::span<const Operon::Scalar>)>::type func;
        if (metric == "rsquared")  func = Operon::RSquared;
        else if (metric == "mse")  func = Operon::MeanSquaredError;
        else if (metric == "nmse") func = Operon::NormalizedMeanSquaredError;
        else if (metric == "rmse") func = Operon::RootMeanSquaredError;
        else throw std::runtime_error("Unsupported error metric");

        auto result = py::array_t<double>(static_cast<pybind11::ssize_t>(trees.size()));
        auto buf = result.request();
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        std::transform(std::execution::par, trees.begin(), trees.end(), (double*)buf.ptr, [&](auto const& t) -> double {
            auto estimated = Operon::Evaluate(t, d, r, (Operon::Scalar*)nullptr);
            return func(estimated, values);
        });

        return result;
    }, py::arg("trees"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");

    // we want to call this from the python side
    m.def("RSquared", [](py::array_t<Operon::Scalar> lhs, py::array_t<Operon::Scalar> rhs) {
        py::buffer_info x = lhs.request();
        py::buffer_info y = rhs.request();
        gsl::span<const Operon::Scalar> sx((Operon::Scalar*)x.ptr, static_cast<decltype(sx)::size_type>(x.size));
        gsl::span<const Operon::Scalar> sy((Operon::Scalar*)y.ptr, static_cast<decltype(sy)::size_type>(y.size));
        auto r = Operon::RSquared(sx, sy);
        return std::isnan(r) ? Operon::Numeric::Min<Operon::Scalar>() : r;
    });

    m.def("NormalizedMeanSquaredError", [](py::array_t<Operon::Scalar> lhs, py::array_t<Operon::Scalar> rhs) {
        py::buffer_info x = lhs.request();
        py::buffer_info y = rhs.request();
        gsl::span<const Operon::Scalar> sx((Operon::Scalar*)x.ptr, static_cast<decltype(sx)::size_type>(x.size));
        gsl::span<const Operon::Scalar> sy((Operon::Scalar*)y.ptr, static_cast<decltype(sy)::size_type>(y.size));
        auto r = Operon::NormalizedMeanSquaredError(sx, sy);
        return std::isnan(r) ? Operon::Numeric::Max<Operon::Scalar>() : r;
    });

    m.def("RootMeanSquaredError", [](py::array_t<Operon::Scalar> lhs, py::array_t<Operon::Scalar> rhs) {
        py::buffer_info x = lhs.request();
        py::buffer_info y = rhs.request();
        gsl::span<const Operon::Scalar> sx((Operon::Scalar*)x.ptr, static_cast<decltype(sx)::size_type>(x.size));
        gsl::span<const Operon::Scalar> sy((Operon::Scalar*)y.ptr, static_cast<decltype(sy)::size_type>(y.size));
        auto r = Operon::RootMeanSquaredError(sx, sy);
        return std::isnan(r) ? Operon::Numeric::Max<Operon::Scalar>()  : r;
    });

    m.def("MeanSquaredError", [](py::array_t<Operon::Scalar> lhs, py::array_t<Operon::Scalar> rhs) {
        py::buffer_info x = lhs.request();
        py::buffer_info y = rhs.request();
        gsl::span<const Operon::Scalar> sx((Operon::Scalar*)x.ptr, static_cast<decltype(sx)::size_type>(x.size));
        gsl::span<const Operon::Scalar> sy((Operon::Scalar*)y.ptr, static_cast<decltype(sy)::size_type>(y.size));
        auto r = Operon::MeanSquaredError(sx, sy);
        return std::isnan(r) ? Operon::Numeric::Max<Operon::Scalar>() : r;
    });

    // random numbers
    m.def("UniformInt", &Operon::Random::Uniform<Operon::RandomGenerator, int>);
    m.def("UniformReal", &Operon::Random::Uniform<Operon::RandomGenerator, double>);

    // classes
    py::class_<Operon::Individual>(m, "Individual")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Individual::operator[]))
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Individual::operator[], py::const_))
        .def_readwrite("Genotype", &Operon::Individual::Genotype)
        .def("SetFitness", [](Operon::Individual& self, Operon::Scalar f, size_t i) { self[i] = f; })
        .def("GetFitness", [](Operon::Individual& self, size_t i) { return self[i]; });

    py::class_<Operon::Comparison>(m, "Comparison");

    py::class_<Operon::SingleObjectiveComparison, Operon::Comparison>(m, "SingleObjectiveComparison")
        .def(py::init<size_t>())
        .def("__call__", &Operon::SingleObjectiveComparison::operator());
    
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
        .def_readwrite("Level", &Operon::Node::Level)
        .def_readwrite("Parent", &Operon::Node::Parent)
        .def_readwrite("Type", &Operon::Node::Type)
        .def_readwrite("IsEnabled", &Operon::Node::IsEnabled)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self >= py::self)
        // pickle support
        .def(py::pickle(
            [](Operon::Node const& n) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(
                    n.HashValue,
                    n.CalculatedHashValue,
                    n.Value,
                    n.Arity,
                    n.Length,
                    n.Depth,
                    n.Parent,
                    n.Type,
                    n.IsEnabled
                );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 9) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                Operon::Node n(t[7].cast<Operon::NodeType>());

                /* Assign any additional state */
                n.HashValue           = t[0].cast<Operon::Hash>();
                n.CalculatedHashValue = t[1].cast<Operon::Hash>();
                n.Value               = t[2].cast<Operon::Scalar>();
                n.Arity               = t[3].cast<uint16_t>();
                n.Length              = t[4].cast<uint16_t>();
                n.Depth               = t[5].cast<uint16_t>();
                n.Parent              = t[6].cast<uint16_t>();
                n.IsEnabled           = t[8].cast<bool>();

                return n;
            }
        ))
        ;

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
        .def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node>& (Operon::Tree::*)()&>(&Operon::Tree::Nodes))
        .def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node> const& (Operon::Tree::*)() const&>(&Operon::Tree::Nodes))
        //.def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node>&& (Operon::Tree::*)() &&>(&Operon::Tree::Nodes))
        .def_property_readonly("Length", &Operon::Tree::Length)
        .def_property_readonly("VisitationLength", &Operon::Tree::VisitationLength)
        .def_property_readonly("Depth", static_cast<size_t (Operon::Tree::*)() const>(&Operon::Tree::Depth))
//        .def_property_readonly("Depth", static_cast<size_t (Operon::Tree::*)(size_t) const>(&Operon::Tree::Depth))
        .def_property_readonly("Empty", &Operon::Tree::Empty)
        .def_property_readonly("HashValue", &Operon::Tree::HashValue)
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Tree::operator[]))
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Tree::operator[], py::const_))
        .def(py::pickle(
            [](Operon::Tree const& tree) {
                return py::make_tuple(tree.Nodes());
            },
            [](py::tuple t) {
                if (t.size() != 1) {
                    throw std::runtime_error("Invalid state!");
                }

                return Operon::Tree(t[0].cast<Operon::Vector<Operon::Node>>()).UpdateNodes();
            }
        ));

    // grammar
    py::class_<Operon::PrimitiveSet>(m, "PrimitiveSet")
        .def(py::init<>())
        .def_property_readonly_static("Arithmetic", [](py::object /* self */) { return Operon::PrimitiveSet::Arithmetic; })
        .def("IsEnabled", &Operon::PrimitiveSet::IsEnabled)
        .def("Enable", &Operon::PrimitiveSet::Enable)
        .def("Disable", &Operon::PrimitiveSet::Disable)
        .def("GetConfig", &Operon::PrimitiveSet::GetConfig)
        .def("SetConfig", &Operon::PrimitiveSet::SetConfig)
        .def("GetFrequency", &Operon::PrimitiveSet::GetFrequency)
        .def("GetMinimumArity", &Operon::PrimitiveSet::GetMinimumArity)
        .def("GetMaximumArity", &Operon::PrimitiveSet::GetMaximumArity)
        .def("GetMinMaxArity", &Operon::PrimitiveSet::GetMinMaxArity)
        .def_property_readonly("EnabledSymbols", &Operon::PrimitiveSet::EnabledSymbols)
        .def("FunctionArityLimits", &Operon::PrimitiveSet::FunctionArityLimits)
        .def("SampleRandomSymbol", &Operon::PrimitiveSet::SampleRandomSymbol);

    // dataset
    py::class_<Operon::Dataset>(m, "Dataset")
        .def(py::init<std::string const&, bool>(), py::arg("filename"), py::arg("has_header"))
        .def(py::init<Operon::Dataset const&>())
        .def(py::init<std::vector<Operon::Variable> const&, const std::vector<std::vector<Operon::Scalar>>&>())
        .def(py::init([](py::array_t<float> array){ return MakeDataset(array); }), py::arg("data").noconvert())
        .def(py::init([](py::array_t<double> array){ return MakeDataset(array); }), py::arg("data").noconvert())
        .def_property_readonly("Rows", &Operon::Dataset::Rows)
        .def_property_readonly("Cols", &Operon::Dataset::Cols)
        .def_property_readonly("Values", &Operon::Dataset::Values)
        .def_property("VariableNames", &Operon::Dataset::VariableNames, &Operon::Dataset::SetVariableNames)
        .def("GetValues", [](Operon::Dataset const& self, std::string const& name) { return MakeView(self.GetValues(name)); })
        .def("GetValues", [](Operon::Dataset const& self, Operon::Hash hash) { return MakeView(self.GetValues(hash)); })
        .def("GetValues", [](Operon::Dataset const& self, int index) { return MakeView(self.GetValues(index)); })
        .def("GetVariable", py::overload_cast<const std::string&>(&Operon::Dataset::GetVariable, py::const_))
        .def("GetVariable", py::overload_cast<Operon::Hash>(&Operon::Dataset::GetVariable, py::const_))
        .def_property_readonly("Variables", [](Operon::Dataset const& self) {
            auto vars = self.Variables();
            return std::vector<Operon::Variable>(vars.begin(), vars.end());
        })
        .def("Shuffle", &Operon::Dataset::Shuffle)
        .def("Normalize", &Operon::Dataset::Normalize)
        .def("Standardize", &Operon::Dataset::Standardize)
        ;

    // tree creator
    py::class_<Operon::CreatorBase>(m, "CreatorBase");

    py::class_<Operon::BalancedTreeCreator, Operon::CreatorBase>(m, "BalancedTreeCreator")
        .def(py::init([](Operon::PrimitiveSet const& grammar, std::vector<Operon::Variable> const& variables, double bias) {
            return Operon::BalancedTreeCreator(grammar, gsl::span<const Operon::Variable>(variables.data(), variables.size()), bias);
        }),
            py::arg("grammar"), py::arg("variables"), py::arg("bias"))
        .def("__call__", &Operon::BalancedTreeCreator::operator())
        .def_property("IrregularityBias", &Operon::BalancedTreeCreator::GetBias, &Operon::BalancedTreeCreator::SetBias);

    py::class_<Operon::ProbabilisticTreeCreator, Operon::CreatorBase>(m, "ProbabilisticTreeCreator")
        .def(py::init<const Operon::PrimitiveSet&, const std::vector<Operon::Variable>>())
        .def("__call__", &Operon::ProbabilisticTreeCreator::operator());

    py::class_<Operon::GrowTreeCreator, Operon::CreatorBase>(m, "GrowTreeCreator")
        .def(py::init<const Operon::PrimitiveSet&, const std::vector<Operon::Variable>>())
        .def("__call__", &Operon::GrowTreeCreator::operator());

    // crossover
    py::class_<Operon::CrossoverBase>(m, "CrossoverBase");

    py::class_<Operon::SubtreeCrossover, Operon::CrossoverBase>(m, "SubtreeCrossover")
        .def(py::init<double, size_t, size_t>())
        .def("__call__", &Operon::SubtreeCrossover::operator());

    // mutation
    py::class_<Operon::MutatorBase>(m, "MutatorBase");

    py::class_<Operon::OnePointMutation, Operon::MutatorBase>(m, "OnePointMutation")
        .def(py::init<>())
        .def("__call__", &Operon::OnePointMutation::operator());

    py::class_<Operon::ChangeVariableMutation, Operon::MutatorBase>(m, "ChangeVariableMutation")
        .def(py::init([](std::vector<Operon::Variable> const& variables) {
            return Operon::ChangeVariableMutation(gsl::span<const Operon::Variable>(variables.data(), variables.size()));
        }),
            py::arg("variables"))
        .def("__call__", &Operon::ChangeVariableMutation::operator());

    py::class_<Operon::ChangeFunctionMutation, Operon::MutatorBase>(m, "ChangeFunctionMutation")
        .def(py::init<Operon::PrimitiveSet>())
        .def("__call__", &Operon::ChangeFunctionMutation::operator());

    py::class_<Operon::ReplaceSubtreeMutation, Operon::MutatorBase>(m, "ReplaceSubtreeMutation")
        .def(py::init<Operon::CreatorBase&, size_t, size_t>())
        .def("__call__", &Operon::ReplaceSubtreeMutation::operator());

    py::class_<Operon::RemoveSubtreeMutation, Operon::MutatorBase>(m, "RemoveSubtreeMutation")
        .def(py::init<Operon::PrimitiveSet>())
        .def("__call__", &Operon::RemoveSubtreeMutation::operator());

    py::class_<Operon::InsertSubtreeMutation, Operon::MutatorBase>(m, "InsertSubtreeMutation")
        .def(py::init<Operon::CreatorBase&, size_t, size_t>())
        .def("__call__", &Operon::InsertSubtreeMutation::operator());

    py::class_<Operon::MultiMutation, Operon::MutatorBase>(m, "MultiMutation")
        .def(py::init<>())
        .def("__call__", &Operon::MultiMutation::operator())
        .def("Add", &Operon::MultiMutation::Add);

    // selection
    py::class_<Operon::SelectorBase>(m, "SelectorBase");

    py::class_<Operon::TournamentSelector, Operon::SelectorBase>(m, "TournamentSelector")
        .def(py::init([](size_t i){ 
                    return Operon::TournamentSelector([i](const auto& a, const auto& b) { return a[i] < b[i]; });
                    }), py::arg("objective_index"))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::TournamentSelector::operator())
        .def_property("TournamentSize", &Operon::TournamentSelector::GetTournamentSize, &Operon::TournamentSelector::SetTournamentSize);

    py::class_<Operon::RankTournamentSelector, Operon::SelectorBase>(m, "RankTournamentSelector")
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::RankTournamentSelector::operator())
        .def("Prepare", &Operon::RankTournamentSelector::Prepare)
        .def_property("TournamentSize", &Operon::RankTournamentSelector::GetTournamentSize, &Operon::RankTournamentSelector::SetTournamentSize);

    py::class_<Operon::ProportionalSelector, Operon::SelectorBase>(m, "ProportionalSelector")
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::ProportionalSelector::operator())
        .def("Prepare", py::overload_cast<const gsl::span<const Operon::Individual>>(&Operon::ProportionalSelector::Prepare, py::const_))
        .def("SetObjIndex", &Operon::ProportionalSelector::SetObjIndex);

    // reinserter
    py::class_<Operon::ReinserterBase>(m, "ReinserterBase");

    py::class_<Operon::ReplaceWorstReinserter<std::execution::parallel_unsequenced_policy>, Operon::ReinserterBase>(m, "ReplaceWorstReinserter")
        .def(py::init([](size_t i) { 
            return Operon::ReplaceWorstReinserter([i](const auto& a, const auto& b) { return a[i] < b[i]; });
                    }), py::arg("objective_index"))
        .def(py::init<Operon::ComparisonCallback>())
        .def("__call__", &Operon::ReplaceWorstReinserter<std::execution::parallel_unsequenced_policy>::operator());

    // offspring generator
    py::class_<Operon::OffspringGeneratorBase>(m, "OffspringGeneratorBase")
        .def_property_readonly("Terminate", &Operon::OffspringGeneratorBase::Terminate);
    
    py::class_<Operon::BasicOffspringGenerator, Operon::OffspringGeneratorBase>(m, "BasicOffspringGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&, 
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", &Operon::BasicOffspringGenerator::operator())
        .def("Prepare", [](Operon::BasicOffspringGenerator& self, std::vector<Operon::Individual> const& individuals) {
                gsl::span<const Operon::Individual> s(individuals.data(), individuals.size());
                self.Prepare(s);
            })
        .def("__call__", [](Operon::BasicOffspringGenerator& self, Operon::RandomGenerator& rng, double pc, double pm,size_t n) {
                std::vector<Operon::Individual> v;
                v.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    if (auto res = self(rng, pc, pm); res.has_value())
                        v.push_back(res.value());
                }
                return v;
            });

    // evaluator
    py::class_<Operon::EvaluatorBase>(m, "EvaluatorBase")
        .def_property("LocalOptimizationIterations", &Operon::EvaluatorBase::GetLocalOptimizationIterations, &Operon::EvaluatorBase::SetLocalOptimizationIterations)
        .def_property("Budget",&Operon::EvaluatorBase::GetBudget, &Operon::EvaluatorBase::SetBudget);

    py::class_<Operon::RSquaredEvaluator, Operon::EvaluatorBase>(m, "RSquaredEvaluator")
        .def(py::init<Operon::Problem&>())
        .def("__call__", &Operon::RSquaredEvaluator::operator());

    // random generators
    py::class_<Operon::Random::RomuTrio>(m, "RomuTrio")
        .def(py::init<uint64_t>())
        .def("__call__", &Operon::Random::RomuTrio::operator());

    // tree format
    py::class_<Operon::TreeFormatter>(m, "TreeFormatter")
        .def_static("Format", &Operon::TreeFormatter::Format);

    py::class_<Operon::InfixFormatter>(m, "InfixFormatter")
        .def_static("Format", &Operon::InfixFormatter::Format);

    // problem
    py::class_<Operon::Problem>(m, "Problem")
        .def(py::init([](Operon::Dataset const& ds, std::vector<Operon::Variable> const& variables, std::string const& target,
                        Operon::Range trainingRange, Operon::Range testRange) {
            gsl::span<const Operon::Variable> vars(variables.data(), variables.size());
            return Operon::Problem(ds).Inputs(variables).Target(target).TrainingRange(trainingRange).TestRange(testRange);
        }));

    // genetic algorithm
    py::class_<Operon::GeneticAlgorithmConfig>(m, "GeneticAlgorithmConfig")
        .def(py::init([](size_t gen, size_t evals, size_t iter, size_t popsize, size_t poolsize, double pc, double pm, size_t seed){
                    Operon::GeneticAlgorithmConfig config;
                    config.Generations = gen;
                    config.Evaluations = evals;
                    config.Iterations = iter;
                    config.PopulationSize = popsize;
                    config.PoolSize = poolsize;
                    config.CrossoverProbability = pc;
                    config.MutationProbability = pm;
                    config.Seed = seed;
                    return config;
        }), py::arg("generations")
          , py::arg("max_evaluations")
          , py::arg("local_iterations")
          , py::arg("population_size")
          , py::arg("pool_size")
          , py::arg("p_crossover")
          , py::arg("p_mutation")
          , py::arg("seed"))
        .def_readwrite("Generations", &Operon::GeneticAlgorithmConfig::Generations)
        .def_readwrite("Evaluations", &Operon::GeneticAlgorithmConfig::Evaluations)
        .def_readwrite("Iterations", &Operon::GeneticAlgorithmConfig::Iterations)
        .def_readwrite("PopulationSize", &Operon::GeneticAlgorithmConfig::PopulationSize)
        .def_readwrite("PoolSize", &Operon::GeneticAlgorithmConfig::PoolSize)
        .def_readwrite("CrossoverProbability", &Operon::GeneticAlgorithmConfig::CrossoverProbability)
        .def_readwrite("MutationProbability", &Operon::GeneticAlgorithmConfig::MutationProbability)
        .def_readwrite("Seed", &Operon::GeneticAlgorithmConfig::Seed);

    using UniformInitializer = Operon::Initializer<std::uniform_int_distribution<size_t>>;
    py::class_<UniformInitializer>(m, "UniformInitializer")
        .def(py::init([](Operon::CreatorBase const& creator, size_t minLength, size_t maxLength) { 
                    std::uniform_int_distribution<size_t> dist(minLength, maxLength);
                    return UniformInitializer(creator, dist);
                    }))
        .def("__call__", &UniformInitializer::operator())
        .def_property("MinDepth"
                , py::overload_cast<>(&UniformInitializer::MinDepth, py::const_)
                , py::overload_cast<size_t>(&UniformInitializer::MinDepth))
        .def_property("MaxDepth"
                , py::overload_cast<>(&UniformInitializer::MaxDepth, py::const_)
                , py::overload_cast<size_t>(&UniformInitializer::MaxDepth))
        ;

    using GeneticProgrammingAlgorithm = Operon::GeneticProgrammingAlgorithm<UniformInitializer>;
    py::class_<GeneticProgrammingAlgorithm>(m, "GeneticProgrammingAlgorithm")
        .def(py::init<Operon::Problem const&, Operon::GeneticAlgorithmConfig const&, UniformInitializer&,
                Operon::OffspringGeneratorBase const&, Operon::ReinserterBase const&>())
        .def("Run", &GeneticProgrammingAlgorithm::Run, py::call_guard<py::gil_scoped_release>(), py::arg("rng"), py::arg("callback"), py::arg("threads") = 0)
        .def("BestModel", [](GeneticProgrammingAlgorithm const& self, Operon::Comparison const& comparison) {
                    auto min_elem = std::min_element(self.Parents().begin(), self.Parents().end(), [&](auto const& a, auto const& b) { return comparison(a, b);});
                    return *min_elem;
                });
}
