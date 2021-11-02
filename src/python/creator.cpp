// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon.hpp"

void init_creator(py::module_ &m)
{
    // tree creator
    py::class_<Operon::CreatorBase>(m, "CreatorBase");

    py::class_<Operon::BalancedTreeCreator, Operon::CreatorBase>(m, "BalancedTreeCreator")
        .def(py::init([](Operon::PrimitiveSet const& grammar, std::vector<Operon::Variable> const& variables, double bias) {
            return Operon::BalancedTreeCreator(grammar, Operon::Span<const Operon::Variable>(variables.data(), variables.size()), bias);
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

    py::class_<Operon::UniformInitializer>(m, "UniformInitializer")
        .def(py::init([](Operon::CreatorBase const& creator, size_t minLength, size_t maxLength) { 
                    std::uniform_int_distribution<size_t> dist(minLength, maxLength);
                    return Operon::UniformInitializer(creator, dist);
                    }))
        .def("__call__", &Operon::UniformInitializer::operator())
        .def_property("MinDepth"
                , py::overload_cast<>(&Operon::UniformInitializer::MinDepth, py::const_)
                , py::overload_cast<size_t>(&Operon::UniformInitializer::MinDepth))
        .def_property("MaxDepth"
                , py::overload_cast<>(&Operon::UniformInitializer::MaxDepth, py::const_)
                , py::overload_cast<size_t>(&Operon::UniformInitializer::MaxDepth));

}
