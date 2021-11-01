// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon.hpp"

void init_pset(py::module_ &m)
{
    // primitive set 
    py::class_<Operon::PrimitiveSet>(m, "PrimitiveSet")
        .def(py::init<>())
        .def_property_readonly_static("Arithmetic", [](py::object /* self */) { return Operon::PrimitiveSet::Arithmetic; })
        .def_property_readonly_static("TypeCoherent", [](py::object /* self */) { return Operon::PrimitiveSet::TypeCoherent; })
        .def("IsEnabled", py::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::IsEnabled, py::const_))
        .def("IsEnabled", py::overload_cast<Operon::Node>(&Operon::PrimitiveSet::IsEnabled, py::const_))
        .def("Enable", py::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::Enable))
        .def("Enable", py::overload_cast<Operon::Node>(&Operon::PrimitiveSet::Enable))
        .def("Disable", py::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::Disable))
        .def("Disable", py::overload_cast<Operon::Node>(&Operon::PrimitiveSet::Disable))
        .def("GetConfig", &Operon::PrimitiveSet::GetConfig)
        .def("SetConfig", &Operon::PrimitiveSet::SetConfig)
        .def("GetFrequency", py::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::GetFrequency, py::const_))
        .def("GetFrequency", py::overload_cast<Operon::Node>(&Operon::PrimitiveSet::GetFrequency, py::const_))
        .def("GetMinimumArity", py::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::GetMinimumArity, py::const_))
        .def("GetMinimumArity", py::overload_cast<Operon::Node>(&Operon::PrimitiveSet::GetMinimumArity, py::const_))
        .def("GetMaximumArity", py::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::GetMaximumArity, py::const_))
        .def("GetMaximumArity", py::overload_cast<Operon::Node>(&Operon::PrimitiveSet::GetMaximumArity, py::const_))
        .def("GetMinMaxArity", py::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::GetMinMaxArity, py::const_))
        .def("GetMinMaxArity", py::overload_cast<Operon::Node>(&Operon::PrimitiveSet::GetMinMaxArity, py::const_))
        .def("SetFrequency", py::overload_cast<Operon::Hash, size_t>(&Operon::PrimitiveSet::SetFrequency))
        .def("SetFrequency", py::overload_cast<Operon::Node, size_t>(&Operon::PrimitiveSet::SetFrequency))
        .def("SetMinimumArity", py::overload_cast<Operon::Hash, size_t>(&Operon::PrimitiveSet::SetMinimumArity))
        .def("SetMinimumArity", py::overload_cast<Operon::Node, size_t>(&Operon::PrimitiveSet::SetMinimumArity))
        .def("SetMaximumArity", py::overload_cast<Operon::Hash, size_t>(&Operon::PrimitiveSet::SetMaximumArity))
        .def("SetMaximumArity", py::overload_cast<Operon::Node, size_t>(&Operon::PrimitiveSet::SetMaximumArity))
        .def("SetMinMaxArity", py::overload_cast<Operon::Hash, size_t, size_t>(&Operon::PrimitiveSet::SetMinMaxArity))
        .def("SetMinMaxArity", py::overload_cast<Operon::Node, size_t, size_t>(&Operon::PrimitiveSet::SetMinMaxArity))
        .def("FunctionArityLimits", &Operon::PrimitiveSet::FunctionArityLimits)
        .def("SampleRandomSymbol", &Operon::PrimitiveSet::SampleRandomSymbol);

}
