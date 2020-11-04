#include "operon.hpp"

void init_pset(py::module_ &m)
{
    // primitive set 
    py::class_<Operon::PrimitiveSet>(m, "PrimitiveSet")
        .def(py::init<>())
        .def_property_readonly_static("Arithmetic", [](py::object /* self */) { return Operon::PrimitiveSet::Arithmetic; })
        .def_property_readonly_static("TypeCoherent", [](py::object /* self */) { return Operon::PrimitiveSet::TypeCoherent; })
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

}
