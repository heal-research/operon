// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "core/node.hpp"
#include "operon.hpp"

namespace py = pybind11;

void init_node(py::module_ &m)
{
    // node type
    py::enum_<Operon::NodeType>(m, "NodeType")
        .value("Add", Operon::NodeType::Add)
        .value("Mul", Operon::NodeType::Mul)
        .value("Sub", Operon::NodeType::Sub)
        .value("Div", Operon::NodeType::Div)
        .value("Aq", Operon::NodeType::Aq)
        .value("Fmax", Operon::NodeType::Fmax)
        .value("Fmin", Operon::NodeType::Fmin)
        .value("Pow", Operon::NodeType::Pow)
        .value("Abs", Operon::NodeType::Abs)
        .value("Acos", Operon::NodeType::Acos)
        .value("Asin", Operon::NodeType::Asin)
        .value("Atan", Operon::NodeType::Atan)
        .value("Cbrt", Operon::NodeType::Cbrt)
        .value("Ceil", Operon::NodeType::Ceil)
        .value("Cos", Operon::NodeType::Cos)
        .value("Cosh", Operon::NodeType::Cosh)
        .value("Erf", Operon::NodeType::Erf)
        .value("Erfc", Operon::NodeType::Erfc)
        .value("Exp", Operon::NodeType::Exp)
        .value("Log", Operon::NodeType::Log)
        .value("Log1p", Operon::NodeType::Log1p)
        .value("Sin", Operon::NodeType::Sin)
        .value("Sinh", Operon::NodeType::Sinh)
        .value("Sqrt", Operon::NodeType::Sqrt)
        .value("Square", Operon::NodeType::Square)
        .value("Tan", Operon::NodeType::Tan)
        .value("Tanh", Operon::NodeType::Tanh)
        .value("Dyn", Operon::NodeType::Dynamic)
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
        // node factory for convenience
        .def("Add", []() { return Operon::Node(Operon::NodeType::Add); })
        .def("Sub", []() { return Operon::Node(Operon::NodeType::Sub); })
        .def("Mul", []() { return Operon::Node(Operon::NodeType::Mul); })
        .def("Div", []() { return Operon::Node(Operon::NodeType::Div); })
        .def("Aq", []() { return Operon::Node(Operon::NodeType::Aq); })
        .def("Fmax", []() { return Operon::Node(Operon::NodeType::Fmax); })
        .def("Fmin", []() { return Operon::Node(Operon::NodeType::Fmin); })
        .def("Abs", []() { return Operon::Node(Operon::NodeType::Abs); })
        .def("Acos", []() { return Operon::Node(Operon::NodeType::Acos); })
        .def("Asin", []() { return Operon::Node(Operon::NodeType::Asin); })
        .def("Atan", []() { return Operon::Node(Operon::NodeType::Atan); })
        .def("Cbrt", []() { return Operon::Node(Operon::NodeType::Cbrt); })
        .def("Ceil", []() { return Operon::Node(Operon::NodeType::Ceil); })
        .def("Cos", []() { return Operon::Node(Operon::NodeType::Cos); })
        .def("Cosh", []() { return Operon::Node(Operon::NodeType::Cosh); })
        .def("Erf", []() { return Operon::Node(Operon::NodeType::Erf); })
        .def("Erfc", []() { return Operon::Node(Operon::NodeType::Erfc); })
        .def("Exp", []() { return Operon::Node(Operon::NodeType::Exp); })
        .def("Log", []() { return Operon::Node(Operon::NodeType::Log); })
        .def("Log1p", []() { return Operon::Node(Operon::NodeType::Log1p); })
        .def("Pow", []() { return Operon::Node(Operon::NodeType::Pow); })
        .def("Sin", []() { return Operon::Node(Operon::NodeType::Sin); })
        .def("Sinh", []() { return Operon::Node(Operon::NodeType::Sinh); })
        .def("Sqrt", []() { return Operon::Node(Operon::NodeType::Sqrt); })
        .def("Square", []() { return Operon::Node(Operon::NodeType::Square); })
        .def("Tan", []() { return Operon::Node(Operon::NodeType::Tan); })
        .def("Tanh", []() { return Operon::Node(Operon::NodeType::Tanh); })
        .def("Dyn", []() { return Operon::Node(Operon::NodeType::Dynamic); })
        .def("Constant", [](double v) {
                Operon::Node constant(Operon::NodeType::Constant);
                constant.Value = static_cast<Operon::Scalar>(v);
                return constant;
                })
        .def("Variable", [](double w) {
                Operon::Node variable(Operon::NodeType::Variable);
                variable.Value = static_cast<Operon::Scalar>(w);
                return variable;
                })
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

}
