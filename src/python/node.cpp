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
        // node factory for convenience
        .def("Add", []() { return Operon::Node(Operon::NodeType::Add); })
        .def("Sub", []() { return Operon::Node(Operon::NodeType::Sub); })
        .def("Mul", []() { return Operon::Node(Operon::NodeType::Mul); })
        .def("Div", []() { return Operon::Node(Operon::NodeType::Div); })
        .def("Exp", []() { return Operon::Node(Operon::NodeType::Exp); })
        .def("Log", []() { return Operon::Node(Operon::NodeType::Log); })
        .def("Sin", []() { return Operon::Node(Operon::NodeType::Sin); })
        .def("Cos", []() { return Operon::Node(Operon::NodeType::Cos); })
        .def("Tan", []() { return Operon::Node(Operon::NodeType::Tan); })
        .def("Sqrt", []() { return Operon::Node(Operon::NodeType::Sqrt); })
        .def("Cbrt", []() { return Operon::Node(Operon::NodeType::Cbrt); })
        .def("Square", []() { return Operon::Node(Operon::NodeType::Square); })
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
