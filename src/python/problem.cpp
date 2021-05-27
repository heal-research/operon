// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon.hpp"
#include "core/range.hpp"

void init_problem(py::module_ &m)
{
    // problem
    py::class_<Operon::Problem>(m, "Problem")
        .def(py::init([](Operon::Dataset const& ds, std::vector<Operon::Variable> const& variables, std::string const& target,
                        Operon::Range trainingRange, Operon::Range testRange) {
            Operon::Span<const Operon::Variable> vars(variables.data(), variables.size());
            return Operon::Problem(ds).Inputs(variables).Target(target).TrainingRange(trainingRange).TestRange(testRange);
        }))
        .def_property_readonly("PrimitiveSet", [](Operon::Problem& self) { return self.GetPrimitiveSet(); });
}
