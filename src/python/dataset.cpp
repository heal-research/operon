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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <gsl/span>

#include "core/dataset.hpp"

#include "operon.hpp"

namespace py = pybind11;

template<typename T>
Operon::Dataset MakeDataset(py::array_t<T> array)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");

    // sanity check
    if (array.ndim() != 2) {
        throw std::runtime_error("The input array must have exactly two dimensions.\n");
    }

    // check if the array satisfies our data storage requirements (contiguous, column-major order)
    if (std::is_same_v<T, Operon::Scalar> && (array.flags() & py::array::f_style)) {
        auto ref = array.template cast<Eigen::Ref<Operon::Dataset::Matrix const>>();
        return Operon::Dataset(ref);
    } else {
        fmt::print(stderr, "operon warning: array does not satisfy contiguity or storage-order requirements. data will be copied.\n");
        auto m = array.template cast<Operon::Dataset::Matrix>();
        return Operon::Dataset(std::move(m));
    }
}

template<typename T>
Operon::Dataset MakeDataset(std::vector<std::vector<T>> const& values)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");

    auto rows = values[0].size();
    auto cols = values.size();

    Operon::Dataset::Matrix m(rows, cols);

    for (size_t i = 0; i < values.size(); ++i) {
        m.col(i) = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor> const>(values[i].data(), rows).template cast<Operon::Scalar>();
    }
    return Operon::Dataset(std::move(m));
}

Operon::Dataset MakeDataset(py::buffer buf)
{
    auto info = buf.request();

    if (info.ndim != 2) {
        throw std::runtime_error("The buffer must have two dimensions.\n");
    }

    if (info.format == py::format_descriptor<Operon::Scalar>::format()) {
        auto ref = buf.template cast<Eigen::Ref<Operon::Dataset::Matrix const>>();
        return Operon::Dataset(ref);
    } else {
        fmt::print(stderr, "operon warning: array does not satisfy contiguity or storage-order requirements. data will be copied.\n");
        auto m = buf.template cast<Operon::Dataset::Matrix>();
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


void init_dataset(py::module_ &m)
{
    // dataset
    py::class_<Operon::Dataset>(m, "Dataset")
        .def(py::init<std::string const&, bool>(), py::arg("filename"), py::arg("has_header"))
        .def(py::init<Operon::Dataset const&>())
        .def(py::init<std::vector<Operon::Variable> const&, const std::vector<std::vector<Operon::Scalar>>&>())
        .def(py::init([](py::array_t<float> array){ return MakeDataset(array); }), py::arg("data").noconvert())
        .def(py::init([](py::array_t<double> array){ return MakeDataset(array); }), py::arg("data").noconvert())
        .def(py::init([](std::vector<std::vector<float>> const& values) { return MakeDataset(values); }), py::arg("data").noconvert())
        .def(py::init([](std::vector<std::vector<double>> const& values) { return MakeDataset(values); }), py::arg("data").noconvert())
        .def(py::init([](py::buffer buf) { return MakeDataset(buf); }), py::arg("data").noconvert())
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
}
