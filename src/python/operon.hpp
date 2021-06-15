// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

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
#include "core/constants.hpp"
#include "core/dataset.hpp"
#include "core/format.hpp"
#include "core/individual.hpp"
#include "core/node.hpp"
#include "core/pset.hpp"
#include "core/metrics.hpp"
#include "core/operator.hpp"

#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/generator.hpp"
#include "operators/initializer.hpp"
#include "operators/mutation.hpp"
#include "operators/selection.hpp"
#include "operators/reinserter/replaceworst.hpp"

namespace py = pybind11;

// enable pass-by-reference semantics for this vector type
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Variable>);
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Individual>);

using UniformInitializer          = Operon::Initializer<std::uniform_int_distribution<size_t>>;
using GeneticProgrammingAlgorithm = Operon::GeneticProgrammingAlgorithm<UniformInitializer>;

template<typename T>
py::array_t<T const> MakeView(Operon::Span<T const> view)
{
    auto sz = static_cast<pybind11::ssize_t>(view.size());
    py::array_t<T const> arr(sz, view.data(), py::capsule(view.data()));
    ENSURE(arr.owndata() == false);
    ENSURE(arr.data() == view.data());
    return arr;
}

template<typename T>
Operon::Span<T> MakeSpan(py::array_t<T> arr)
{
    py::buffer_info info = arr.request();
    using size_type = Operon::Span<const Operon::Scalar>::size_type;
    return Operon::Span<T>(static_cast<T*>(info.ptr), static_cast<size_type>(info.size));
}

template<typename T>
Operon::Span<T const> MakeConstSpan(py::array_t<T> arr)
{
    py::buffer_info info = arr.request();
    using size_type = Operon::Span<const Operon::Scalar>::size_type;
    return Operon::Span<T const>(static_cast<T const*>(info.ptr), static_cast<size_type>(info.size));
}

void init_algorithm(py::module_&);
void init_creator(py::module_&);
void init_crossover(py::module_&);
void init_dataset(py::module_&);
void init_eval(py::module_&);
void init_generator(py::module_&);
void init_mutation(py::module_&);
void init_node(py::module_&);
void init_problem(py::module_&);
void init_pset(py::module_&);
void init_reinserter(py::module_&m);
void init_selection(py::module_&m);
void init_tree(py::module_&);
