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
#include "core/individual.hpp"
#include "core/pset.hpp"
#include "core/metrics.hpp"
#include "core/operator.hpp"

#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/evaluator.hpp"
#include "operators/generator.hpp"
#include "operators/initializer.hpp"
#include "operators/mutation.hpp"
#include "operators/selection.hpp"
#include "operators/reinserter/replaceworst.hpp"

#include "stat/pearson.hpp"

namespace py = pybind11;

// enable pass-by-reference semantics for this vector type
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Variable>)
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Individual>)

using UniformInitializer          = Operon::Initializer<std::uniform_int_distribution<size_t>>;
using GeneticProgrammingAlgorithm = Operon::GeneticProgrammingAlgorithm<UniformInitializer>;

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
