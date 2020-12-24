/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
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

#ifndef OPERON_NNLS_HPP
#define OPERON_NNLS_HPP

#include "core/types.hpp"
#include "tiny_optimizer.hpp"

#if defined(HAVE_CERES)
#include "ceres_optimizer.hpp"
#endif

namespace Operon {

OptimizerSummary Optimize(Tree& tree, Dataset const& dataset, const gsl::span<const Operon::Scalar> targetValues, Range const range, size_t iterations = 50, bool writeCoefficients = true, bool report = false) {
#if defined(CERES_TINY_SOLVER) || !defined(HAVE_CERES)
    Optimizer<DerivativeMethod::AUTODIFF, OptimizerType::TINY> optimizer;
#else
    Optimizer<DerivativeMethod::AUTODIFF, OptimizerType::CERES> optimizer;
#endif
    return optimizer.Optimize(tree, dataset, targetValues, range, iterations, writeCoefficients, report);
}
}

#endif

