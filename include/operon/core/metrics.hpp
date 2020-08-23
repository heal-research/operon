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

#ifndef METRICS_HPP
#define METRICS_HPP

#include "core/types.hpp"
#include "core/contracts.hpp"
#include "core/stats.hpp"

namespace Operon {

double NormalizedMeanSquaredError(gsl::span<const Operon::Scalar> x, gsl::span<const Operon::Scalar> y);
double MeanSquaredError(gsl::span<const Operon::Scalar> x, gsl::span<const Operon::Scalar> y);
double RootMeanSquaredError(gsl::span<const Operon::Scalar> x, gsl::span<const Operon::Scalar> y);
double RSquared(gsl::span<const Operon::Scalar> x, gsl::span<const Operon::Scalar> y);
} // namespace
#endif
