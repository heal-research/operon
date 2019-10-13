/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
 */

#ifndef METRICS_HPP
#define METRICS_HPP

#include "core/common.hpp"
#include "core/stats.hpp"

namespace Operon {

operon::scalar_t NormalizedMeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y);
operon::scalar_t MeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y);
operon::scalar_t RootMeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y);
operon::scalar_t RSquared(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y);
} // namespace
#endif
