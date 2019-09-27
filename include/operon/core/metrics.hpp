#ifndef METRICS_HPP
#define METRICS_HPP

#include "core/common.hpp"
#include "core/stats.hpp"

namespace Operon {

double NormalizedMeanSquaredError(gsl::span<const double> x, gsl::span<const double> y);
double MeanSquaredError(gsl::span<const double> x, gsl::span<const double> y);
double RootMeanSquaredError(gsl::span<const double> x, gsl::span<const double> y);
double RSquared(gsl::span<const double> x, gsl::span<const double> y);
} // namespace
#endif
