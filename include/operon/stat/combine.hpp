#ifndef OPERON_STAT_COMBINE_HPP
#define OPERON_STAT_COMBINE_HPP

#include <tuple>
#include <Eigen/Core>

namespace Operon {
    using R = Eigen::Ref<Eigen::Array4d>;

    // combine variance
    double Combine(double, R, R);
    double Combine(R, R, R);

    // combine covariance + variance
    std::tuple<double, double, double> Combine(double, R, R, R, R, R);
    std::tuple<double, double, double> Combine(R, R, R, R, R, R);
}

#endif

