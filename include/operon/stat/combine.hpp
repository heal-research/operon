#ifndef OPERON_STAT_COMBINE_HPP
#define OPERON_STAT_COMBINE_HPP

#include <tuple>
#include <Eigen/Core>

namespace Operon {
    template<typename R>
    double Combine(double n, R _sumV, R _sumVV)
    {
        auto square = [](double v) { return v * v; };
        double s0 = _sumV(0), q0 = _sumVV(0);
        double s1 = _sumV(1), q1 = _sumVV(1);
        double s2 = _sumV(2), q2 = _sumVV(2);
        double s3 = _sumV(3), q3 = _sumVV(3);
        double s01 = s0 + s1;
        double s23 = s2 + s3;
        double q01 = q0 + q1 + square(s1 - s0) / (2 * n);
        double q23 = q2 + q3 + square(s3 - s2) / (2 * n);
        return q01 + q23 + square(s23 - s01) / (4 * n);
    }

    // returns a combined variance value
    template<typename R>
    double Combine(R _sumWe, R _sumV, R _sumVV)
    {
        auto square = [](double v) { return v * v; };
        double n0 = _sumWe(0), s0 = _sumV(0), q0 = _sumVV(0);
        double n1 = _sumWe(1), s1 = _sumV(1), q1 = _sumVV(1);
        double n2 = _sumWe(2), s2 = _sumV(2), q2 = _sumVV(2);
        double n3 = _sumWe(3), s3 = _sumV(3), q3 = _sumVV(3);

        double n01 = n0 + n1, s01 = s0 + s1;
        double n23 = n2 + n3, s23 = s2 + s3;

        double f01 = 1. / (n0 * n01 * n1);
        double f23 = 1. / (n2 * n23 * n3);
        double f = 1. / (n01 * (n01 + n23) * n23);

        double q01 = q0 + q1 + f01 * square(n0 * s1 - n1 * s0);
        double q23 = q2 + q3 + f23 * square(n2 * s3 - n3 * s2);
        return q01 + q23 + f * square(n01 * s23 - n23 * s01);
    }

    // combines four sets of statistics (corresponding to four partitions of (x,y)-data) into a single set
    // statistics include sums, variances, covariance
    // simplified version assuming all data partitions have the same length/weight n
    template<typename R>
    std::tuple<double, double, double> Combine(double n, R _sumX, R _sumY, R _sumXX, R _sumYY, R _sumXY)
    {
        // see Schubert et al. - Numerically Stable Parallel Computation of (Co-)Variance, p. 4, eq. 22-26
        // https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
        // merge covariance from individual data partitions A,B

        double sx0 = _sumX(0), sy0 = _sumY(0), sxx0 = _sumXX(0), sxy0 = _sumXY(0), syy0 = _sumYY(0);
        double sx1 = _sumX(1), sy1 = _sumY(1), sxx1 = _sumXX(1), sxy1 = _sumXY(1), syy1 = _sumYY(1);
        double sx2 = _sumX(2), sy2 = _sumY(2), sxx2 = _sumXX(2), sxy2 = _sumXY(2), syy2 = _sumYY(2);
        double sx3 = _sumX(3), sy3 = _sumY(3), sxx3 = _sumXX(3), sxy3 = _sumXY(3), syy3 = _sumYY(3);

        double sx01 = sx0 + sx1, sy01 = sy0 + sy1;
        double sx23 = sx2 + sx3, sy23 = sy2 + sy3;

        double n2 = n + n;

        auto square = [](double v) { return v * v; };
        double f = 0.25 / n;

        // X
        double qx01 = sxx0 + sxx1 + square(sx1 - sx0) / n2;
        double qx23 = sxx2 + sxx3 + square(sx3 - sx2) / n2;
        double sxx = qx01 + qx23 + f * square(sx23 - sx01);

        // Y
        double qy01 = syy0 + syy1 + square(sy1 - sy0) / n2;
        double qy23 = syy2 + syy3 + square(sy3 - sy2) / n2;
        double syy = qy01 + qy23 + f * square(sy23 - sy01);

        // XY
        double q01 = sxy0 + sxy1 + (sx0 - sx1) * (sy0 - sy1) / n2;
        double q23 = sxy2 + sxy3 + (sx2 - sx3) * (sy2 - sy3) / n2;
        double sxy = q01 + q23 + (sx01 - sx23) * (sy01 - sy23) / (2 * n2);

        return { sxx, syy, sxy };
    }

    // combines four sets of statistics (corresponding to four partitions of (x,y)-data) into a single set
    // statistics include sums, variances, covariance
    template<typename R>
    std::tuple<double, double, double> Combine(R _sumWe, R _sumX, R _sumY, R _sumXX, R _sumYY, R _sumXY)
    {
        static_assert(R::RowsAtCompileTime == 4);
        // see Schubert et al. - Numerically Stable Parallel Computation of (Co-)Variance, p. 4, eq. 22-26
        // https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
        // merge covariance from individual data partitions A,B
        double n0 = _sumWe(0), sx0 = _sumX(0), sy0 = _sumY(0), sxx0 = _sumXX(0), sxy0 = _sumXY(0), syy0 = _sumYY(0);
        double n1 = _sumWe(1), sx1 = _sumX(1), sy1 = _sumY(1), sxx1 = _sumXX(1), sxy1 = _sumXY(1), syy1 = _sumYY(1);
        double n2 = _sumWe(2), sx2 = _sumX(2), sy2 = _sumY(2), sxx2 = _sumXX(2), sxy2 = _sumXY(2), syy2 = _sumYY(2);
        double n3 = _sumWe(3), sx3 = _sumX(3), sy3 = _sumY(3), sxx3 = _sumXX(3), sxy3 = _sumXY(3), syy3 = _sumYY(3);

        double n01 = n0 + n1, sx01 = sx0 + sx1, sy01 = sy0 + sy1;
        double n23 = n2 + n3, sx23 = sx2 + sx3, sy23 = sy2 + sy3;

        double f01 = 1. / (n0 * n01 * n1);
        double f23 = 1. / (n2 * n23 * n3);
        double f = 1. / (n01 * (n01 + n23) * n23);

        auto square = [](auto a) { return a * a; };
        // X
        double qx01 = sxx0 + sxx1 + f01 * square(n0 * sx1 - n1 * sx0);
        double qx23 = sxx2 + sxx3 + f23 * square(n2 * sx3 - n3 * sx2);
        double sxx = qx01 + qx23 + f * square(n01 * sx23 - n23 * sx01);

        // Y
        double qy01 = syy0 + syy1 + f01 * square(n0 * sy1 - n1 * sy0);
        double qy23 = syy2 + syy3 + f23 * square(n2 * sy3 - n3 * sy2);
        double syy = qy01 + qy23 + f * square(n01 * sy23 - n23 * sy01);

        // XY
        double q01 = sxy0 + sxy1 + f01 * (n1 * sx0 - n0 * sx1) * (n1 * sy0 - n0 * sy1);
        double q23 = sxy2 + sxy3 + f23 * (n3 * sx2 - n2 * sx3) * (n3 * sy2 - n2 * sy3);
        double sxy = q01 + q23 + f * (n23 * sx01 - n01 * sx23) * (n23 * sy01 - n01 * sy23);

        return { sxx, syy, sxy };
    }
}

#endif

