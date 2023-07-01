#ifndef ELKI_STATS_HPP
#define ELKI_STATS_HPP

#include <cassert>
#include <cmath>
#include <iterator>

/*
 * This code is a direct port of the statistical methods from ELKI
 * https://github.com/elki-project/elki/blob/master/elki-core-math/src/main/java/elki/math/MeanVariance.java
 *
 * Copyright ELKI Development Team
 * Code distributed under the terms of the GNU Affero General Public License version 3 or later
 *
 * We use it here for unit testing Operon/Vstat statistical methods
 */

namespace Elki {
    struct MeanVarianceResult {
        double Sum;
        double Mean;
        double Variance;
    };

    struct MeanVariance {
        double n{0};
        double sum{0};
        double m2{0};

        void put(double val) {
            if(n <= 0) {
                n = 1;
                sum = val;
                m2 = 0;
                return;
            }
            double tmp = n * val - sum;
            double oldn = n; // tmp copy
            n += 1.0;
            sum += val;
            m2 += tmp * tmp / (n * oldn);
        }

        void put(double val, double weight) {
            if(weight == 0.) {
                return;
            }
            if(n <= 0) {
                n = weight;
                sum = val * weight;
                return;
            }
            val *= weight;
            double tmp = n * val - sum * weight;
            double oldn = n; // tmp copy
            n += weight;
            sum += val;
            m2 += tmp * tmp / (weight * n * oldn);
        }

        auto put(auto const& x) {
            for (auto v : x) { put(v); }
        }

        auto put(auto const& x, auto const& w) {
            assert(std::ssize(x) == std::ssize(w));
            for (auto i = 0; i < std::ssize(x); ++i) {
                put(x[i], w[i]);
            }
        }

        double Sum() const { return sum; }

        double PopulationMean() const { return sum / n; }
        double SampleMean() const { return sum / (n-1); }

        double PopulationVariance() const { return m2 / n; }
        double SampleVariance() const { return m2 / (n-1); }

        static inline auto PopulationStats(auto const& x) {
            MeanVariance mv;
            mv.put(x);
            return MeanVarianceResult{ mv.Sum(), mv.PopulationMean(), mv.PopulationVariance() };
        }

        static inline auto PopulationStats(auto const& x, auto const& w) {
            MeanVariance mv;
            mv.put(x, w);
            return MeanVarianceResult{ mv.Sum(), mv.PopulationMean(), mv.PopulationVariance() };
        }

        static inline auto SampleStats(auto const& x) {
            MeanVariance mv;
            mv.put(x);
            return MeanVarianceResult{ mv.Sum(), mv.SampleMean(), mv.SampleVariance() };
        }

        static inline auto SampleStats(auto const& x, auto const& w) {
            MeanVariance mv;
            mv.put(x, w);
            return MeanVarianceResult{ mv.Sum(), mv.SampleMean(), mv.SampleVariance() };
        }

        static inline auto SSR(auto const& x, auto const& y) {
            assert(std::ssize(x) == std::ssize(y));
            auto const n{ std::ssize(x) };
            MeanVariance mv;
            for (auto i = 0; i < n; ++i) {
                auto const e = x[i] - y[i];
                mv.put(e * e);
            }
            return mv.Sum();
        }

        static inline auto SSR(auto const& x, auto const& y, auto const& z) {
            assert(std::ssize(x) == std::ssize(y));
            assert(std::ssize(x) == std::ssize(z));
            auto const n{ std::ssize(x) };
            MeanVariance mv;
            for (auto i = 0; i < n; ++i) {
                auto const e = x[i] - y[i];
                mv.put(e * e, z[i]);
            }
            return mv.Sum();
        }

        static inline auto MSE(auto const& x, auto const& y) {
            assert(std::ssize(x) == std::ssize(y));
            auto const n{ std::ssize(x) };
            MeanVariance mv;
            for (auto i = 0; i < n; ++i) {
                auto const e = x[i] - y[i];
                mv.put(e * e);
            }
            return mv.PopulationMean();
        }

        static inline auto MSE(auto const& x, auto const& y, auto const& z) {
            assert(std::ssize(x) == std::ssize(y));
            assert(std::ssize(x) == std::ssize(z));
            auto const n{ std::ssize(x) };
            MeanVariance mv;
            for (auto i = 0; i < n; ++i) {
                auto const e = x[i] - y[i];
                mv.put(e * e, z[i]);
            }
            return mv.PopulationMean();
        }

        static inline auto NMSE(auto const& x, auto const& y) {
            assert(std::ssize(x) == std::ssize(y));
            auto const n{ std::ssize(x) };
            MeanVariance mv, mv1;
            for (auto i = 0; i < n; ++i) {
                auto const e = x[i] - y[i];
                mv.put(e * e);
                mv1.put(y[i]);
            }
            return mv.PopulationMean() / mv1.PopulationVariance();
        }

        static inline auto NMSE(auto const& x, auto const& y, auto const& z) {
            assert(std::ssize(x) == std::ssize(y));
            assert(std::ssize(x) == std::ssize(z));
            auto const n{ std::ssize(x) };
            MeanVariance mv, mv1;
            for (auto i = 0; i < n; ++i) {
                auto const e = x[i] - y[i];
                mv.put(e * e, z[i]);
                mv1.put(y[i], z[i]);
            }
            return mv.PopulationMean() / mv1.PopulationVariance();
        }

        static inline auto MAE(auto const& x, auto const& y) {
            assert(std::ssize(x) == std::ssize(y));
            auto const n{ std::ssize(x) };
            MeanVariance mv;
            for (auto i = 0; i < n; ++i) {
                mv.put(std::abs(x[i] - y[i]));
            }
            return mv.PopulationMean();
        }

        static inline auto MAE(auto const& x, auto const& y, auto const& z) {
            assert(std::ssize(x) == std::ssize(y));
            assert(std::ssize(x) == std::ssize(z));
            auto const n{ std::ssize(x) };
            MeanVariance mv;
            for (auto i = 0; i < n; ++i) {
                mv.put(std::abs(x[i] - y[i]), z[i]);
            }
            return mv.PopulationMean();
        }

        static inline auto Corr(auto const& x, auto const& y) {
            assert(std::ssize(x) == std::ssize(y));
            auto const n{ std::ssize(x) };
            auto sx = MeanVariance::PopulationStats(x); // mean-variance for x
            auto sy = MeanVariance::PopulationStats(y); // mean-variance for y

            MeanVariance xx, xy, yy;
            for (auto i = 0; i < n; ++i) {
                auto dx = x[i] - sx.Mean;
                auto dy = y[i] - sy.Mean;
                xx.put(dx * dx);
                yy.put(dy * dy);
                xy.put(dx * dy);
            }

            return xy.Sum() / std::sqrt(xx.Sum() * yy.Sum());
        }

        static inline auto Corr(auto const& x, auto const& y, auto const& z) {
            assert(std::ssize(x) == std::ssize(y));
            assert(std::ssize(x) == std::ssize(z));
            auto const n{ std::ssize(x) };
            auto sx = MeanVariance::PopulationStats(x, z);
            auto sy = MeanVariance::PopulationStats(y, z);

            MeanVariance xx, xy, yy;
            for (auto i = 0; i < n; ++i) {
                auto dx = x[i] - sx.Mean;
                auto dy = y[i] - sy.Mean;
                xx.put(z[i] * dx * dx);
                yy.put(z[i] * dy * dy);
                xy.put(z[i] * dx * dy);
            }
            return xy.Sum() / std::sqrt(xx.Sum() * yy.Sum());
        }

        static inline auto R2(auto const& x, auto const& y) {
            assert(std::ssize(x) == std::ssize(y));
            auto my = MeanVariance::PopulationStats(y).Mean;
            MeanVariance sx, sy;
            for (auto i = 0; i < std::ssize(x); ++i) {
                auto e1 = x[i] - y[i];
                auto e2 = y[i] - my;
                sx.put(e1 * e1);
                sy.put(e2 * e2);
            }
            return 1.0 - sx.Sum() / sy.Sum();
        }

        static inline auto R2(auto const& x, auto const& y, auto const& z) {
            assert(std::ssize(x) == std::ssize(y));
            auto my = MeanVariance::PopulationStats(y).Mean;
            MeanVariance sx, sy;
            for (auto i = 0; i < std::ssize(x); ++i) {
                auto e1 = x[i] - y[i];
                auto e2 = y[i] - my;
                sx.put(z[i] * e1 * e1);
                sy.put(z[i] * e2 * e2);
            }
            return 1.0 - sx.Sum() / sy.Sum();
        }
    };
}

#endif
