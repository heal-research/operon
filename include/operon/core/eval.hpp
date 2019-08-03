#ifndef EVALUATE_HPP
#define EVALUATE_HPP

#include <ceres/ceres.h>
#include "tree.hpp"
#include "dataset.hpp"
#include "gsl/gsl"

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#define FOR(i) for(gsl::index i = 0; i < BATCHSIZE; ++i)

namespace Operon {
    constexpr gsl::index BATCHSIZE = 64;
    constexpr int JET_STRIDE       = 4;
    using     Dual                 = ceres::Jet<double, JET_STRIDE>;

    template<typename T> inline std::pair<T, T> MinMax(gsl::span<T> values) noexcept
    {
        // get first finite (not NaN, not infinity) value
        auto min = T(std::numeric_limits<double>::max());
        auto max = T(std::numeric_limits<double>::min());
        for (auto& v : values)
        {
            if (!ceres::IsFinite(v)) continue;
            if (min > v) min = v;
            if (max < v) max = v;
        }
        return { min, max };
    }

    template<typename T> inline void LimitToRange(gsl::span<T> values, T min, T max) noexcept
    {
        for (auto& v : values)
        {
            if (!ceres::IsFinite(v)) { v = (min + max) / 2.0; }
            if (v < min) { v = min; }
            if (v > max) { v = max; }
        }
    }

    template<typename T>
    std::vector<T> Evaluate(const Tree& tree, const Dataset& dataset, const Range range, T const* const parameters = nullptr)
    {
        std::vector<T> result(range.Size());
        Evaluate(tree, dataset, range, parameters, gsl::span<T>(result));
        return result;
    }

    template<typename T>
    void Evaluate(const Tree& tree, const Dataset& dataset, const Range range, T const* const parameters, gsl::span<T> result)
    {
        auto& nodes = tree.Nodes();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(nodes.size(), BATCHSIZE);

        // fill in buf for constants and non-terminal nodes
        gsl::index idx = 0;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            auto& s = nodes[i];

            if (s.IsConstant())
            {
                auto v = parameters == nullptr ? T(s.Value) : parameters[idx++];
                m.row(i).setConstant(v);
            }
        }

        gsl::index numRows = range.Size();
        for (gsl::index row = 0; row < numRows; row += BATCHSIZE)
        {
            auto remainingRows = std::min(BATCHSIZE, numRows - row);
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                auto& s = nodes[i];
                auto r = m.row(i).array();

                switch (s.Type)
                {
                    case NodeType::Variable:
                        {
                            auto values = dataset.GetValues(s.CalculatedHashValue).subspan(range.Start + row, remainingRows);
                            std::transform(values.begin(), values.end(), r.data(), [&](double v) { return T(v * s.Value); });
                            break;
                        }
                    case NodeType::Add:
                        {
                            auto c = i - 1;             // first child index
                            r = m.row(c).array();
                            for (gsl::index k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                            {
                                r += m.row(j).array();
                            }
                            break;
                        }
                    case NodeType::Sub:
                        {
                            auto c = i - 1;             // first child index
                            r = m.row(c).array();
                            if (s.Arity == 1)
                            {
                                r = -r;
                            }
                            else 
                            {
                                for (gsl::index k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                                {
                                    r -= m.row(j).array();
                                }
                            }
                            break;
                        }
                    case NodeType::Mul:
                        {
                            auto c = i - 1;             // first child index
                            r = m.row(c).array();
                            for (gsl::index k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                            {
                                r  *= m.row(j).array();
                            }
                            break;
                        }
                    case NodeType::Div:
                        {
                            auto c = i - 1;             // first child index
                            if (s.Arity == 1)
                            {
                                r = r.inverse();
                            }
                            else
                            {
                                for (gsl::index k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                                {
                                    r /= m.row(j).array(); 
                                }
                            }
                            break;
                        }
                    case NodeType::Log:
                        {
                            r = m.row(i-1).array().log();
                            break;
                        }
                    case NodeType::Exp:
                        {
                            r = m.row(i-1).array().exp();
                            break;
                        }
                    case NodeType::Sin:
                        {
                            r = m.row(i-1).array().sin();
                            break;
                        }
                    case NodeType::Cos:
                        {
                            r = m.row(i-1).array().cos();
                            break;
                        }
                    case NodeType::Tan:
                        {
                            r = m.row(i-1).array().tan();
                            break;
                        }
                    case NodeType::Sqrt:
                        {
                            r = m.row(i-1).array().sqrt();
                            break;
                        }
                    case NodeType::Cbrt:
                        {
                            r = m.row(i-1).array().unaryExpr([](T v) { return ceres::cbrt(v); });
                            break;
                        }
                    case NodeType::Square:
                        {
                            r = m.row(i-1).array().square();
                            break;
                        }
                    default:
                        {
                            break;
                        }
                }
            }
            // the final result is found in the last section of the buffer corresponding to the root node
            std::copy_n(m.row(m.rows()-1).data(), remainingRows, result.begin() + row); 
        }
        // replace nan and inf values 
        auto [min, max] = MinMax(result);
        LimitToRange(result, min, max);
    }

    struct ParameterizedEvaluation
    {
        ParameterizedEvaluation(const Tree& tree, const Dataset& dataset, const gsl::span<const double> targetValues, const Range range) 
            : tree_ref(tree)
            , dataset_ref(dataset)
            , target_ref(targetValues)
            , range(range) { }

        template<typename T> bool operator()(T const* const* parameters, T* residuals) const
        {
            auto res = gsl::span<T>(residuals, range.Size());
            Evaluate(tree_ref, dataset_ref, range, parameters[0], res);
            std::transform(res.cbegin(), res.cend(), target_ref.cbegin() + range.Start, res.begin(), [](const T& a, const double b) { return a - b; });
            return true;
        }

        private:
        std::reference_wrapper<const Tree> tree_ref;
        std::reference_wrapper<const Dataset> dataset_ref;
        gsl::span<const double> target_ref;
        Range           range;
    };

    // returns an array of optimized parameters
    template <bool autodiff = true>
    std::vector<double> Optimize(Tree& tree, const Dataset& dataset, const gsl::span<const double> targetValues, const Range range, size_t iterations = 50, bool report = false)
    {
        using ceres::DynamicCostFunction;

        using ceres::DynamicNumericDiffCostFunction;
        using ceres::DynamicAutoDiffCostFunction;
        using ceres::CauchyLoss;
        using ceres::Problem;
        using ceres::Solver;
        using ceres::Solve;

        auto coef = tree.GetCoefficients();
        if (coef.empty())
        {
            return coef;
        }
        if (report)
        {
            fmt::print("x_0: ");
            for(auto c : coef) 
                fmt::print("{} ", c);
            fmt::print("\n");
        }

        auto eval = new ParameterizedEvaluation(tree, dataset, targetValues, range);
        DynamicCostFunction* costFunction;
        if constexpr (autodiff) { costFunction = new DynamicAutoDiffCostFunction<ParameterizedEvaluation, JET_STRIDE>(eval); }
        else                    { costFunction = new DynamicNumericDiffCostFunction(eval);                                   }
        costFunction->AddParameterBlock(coef.size());
        costFunction->SetNumResiduals(range.Size());
        //auto lossFunction = new CauchyLoss(0.5); // see http://ceres-solver.org/nnls_tutorial.html#robust-curve-fitting      

        Problem problem;
        problem.AddResidualBlock(costFunction, nullptr, coef.data());

        Solver::Options options;
        options.max_num_iterations = iterations;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = report;
        options.num_threads = 1;
        Solver::Summary summary;
        Solve(options, &problem, &summary);

        if (report)
        {
            fmt::print("{}\n", summary.BriefReport());
            fmt::print("x_final: ");
            for(auto c : coef) 
                fmt::print("{} ", c);
            fmt::print("\n");
        }
        tree.SetCoefficients(coef);
        return coef;
    }

    // set up some convenience methods using perfect forwarding
    template<typename... Args> auto OptimizeAutodiff(Args&&... args)  
    {
        return Optimize<true>(std::forward<Args>(args)...);
    }

    template<typename... Args> auto OptimizeNumeric(Args&&... args)  
    {
        return Optimize<false>(std::forward<Args>(args)...);
    }
}

#undef FOR
#endif

