#ifndef EVALUATE_HPP
#define EVALUATE_HPP

#include <execution>
#include <ceres/ceres.h>
#include "tree.hpp"
#include "dataset.hpp"
#include "gsl/gsl"

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace Operon {
    constexpr gsl::index BATCHSIZE = 64;

    template<typename T> inline std::pair<T, T> MinMax(gsl::span<T> values) noexcept
    {
        // get first finite (not NaN, not infinity) value
        auto min = T(std::numeric_limits<double>::max());
        auto max = T(std::numeric_limits<double>::min());
        for (auto const& v : values)
        {
            if (!ceres::IsFinite(v)) continue;
            if (min > v) min = v;
            if (max < v) max = v;
        }
        return { min, max };
    }

    template<typename T> inline void LimitToRange(gsl::span<T> values, T min, T max) noexcept
    {
        auto mid = (min + max) / 2.0;
        for (auto& v : values)
        {
            if (ceres::IsFinite(v)) { v = std::clamp(v, min, max); }
            else                    { v = mid;                 }
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
        Eigen::Matrix<T, BATCHSIZE, Eigen::Dynamic,  Eigen::ColMajor> m(BATCHSIZE, nodes.size());

        gsl::index numRows = range.Size();
        for (gsl::index row = 0; row < numRows; row += BATCHSIZE)
        {
            gsl::index idx = 0;
            auto remainingRows = std::min(BATCHSIZE, numRows - row);
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                auto r = m.col(i).array();

                switch (auto const& s = nodes[i]; s.Type)
                {
                    case NodeType::Constant:
                        {
                            auto v = parameters == nullptr ? T(s.Value) : parameters[idx++];
                            r.setConstant(v);
                            break;
                        }
                    case NodeType::Variable:
                        {
                            auto values = dataset.GetValues(s.HashValue).subspan(range.Start + row, remainingRows);
                            auto w = parameters == nullptr ? T(s.Value) : parameters[idx++];
                            std::transform(values.begin(), values.end(), r.data(), [&](double v) { return T(v * w); });
                            break;
                        }
                    case NodeType::Add:
                        {
                            auto c = i - 1;             // first child index
                            r = m.col(c).array();
                            for (gsl::index k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                            {
                                r += m.col(j).array();
                            }
                            break;
                        }
                    case NodeType::Sub:
                        {
                            auto c = i - 1;             // first child index
                            if (s.Arity == 1)
                            {
                                r = -m.col(c).array();
                            }
                            else 
                            {
                                r = m.col(c).array();
                                for (gsl::index k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                                {
                                    r -= m.col(j).array();
                                }
                            }
                            break;
                        }
                    case NodeType::Mul:
                        {
                            auto c = i - 1;             // first child index
                            r = m.col(c).array();
                            for (gsl::index k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                            {
                                r  *= m.col(j).array();
                            }
                            break;
                        }
                    case NodeType::Div:
                        {
                            auto c = i - 1;             // first child index
                            if (s.Arity == 1)
                            {
                                r = m.col(c).array().inverse();
                            }
                            else
                            {
                                r = m.col(c).array();
                                for (gsl::index k = 1, j = c - 1 - nodes[c].Length; k < s.Arity; ++k, j -= 1 + nodes[j].Length)
                                {
                                    r /= m.col(j).array(); 
                                }
                            }
                            break;
                        }
                    case NodeType::Log:
                        {
                            r = m.col(i-1).array().log();
                            break;
                        }
                    case NodeType::Exp:
                        {
                            r = m.col(i-1).array().exp();
                            break;
                        }
                    case NodeType::Sin:
                        {
                            r = m.col(i-1).array().sin();
                            break;
                        }
                    case NodeType::Cos:
                        {
                            r = m.col(i-1).array().cos();
                            break;
                        }
                    case NodeType::Tan:
                        {
                            r = m.col(i-1).array().tan();
                            break;
                        }
                    case NodeType::Sqrt:
                        {
                            r = m.col(i-1).array().sqrt();
                            break;
                        }
                    case NodeType::Cbrt:
                        {
                            r = m.col(i-1).array().unaryExpr([](T v) { return ceres::cbrt(v); });
                            break;
                        }
                    case NodeType::Square:
                        {
                            r = m.col(i-1).array().square();
                            break;
                        }
                    default:
                        {
                            std::terminate();
                            break;
                        }
                }
            }
            // the final result is found in the last section of the buffer corresponding to the root node
            std::copy_n(m.rightCols(1).data(), remainingRows, result.begin() + row); 
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
            std::transform(res.cbegin(), res.cend(), target_ref.begin(), res.begin(), [](const T& a, const double b) { return a - b; });
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
    ceres::Solver::Summary Optimize(Tree& tree, const Dataset& dataset, const gsl::span<const double> targetValues, const Range range, size_t iterations = 50, bool writeCoefficients = true, bool report = false)
    {
        using ceres::DynamicCostFunction;
        using ceres::DynamicNumericDiffCostFunction;
        using ceres::DynamicAutoDiffCostFunction;
        using ceres::CauchyLoss;
        using ceres::Problem;
        using ceres::Solver;
        using ceres::Solve;

        Solver::Summary summary;
        auto coef = tree.GetCoefficients();
        if (coef.empty()) 
        { 
            return summary; 
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
        if constexpr (autodiff) { costFunction = new DynamicAutoDiffCostFunction<ParameterizedEvaluation>(eval); }
        else                    { costFunction = new DynamicNumericDiffCostFunction(eval);                       }
        costFunction->AddParameterBlock(coef.size());
        costFunction->SetNumResiduals(range.Size());
        //auto lossFunction = new CauchyLoss(0.5); // see http://ceres-solver.org/nnls_tutorial.html#robust-curve-fitting      

        Problem problem;
        problem.AddResidualBlock(costFunction, nullptr, coef.data());

        Solver::Options options;
        options.max_num_iterations = iterations - 1; // workaround since for some reason ceres sometimes does 1 more iteration
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = report;
        options.num_threads = 1;
        Solve(options, &problem, &summary);

        if (report)
        {
            fmt::print("{}\n", summary.BriefReport());
            fmt::print("x_final: ");
            for(auto c : coef) 
                fmt::print("{} ", c);
            fmt::print("\n");
        }
        if (writeCoefficients)
        {
            tree.SetCoefficients(coef);
        }
        return summary;
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
#endif

