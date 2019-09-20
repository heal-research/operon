#ifndef DIVERSITY_HPP
#define DIVERSITY_HPP

#include <Eigen/Core>

#include "core/operator.hpp"
#include <execution>

namespace Operon {
template <typename T>
class PopulationDiversityAnalyzer : PopulationAnalyzerBase<T> {
public:
    double operator()(operon::rand_t&, gsl::index i) const
    {
        return this->diversityMatrix.row(i).mean();
    }

    double HybridDiversity() const
    {
        double mean = 0;
        auto dim = diversityMatrix.rows();
        for (Eigen::Index i = 0; i < dim - 1; ++i) {
            for (Eigen::Index j = i + 1; j < dim; ++j) {
                mean += diversityMatrix(i, j);
            }
        }
        return mean / (dim * (dim - 1) / 2);
    }

    double StructuralDiversity() const
    {
        double mean = 0;
        auto dim = diversityMatrix.rows();
        for (Eigen::Index i = 0; i < dim - 1; ++i) {
            for (Eigen::Index j = i + 1; j < dim; ++j) {
                mean += diversityMatrix(j, i);
            }
        }
        return mean / (dim * (dim - 1) / 2);
    }

    void Prepare(const gsl::span<T> pop)
    {
        std::vector<std::vector<operon::hash_t>> hashesHybrid(pop.size());
        std::vector<std::vector<operon::hash_t>> hashesStruct(pop.size());

        std::vector<gsl::index> indices(pop.size());
        std::iota(indices.begin(), indices.end(), 0);

        auto hashTree = [&](gsl::index i) {
            auto& ind = pop[i];
            const auto& nodes = ind.Genotype.Nodes();
            // hybrid hashing
            ind.Genotype.Sort(/* strict = */ true);
            auto& hHybrid = hashesHybrid[i];
            hHybrid.resize(nodes.size());
            std::transform(nodes.begin(), nodes.end(), hHybrid.begin(), [](const auto& node) { return node.CalculatedHashValue; });
            std::sort(hHybrid.begin(), hHybrid.end());
            // structural hashing
            ind.Genotype.Sort(/* strict = */ false);
            auto& hStruct = hashesStruct[i];
            hStruct.resize(nodes.size());
            std::transform(nodes.begin(), nodes.end(), hStruct.begin(), [](const auto& node) { return node.CalculatedHashValue; });
            std::sort(hStruct.begin(), hStruct.end());
        };

        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), hashTree);

        this->diversityMatrix = Eigen::MatrixXd::Zero(pop.size(), pop.size());

        std::for_each(std::execution::par_unseq, indices.begin(), indices.end() - 1, [&](gsl::index i) {
            for (size_t j = i + 1; j < indices.size(); ++j) {
                diversityMatrix(i, j) = CalculateDistance(hashesHybrid[i], hashesHybrid[j]);
                diversityMatrix(j, i) = CalculateDistance(hashesStruct[i], hashesStruct[j]);
            };
        });
    }

private:
    Eigen::MatrixXd diversityMatrix;
    double CalculateDistance(const std::vector<operon::hash_t>& lhs, const std::vector<operon::hash_t>& rhs)
    {
        size_t count = 0;
        double total = lhs.size() + rhs.size();

        for (size_t i = 0, j = 0; i < lhs.size() && j < rhs.size();) {
            if (lhs[i] == rhs[j]) {
                ++count;
                ++i;
                ++j;
            } else if (lhs[i] < rhs[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        auto distance = (total - count) / total;

        return distance;
    }
};
}

#endif
