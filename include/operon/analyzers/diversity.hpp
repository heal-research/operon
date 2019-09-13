#ifndef DIVERSITY_HPP
#define DIVERSITY_HPP

#include <execution>
#include "core/operator.hpp"

namespace Operon
{
    template<typename T>
    class PopulationDiversityAnalyzer : PopulationAnalyzerBase<T>
    {
        public:
            double operator()(operon::rand_t&, gsl::index i) const
            {
                return this->diversityMatrix.row(i).mean();
            }

            double Diversity() const { return this->diversityMatrix.mean(); }

            void Prepare(const gsl::span<T> pop)
            {
                //auto individuals = std::vector<T>(pop.begin(), pop.end());

                std::vector<std::vector<operon::hash_t>> hashes(pop.size());
                std::vector<gsl::index> indices(pop.size());
                std::iota(indices.begin(), indices.end(), 0);

                auto hashTree = [&](gsl::index i)
                {
                    auto& ind = pop[i];
                    ind.Genotype.Sort();
                    const auto& nodes = ind.Genotype.Nodes();
                    auto& h = hashes[i];
                    h.resize(nodes.size());
                    std::transform(nodes.begin(), nodes.end(), h.begin(), [](const auto& node) { return node.CalculatedHashValue; });
                    std::sort(h.begin(), h.end());
                };

                std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), hashTree);

                this->diversityMatrix = Eigen::MatrixXd::Zero(hashes.size(), hashes.size());

                for (size_t i = 0; i < hashes.size() - 1; ++i)
                {
                    for (size_t j = i + 1; j < hashes.size(); ++j)
                    {
                        auto distance = CalculateDistance(hashes[i], hashes[j]);
                        this->diversityMatrix(i, j) = distance;
                        this->diversityMatrix(j, i) = distance;
                    }
                }
            }

        private:
            double CalculateDistance(const std::vector<operon::hash_t>& lhs, const std::vector<operon::hash_t>& rhs)
            {
                size_t count = 0;
                double total = lhs.size() + rhs.size();

                for (size_t i = 0, j = 0; i < lhs.size() && j < rhs.size(); )
                {
                    if (lhs[i] == rhs[j])
                    {
                        ++count;
                        ++i;
                        ++j;
                    }
                    else if (lhs[i] < rhs[j])
                    {
                        ++i;
                    }
                    else 
                    {
                        ++j;
                    }
                }
                auto distance = (total - count) / total;

                return distance;
            }

    };
}

#endif

