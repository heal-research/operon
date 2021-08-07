#ifndef OPERON_PARETO_EFFICIENT_NONDOMINATED_SORT
#define OPERON_PARETO_EFFICIENT_NONDOMINATED_SORT

#include "core/individual.hpp"
#include "core/operator.hpp"

#include "robin_hood.h"

namespace Operon {

namespace detail {
    template<typename T>
    std::vector<T> make_indices(T start, size_t n)  {
        static_assert(std::is_integral_v<T>);
        std::vector<T> vec(n);
        std::iota(vec.begin(), vec.end(), start);
        return vec;
    }
}

// Zhang et al. 2014 - "An Efficient Approach to Nondominated Sorting for Evolutionary Multiobjective Optimization"
// https://doi.org/10.1109/TEVC.2014.2308305
// BB: this method is very simple and elegant and works very well for small m, but scales badly with the number of objectives
template<int BinarySearch = 0>
struct EfficientSorter : public NondominatedSorterBase {
    inline std::vector<std::vector<size_t>> operator()(Operon::RandomGenerator&, Operon::Span<Operon::Individual const> pop) const {
        size_t m = pop.front().Fitness.size();
        ENSURE(m > 1);
        switch (m) {
        case 2:
            return Sort<2>(pop);
        case 3:
            return Sort<3>(pop);
        case 4:
            return Sort<4>(pop);
        case 5:
            return Sort<5>(pop);
        case 6:
            return Sort<6>(pop);
        case 7:
            return Sort<7>(pop);
        default:
            return Sort<0>(pop);
        }
    }

    private:
    template<size_t N>
    std::vector<std::vector<size_t>> Sort(Operon::Span<Operon::Individual const> pop) const noexcept
    {
        auto idx = detail::make_indices(0, pop.size());
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            ++this->Stats.LexicographicalComparisons; return pop[a].LexicographicalCompare(pop[b]);
        });
        // check if individual i is dominated by any individual in the front f
        auto dominated = [&](auto const& f, size_t i) {
            return !std::all_of(f.rbegin(), f.rend(), [&](size_t j) {
                return pop[j].ParetoCompare<N>(pop[i]) == Dominance::None;
            });
        };

        std::vector<std::vector<size_t>> fronts;
        for (auto i : idx) {
            size_t k;
            if constexpr (BinarySearch) { // binary search
                k = std::distance(fronts.begin(), std::partition_point(fronts.begin(), fronts.end(), [&](auto const& f) { return dominated(f, i); }));
            } else { // sequential search
                k = std::distance(fronts.begin(), std::find_if(fronts.begin(), fronts.end(), [&](auto const& f) { return !dominated(f, i); }));
            }
            if (k == fronts.size()) {
                fronts.push_back({});
            }
            fronts[k].push_back(i);
        }
        //fmt::print("ncomp: {}\n", ncomp);
        return fronts;
    }
};

} // namespace Operon

#endif
