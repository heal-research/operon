// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/comparison.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "pareto_front.hpp"

namespace Operon::Test {

namespace {
    auto Make(std::vector<std::vector<Operon::Scalar>> const& fits, std::vector<size_t> const& ranks)
        -> std::vector<Individual>
    {
        std::vector<Individual> pop;
        for (auto i = 0UL; i < fits.size(); ++i) {
            Individual ind(fits[i].size());
            for (auto j = 0UL; j < fits[i].size(); ++j) {
                ind[j] = fits[i][j];
            }
            ind.Rank = ranks[i];
            pop.push_back(std::move(ind));
        }
        return pop;
    }
} // namespace

TEST_CASE("SelectParetoFront ignores stale ranks and uses fitness", "[cli][pareto]")
{
    // Ranks as left behind by sorting parents+offspring: the rank-0 member (0,0) was discarded by the reinserter, so
    // the retained individuals carry nonzero ranks although they are mutually non-dominated or dominate each other.
    auto pop = Make({ { 1, 4 }, { 2, 2 }, { 4, 1 }, { 3, 3 }, { 2, 2 }, { 5, 5 } }, { 1, 1, 2, 3, 0, 0 });
    auto const front = SelectParetoFront({ pop.data(), pop.size() }, RankIntersectSorter {}, 0);
    // (3,3) and (5,5) are dominated; (2,2) duplicate reported once (first); ordered by first objective.
    REQUIRE(front.size() == 3);
    CHECK(front[0] == &pop[0]);
    CHECK(front[1] == &pop[1]);
    CHECK(front[2] == &pop[2]);
}

TEST_CASE("SelectParetoFront on empty and single populations", "[cli][pareto]")
{
    std::vector<Individual> none;
    CHECK(SelectParetoFront({ none.data(), none.size() }, RankIntersectSorter {}, 0).empty());
    auto one = Make({ { 1 } }, { 7 });
    CHECK(SelectParetoFront({ one.data(), one.size() }, RankIntersectSorter {}, 0).size() == 1);
}

TEST_CASE("SelectParetoFront uses the epsilon-aware dominance of the run", "[cli][pareto]")
{
    // a = (0, 1) and b = (5e-7, 0): below the tolerance b's degradation in objective 0 does not count, so b
    // dominates a. With eps = 0 the two are a genuine trade-off. (The deductive sorter applies the tolerance to every
    // pair; the rank-intersect and merge sorters assume exact lexicographic order and keep both, and the export
    // follows whichever sorter the run used.)
    auto pop = Make({ { 0, 1 }, { 5e-7F, 0 } }, { 0, 0 });
    auto const span = Operon::Span<Individual const> { pop.data(), pop.size() };
    auto const exact = SelectParetoFront(span, DeductiveSorter {}, 0);
    REQUIRE(exact.size() == 2);
    auto const tolerant = SelectParetoFront(span, DeductiveSorter {}, 1e-6F);
    REQUIRE(tolerant.size() == 1);
    CHECK(tolerant[0] == &pop[1]);
}

TEST_CASE("SelectParetoFront collapses epsilon-equivalent duplicates", "[cli][pareto]")
{
    auto pop = Make({ { 1, 2 }, { 1 + 4e-7F, 2 - 3e-7F }, { 3, 0 } }, { 0, 0, 0 });
    auto const span = Operon::Span<Individual const> { pop.data(), pop.size() };
    CHECK(SelectParetoFront(span, RankIntersectSorter {}, 0).size() == 3);
    auto const front = SelectParetoFront(span, RankIntersectSorter {}, 1e-6F);
    REQUIRE(front.size() == 2);
    // the lexicographically smallest representative of the equivalent pair is kept, ordered by first objective
    CHECK(front[0] == &pop[0]);
    CHECK(front[1] == &pop[2]);
}

namespace {
    // Two structural objectives on a 3e-7 grid, so many pairs of individuals are within, or just beyond, the 1e-6
    // tolerance of each other.
    struct NsgaFixture {
        static constexpr Operon::Scalar Step = 3e-7F;
        static constexpr std::size_t PopSize = 300;

        std::vector<std::vector<Operon::Scalar>> Cols { 2, std::vector<Operon::Scalar>(10) };
        Operon::Dataset Ds;
        Operon::Problem Problem;
        std::vector<Operon::Hash> Inputs;
        Operon::TreePropertyEvaluator LengthEval;
        Operon::TreePropertyEvaluator DepthEval;
        Operon::MultiEvaluator Evaluator;
        Operon::SubtreeCrossover Crossover { 0.9, 10, 30 };
        Operon::MultiMutation Mutator;
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> OnePoint;
        Operon::NormalCoefficientInitializer CoeffInit;
        Operon::BalancedTreeCreator Creator;
        Operon::UniformTreeInitializer TreeInit;
        Operon::CrowdedComparison Comp;
        Operon::TournamentSelector FemSel { Comp };
        Operon::TournamentSelector MaleSel { Comp };
        Operon::BasicOffspringGenerator Generator;
        Operon::KeepBestReinserter KeepBest { Comp };
        Operon::ReplaceWorstReinserter ReplaceWorst { Comp };

        NsgaFixture()
            : Ds([this] {
                for (auto i = 0UL; i < 10; ++i) {
                    Cols[0][i] = static_cast<Operon::Scalar>(i);
                    Cols[1][i] = static_cast<Operon::Scalar>(2 * i);
                }
                return Operon::Dataset({ "X", "Y" }, Cols);
            }())
            , Problem(gsl::not_null<Operon::Dataset*>(&Ds))
            , Inputs { Ds.GetVariable("X")->Hash }
            , LengthEval(
                  &Problem,
                  [](Operon::Tree const& t) -> Operon::Scalar {
                      return Step * static_cast<Operon::Scalar>(t.Length() % 7);
                  },
                  1.F)
            , DepthEval(
                  &Problem,
                  [](Operon::Tree const& t) -> Operon::Scalar {
                      return Step * static_cast<Operon::Scalar>((t.Depth() + 2 * t.Length()) % 7);
                  },
                  1.F)
            , Evaluator(&Problem)
            , Creator(&Problem.GetPrimitiveSet(), Inputs, 0.0, 30)
            , TreeInit(&Creator)
            , Generator(&Evaluator, &Crossover, &Mutator, &FemSel, &MaleSel)
        {
            Problem.SetTrainingRange({ 0, 10 });
            Problem.SetTestRange({ 0, 10 });
            Problem.SetTarget("Y");
            Problem.ConfigurePrimitiveSet(Operon::PrimitiveSet::Arithmetic);
            Problem.SetInputs(Inputs);
            Evaluator.Add(&LengthEval);
            Evaluator.Add(&DepthEval);
            Mutator.Add(&OnePoint, 1.0);
            TreeInit.ParameterizeDistribution(1U, 30U);
            TreeInit.SetMaxDepth(10);
        }

        [[nodiscard]] auto Config(std::size_t generations, double eps) const -> Operon::GeneticAlgorithmConfig
        {
            Operon::GeneticAlgorithmConfig config {};
            config.PopulationSize = PopSize;
            config.PoolSize = PopSize;
            config.Generations = generations;
            config.Epsilon = eps;
            return config;
        }
    };

    using FitnessSet = std::multiset<std::vector<Operon::Scalar>>;

    auto Fitnesses(std::vector<Individual const*> const& front) -> FitnessSet
    {
        FitnessSet out;
        for (auto const* ind : front) {
            out.emplace(ind->Fitness.begin(), ind->Fitness.end());
        }
        return out;
    }

    // The non-dominated set of `parents` computed by brute force from the pairwise epsilon-aware dominance relation,
    // after collapsing epsilon-equivalent individuals onto the lexicographically smallest one. Valid for the sorters
    // whose first front is that set (the deductive sorter, and every sorter at eps = 0).
    auto BruteForceFront(Operon::Span<Individual const> parents, Operon::Scalar eps) -> FitnessSet
    {
        std::vector<std::vector<Operon::Scalar>> fits;
        for (auto const& p : parents) {
            fits.emplace_back(p.Fitness.begin(), p.Fitness.end());
        }
        std::ranges::stable_sort(fits);
        std::vector<std::vector<Operon::Scalar>> unique;
        for (auto const& f : fits) {
            if (unique.empty() || !Operon::Equal {}(unique.back(), f, eps)) {
                unique.push_back(f);
            }
        }
        FitnessSet out;
        for (auto const& a : unique) {
            auto const dominated = std::ranges::any_of(unique, [&](auto const& b) -> bool {
                return Operon::ParetoDominance {}(b.begin(), b.end(), a.begin(), a.end(), eps)
                    == Operon::Dominance::Left;
            });
            if (!dominated) {
                out.insert(a);
            }
        }
        return out;
    }
} // namespace

TEST_CASE("SelectParetoFront equals NSGA2::Best() of the initial population under the same epsilon", "[cli][pareto]")
{
    // With no generation completed, Best() and Parents() describe the same, initially sorted population. (After a
    // completed generation Best() is the front of parents plus offspring, which reinsertion then reduces, so the two
    // differ; the next test covers that case.)
    NsgaFixture f;
    RankIntersectSorter const rankIntersect;
    MergeSorter const merge;
    DeductiveSorter const deductive;
    for (NondominatedSorterBase const* sorter : { static_cast<NondominatedSorterBase const*>(&rankIntersect),
             static_cast<NondominatedSorterBase const*>(&merge),
             static_cast<NondominatedSorterBase const*>(&deductive) }) {
        for (auto const eps : { 0.0, 1e-6, 5e-6 }) {
            Operon::NSGA2 nsga { f.Config(0, eps), &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator, &f.KeepBest,
                sorter };
            Operon::RandomGenerator rng { 42 };
            nsga.Run(rng, nullptr, /*threads=*/1);
            REQUIRE(nsga.Parents().size() == NsgaFixture::PopSize);

            FitnessSet retained;
            for (auto const& ind : nsga.Best()) {
                retained.emplace(ind.Fitness.begin(), ind.Fitness.end());
            }
            auto const parents = nsga.Parents();
            auto const front
                = SelectParetoFront({ parents.data(), parents.size() }, *sorter, static_cast<Operon::Scalar>(eps));
            INFO("eps = " << eps);
            CHECK(!front.empty());
            CHECK(Fitnesses(front) == retained);
            if (eps > 0) { // a larger tolerance merges more individuals
                CHECK(front.size() < parents.size());
            }
        }
    }
}

TEST_CASE("SelectParetoFront lists the non-dominated retained parents after completed generations", "[cli][pareto]")
{
    // After completed generations the export is the front of the retained parents: it is computed here by brute
    // force from the pairwise dominance relation, independently of the sorter and of NSGA2::Best(). The reinserter
    // changes which parents are retained, so both are covered.
    NsgaFixture f;
    DeductiveSorter const deductive;
    RankIntersectSorter const rankIntersect;
    for (auto const replaceWorst : { false, true }) {
        Operon::ReinserterBase& reinserter = replaceWorst ? static_cast<Operon::ReinserterBase&>(f.ReplaceWorst)
                                                          : static_cast<Operon::ReinserterBase&>(f.KeepBest);
        for (auto const eps : { 0.0, 1e-6 }) {
            for (NondominatedSorterBase const* sorter : { static_cast<NondominatedSorterBase const*>(&deductive),
                     static_cast<NondominatedSorterBase const*>(&rankIntersect) }) {
                if (sorter == &rankIntersect && eps > 0) {
                    continue; // does not implement the pairwise epsilon relation, see the earlier test
                }
                Operon::NSGA2 nsga { f.Config(3, eps), &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator, &reinserter,
                    sorter };
                Operon::RandomGenerator rng { 7 };
                nsga.Run(rng, nullptr, /*threads=*/1);
                REQUIRE(nsga.Generation() == 3);
                REQUIRE(nsga.Parents().size() == NsgaFixture::PopSize);

                auto const parents = nsga.Parents();
                auto const front
                    = SelectParetoFront({ parents.data(), parents.size() }, *sorter, static_cast<Operon::Scalar>(eps));
                INFO("replace-worst = " << replaceWorst << ", eps = " << eps);
                CHECK(!front.empty());
                CHECK(front.size() < parents.size()); // dominated and duplicate individuals were dropped
                // every exported individual is one of the retained parents
                for (auto const* ind : front) {
                    CHECK(ind >= parents.data());
                    CHECK(ind < parents.data() + parents.size());
                }
                CHECK(Fitnesses(front)
                    == BruteForceFront({ parents.data(), parents.size() }, static_cast<Operon::Scalar>(eps)));
            }
        }
    }
}

} // namespace Operon::Test
