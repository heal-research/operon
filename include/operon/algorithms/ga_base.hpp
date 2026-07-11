// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef GA_BASE_HPP
#define GA_BASE_HPP

#include <string>
#include <operon/operon_export.hpp>
#include "operon/core/types.hpp"
#include "operon/operators/generator.hpp"
#include "config.hpp"
#include "stoppable.hpp"

namespace Operon {

class Problem;
class ReinserterBase;
struct CoefficientInitializerBase;
struct TreeInitializerBase;

class GeneticAlgorithmBase : public StoppableAlgorithm {
public:
    ~GeneticAlgorithmBase() override = default;
    // Not `= default`: parents_/offspring_ are non-owning spans into
    // individuals_'s storage, so a plain memberwise copy would leave the
    // copy's spans pointing at the *source* object's individuals_ buffer
    // (dangling once the source is destroyed, and aliasing it while both
    // are alive). They're rebound here to the copy's own individuals_,
    // exactly like the primary constructor and RestoreIndividuals() do -
    // both derive them from config_.PopulationSize/PoolSize rather than
    // copying the span objects themselves. The atomic stopRequested_ still
    // gets its own handling via StoppableAlgorithm's copy ctor/assign
    // (invoked explicitly here since every other member is memberwise-safe
    // to copy or, like the spans, needs deriving instead).
    GeneticAlgorithmBase(GeneticAlgorithmBase const& other)
        : StoppableAlgorithm(other)
        , config_(other.config_)
        , problem_(other.problem_)
        , treeInit_(other.treeInit_)
        , coeffInit_(other.coeffInit_)
        , generator_(other.generator_)
        , reinserter_(other.reinserter_)
        , individuals_(other.individuals_)
        , parents_(individuals_.data(), config_.PopulationSize)
        , offspring_(individuals_.data() + config_.PopulationSize, config_.PoolSize)
        , workerRngs_(other.workerRngs_)
        , generation_(other.generation_)
        , elapsed_(other.elapsed_)
        , phaseTimes_(other.phaseTimes_)
        , isFitted_(other.isFitted_)
    {
    }
    GeneticAlgorithmBase(GeneticAlgorithmBase&&) = delete;
    auto operator=(GeneticAlgorithmBase const& other) -> GeneticAlgorithmBase&
    {
        if (this == &other) { return *this; }
        StoppableAlgorithm::operator=(other);
        config_ = other.config_;
        problem_ = other.problem_;
        treeInit_ = other.treeInit_;
        coeffInit_ = other.coeffInit_;
        generator_ = other.generator_;
        reinserter_ = other.reinserter_;
        individuals_ = other.individuals_;
        parents_ = Operon::Span<Individual>(individuals_.data(), config_.PopulationSize);
        offspring_ = Operon::Span<Individual>(individuals_.data() + config_.PopulationSize, config_.PoolSize);
        workerRngs_ = other.workerRngs_;
        generation_ = other.generation_;
        elapsed_ = other.elapsed_;
        phaseTimes_ = other.phaseTimes_;
        isFitted_ = other.isFitted_;
        return *this;
    }
    auto operator=(GeneticAlgorithmBase&&) -> GeneticAlgorithmBase& = delete;

    GeneticAlgorithmBase(GeneticAlgorithmConfig config, gsl::not_null<Problem const*> problem, gsl::not_null<TreeInitializerBase const*> treeInit, gsl::not_null<CoefficientInitializerBase const*> coeffInit, gsl::not_null<OffspringGeneratorBase const*> generator, gsl::not_null<ReinserterBase const*> reinserter)
        : config_(config)
        , problem_(problem)
        , treeInit_(treeInit)
        , coeffInit_(coeffInit)
        , generator_(generator)
        , reinserter_(reinserter)
        , individuals_(config.PopulationSize + config.PoolSize)
        , parents_(individuals_.data(), config.PopulationSize)
        , offspring_(individuals_.data() + config.PopulationSize, config.PoolSize)
    {
        generator_->SetCache(config.Cache);
    }

    // Parents()/Offspring() genuinely change *type* between const and
    // non-const (Span<Individual const> vs Span<Individual>), not just
    // reference-qualification - a plain forwarding-reference return would
    // return `Operon::Span<Individual> const&` for a const Self, which
    // still exposes mutable *elements* through that span (Span's
    // operator[] isn't disabled by the span object's own constness), i.e.
    // it wouldn't actually be const-correct. The conditional_t below keeps
    // the const-correctness the original two overloads had.
    //
    // Self& (not Self&&): the original non-ref-qualified overloads were
    // technically callable on an rvalue *this, immediately returning a Span
    // into that rvalue's about-to-be-destroyed individuals_ - a dangling
    // Span by construction. Self& refuses to compile for a *mutable* rvalue
    // caller instead, a deliberate, source-incompatible tightening (no
    // in-repo or downstream caller does this). A const rvalue caller still
    // compiles here (Self deduces to a const type, so Self& becomes
    // Self const&, which binds a const rvalue same as it always could) -
    // this only closes off the mutable-rvalue case, not rvalues generally.
    template<typename Self>
    [[nodiscard]] auto Parents(this Self& self) -> Operon::Span<std::conditional_t<std::is_const_v<Self>, Individual const, Individual>> { return self.parents_; }

    template<typename Self>
    [[nodiscard]] auto Offspring(this Self& self) -> Operon::Span<std::conditional_t<std::is_const_v<Self>, Individual const, Individual>> { return self.offspring_; }

    // Individuals()/WorkerRngs()/Timings() only ever changed reference
    // qualification (T& vs T const&) between overloads (never by-value, so
    // these don't need the conditional_t treatment Generation()/Elapsed()/
    // IsFitted() below get), and ordinary forwarding-reference deduction via
    // `auto&&` reproduces both lvalue cases exactly. It's not a byte-for-
    // byte match for rvalue callers specifically - the old, non-ref-
    // qualified overloads always returned a plain T&/T const& regardless of
    // the object's value category, while `auto&&` forwarding gives T&&/
    // T const&& for a (mutable or const) rvalue caller instead. Not a new
    // hazard (both are still references, and neither version's reference
    // outlives the rvalue any better than the other - no in-repo or
    // downstream caller does this), just a different reference category
    // than before. decltype(auto) would have been the wrong choice here for
    // the lvalue/const-lvalue cases specifically: on an unparenthesized
    // member-access return expression it yields the *declared* member type
    // by value (a silent copy), losing reference identity entirely - see
    // Nodes() in tree.hpp for the full explanation.
    template<typename Self>
    [[nodiscard]] auto&& Individuals(this Self&& self) { return std::forward<Self>(self).individuals_; }

    [[nodiscard]] auto GetProblem() const -> const Problem* { return problem_.get(); }
    [[nodiscard]] auto GetConfig() const -> GeneticAlgorithmConfig { return config_; }

    [[nodiscard]] auto GetTreeInitializer() const -> TreeInitializerBase const* { return treeInit_.get(); }
    [[nodiscard]] auto GetCoefficientInitializer() const -> CoefficientInitializerBase const* { return coeffInit_.get(); }
    [[nodiscard]] auto GetGenerator() const -> OffspringGeneratorBase const* { return generator_.get(); }
    [[nodiscard]] auto GetReinserter() const -> ReinserterBase const* { return reinserter_.get(); }

    template<typename Self>
    [[nodiscard]] auto&& WorkerRngs(this Self&& self) { return std::forward<Self>(self).workerRngs_; }

    // Generation()/Elapsed()/IsFitted() are trivially-copyable scalars whose
    // non-const overload existed only to allow direct assignment (e.g.
    // `algo.IsFitted() = true;`); neither original overload was ref-
    // qualified, so the const one returned by value regardless of whether
    // the object was an lvalue or rvalue. A plain `auto&&` here would get
    // that wrong for a const *rvalue* specifically: Self deduces to a
    // non-reference type, and forwarding through it yields a `T const&&`
    // binding - a reference into a temporary - where the old code always
    // handed back a safe, independent copy. (Reading through a const
    // *lvalue* isn't affected: only the reference category changes there,
    // same as Individuals()/WorkerRngs()/Timings() above.)
    //
    // No caller does this today (confirmed: nothing in this repo or its
    // downstream consumers calls these getters on a const rvalue), but it's
    // one line of conditional_t to close off entirely rather than leave as
    // a latent trap. Note this deliberately does *not* forward self: naming
    // `self` inside the function body is always an lvalue expression
    // regardless of which reference type Self deduced to, so branching the
    // return *type* on constness alone (ignoring value category) is enough
    // to reproduce the original by-value-for-const/reference-for-mutable
    // split exactly, for all four Self×value-category combinations.
    template<typename Self>
    [[nodiscard]] auto Generation(this Self&& self) -> std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, size_t, size_t&> { return self.generation_; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template<typename Self>
    [[nodiscard]] auto Elapsed(this Self&& self) -> std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, double, double&> { return self.elapsed_; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template<typename Self>
    [[nodiscard]] auto&& Timings(this Self&& self) { return std::forward<Self>(self).phaseTimes_; }

    template<typename Self>
    [[nodiscard]] auto IsFitted(this Self&& self) -> std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, bool, bool&> { return self.isFitted_; } // NOLINT(cppcoreguidelines-missing-std-forward)

    // StopRequested()/RequestStop() are inherited from StoppableAlgorithm.

    // Valid to call between runs only. The PhaseTimer observer owns its own
    // totals and is recreated each Run(), so Reset() mid-run would cause the
    // next reportProgress sync to overwrite the cleared map with stale data.
    auto Reset() -> void
    {
        generation_ = 0;
        elapsed_ = 0;
        ClearStopRequested();
        phaseTimes_.clear();
        GetGenerator()->Evaluator()->Reset();
    }

    auto RestoreIndividuals(std::vector<Individual> inds) -> void
    {
        EXPECT(inds.size() == config_.PoolSize + config_.PopulationSize,
                "Mismatched number of individuals (must match pool/population sizes)");
        individuals_ = std::move(inds);
        parents_ = Operon::Span<Individual>(individuals_.data(), config_.PopulationSize);
        offspring_ = Operon::Span<Individual>(individuals_.data() + config_.PopulationSize, config_.PoolSize);
    }
    
private:
    GeneticAlgorithmConfig config_;

    gsl::not_null<Problem const*> problem_;
    gsl::not_null<TreeInitializerBase const*> treeInit_;
    gsl::not_null<CoefficientInitializerBase const*> coeffInit_;
    gsl::not_null<OffspringGeneratorBase const*> generator_;
    gsl::not_null<ReinserterBase const*> reinserter_;

    Operon::Vector<Individual> individuals_;
    Operon::Span<Individual> parents_;
    Operon::Span<Individual> offspring_;

    std::vector<Operon::RandomGenerator> workerRngs_;
    size_t generation_{0};
    double elapsed_{0};
    Operon::Map<std::string, double> phaseTimes_;
    bool isFitted_{false};
};

} // namespace Operon

#endif
