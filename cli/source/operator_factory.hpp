// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_CLI_OPERATOR_FACTORY_HPP
#define OPERON_CLI_OPERATOR_FACTORY_HPP

#include <scn/detail/scan.h>                   // for scan
#include <stddef.h>                            // for size_t
#include <memory>                              // for unique_ptr, make_unique
#include <string>                              // for operator==, string
#include <utility>                             // for addressof
#include <vector>                              // for vector
#include "operon/core/types.hpp"               // for Span
#include "operon/interpreter/interpreter.hpp"  // for Interpreter
#include "operon/operators/selector.hpp"       // for SelectorBase, Proporti...
#include "util.hpp"                            // for Split
namespace Operon { class EvaluatorBase; }
namespace Operon { class KeepBestReinserter; }
namespace Operon { class OffspringGeneratorBase; }
namespace Operon { class PrimitiveSet; }
namespace Operon { class Problem; }
namespace Operon { class ReinserterBase; }
namespace Operon { class ReplaceWorstReinserter; }
namespace Operon { struct CreatorBase; }
namespace Operon { struct CrossoverBase; }
namespace Operon { struct MutatorBase; }
namespace Operon { struct Variable; }

namespace Operon {

template<typename Comp>
auto ParseReinserter(std::string const& str, Comp&& comp)
{
    std::unique_ptr<ReinserterBase> reinserter;
    if (str == "keep-best") {
        reinserter = std::make_unique<KeepBestReinserter>(comp);
    } else if (str == "replace-worst") {
        reinserter = std::make_unique<ReplaceWorstReinserter>(comp);
    }
    return reinserter;
}

template<typename Comp>
auto ParseSelector(std::string const& str, Comp&& comp) -> std::unique_ptr<Operon::SelectorBase>
{
    auto tok = Split(str, ':');
    auto name = tok[0];
    std::unique_ptr<Operon::SelectorBase> selector;
    constexpr size_t defaultTournamentSize{5};
    if (name == "tournament") {
        selector = std::make_unique<Operon::TournamentSelector>(comp);
        size_t tournamentSize{defaultTournamentSize};
        if (tok.size() > 1) { scn::scan(tok[1], "{}", tournamentSize); }
        dynamic_cast<Operon::TournamentSelector*>(selector.get())->SetTournamentSize(tournamentSize);
    } else if (name == "proportional") {
        selector = std::make_unique<Operon::ProportionalSelector>(comp);
        dynamic_cast<Operon::ProportionalSelector*>(selector.get())->SetObjIndex(0);
    } else if (name == "rank") {
        selector = std::make_unique<Operon::RankTournamentSelector>(comp);
        size_t tournamentSize{defaultTournamentSize};
        if (tok.size() > 1) { scn::scan(tok[1], "{}", tournamentSize); }
        dynamic_cast<Operon::RankTournamentSelector*>(selector.get())->SetTournamentSize(tournamentSize);
    } else if (name == "random") {
        selector = std::make_unique<Operon::RandomSelector>();
    }
        
    return selector;
}

auto ParseCreator(std::string const& str, PrimitiveSet const& pset, Operon::Span<Variable const> inputs) -> std::unique_ptr<CreatorBase>;

auto ParseEvaluator(std::string const& str, Problem& problem, Interpreter& interpreter) -> std::unique_ptr<EvaluatorBase>;

auto ParseGenerator(std::string const& str, EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel) -> std::unique_ptr<OffspringGeneratorBase>;

} // namespace Operon

#endif
