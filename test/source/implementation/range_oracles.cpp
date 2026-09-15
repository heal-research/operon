// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Range oracle data is a hermetic projection of the authoritative Florian
// Bachinger JDIQ repository. See test/data/range-oracles/florian-jdiq.tsv.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../operon_test.hpp"

#include "operon/core/constraint.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/problem.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {
namespace {
    using S = Operon::Scalar;
    using IE = IntervalEvaluator<S>;

    struct Domain {
        std::string name;
        S lo;
        S hi;
    };
    struct Bounds {
        S lo;
        S hi;
    };
    struct RangeOracle {
        std::string id;
        std::string expression;
        std::vector<Domain> domains;
        Bounds expected;
        S tolerance;
    };

    [[nodiscard]] auto CorpusPath() -> std::string
    {
        return std::string { OPERON_TEST_SOURCE_DIR } + "/test/data/range-oracles/florian-jdiq.tsv";
    }

    [[nodiscard]] auto Split(std::string const& line) -> std::vector<std::string>
    {
        std::vector<std::string> fields;
        std::istringstream input(line);
        for (std::string field; std::getline(input, field, '\t');) {
            fields.push_back(std::move(field));
        }
        return fields;
    }

    [[nodiscard]] auto Number(std::string const& text) -> S { return static_cast<S>(std::stod(text)); }

    [[nodiscard]] auto Oracles() -> std::vector<RangeOracle>
    {
        std::ifstream input(CorpusPath());
        if (!input) {
            throw std::runtime_error("cannot open range oracle corpus");
        }
        std::vector<RangeOracle> oracles;
        for (std::string line; std::getline(input, line);) {
            if (line.empty() || line.starts_with('#')) {
                continue;
            }
            auto const fields = Split(line);
            if (fields.front() == "oracle") {
                if (fields.size() != 6) {
                    throw std::runtime_error("malformed range oracle record");
                }
                oracles.push_back({ fields[1], fields[2], {}, { Number(fields[3]), Number(fields[4]) }, Number(fields[5]) });
            } else if (fields.front() == "domain") {
                if (fields.size() != 5) {
                    throw std::runtime_error("malformed range oracle domain");
                }
                auto const it = std::find_if(oracles.begin(), oracles.end(), [&](auto const& oracle) { return oracle.id == fields[1]; });
                if (it == oracles.end()) {
                    throw std::runtime_error("domain precedes its range oracle");
                }
                it->domains.push_back({ fields[2], Number(fields[3]), Number(fields[4]) });
            } else {
                throw std::runtime_error("unknown range oracle record");
            }
        }
        return oracles;
    }

    struct Prepared {
        Dataset dataset;
        Tree tree;
        IE::DomainMap domains;
    };
    [[nodiscard]] auto Prepare(RangeOracle const& oracle) -> Prepared
    {
        std::vector<std::string> names;
        std::vector<std::vector<S>> values;
        for (auto const& domain : oracle.domains) {
            names.push_back(domain.name);
            values.push_back({ (domain.lo + domain.hi) / S { 2 } });
        }
        names.push_back("Y");
        values.push_back({ S { 0 } });
        Dataset dataset(std::move(names), std::move(values));
        auto tree = InfixParser::Parse(oracle.expression, dataset);
        IE::DomainMap domains;
        for (auto const& domain : oracle.domains) {
            domains.emplace(dataset.GetVariable(domain.name)->Hash, std::pair { domain.lo, domain.hi });
        }
        return { std::move(dataset), std::move(tree), std::move(domains) };
    }

    [[nodiscard]] auto BisectedBound(Prepared& prepared, RangeOracle const& oracle) -> Bounds
    {
        Problem problem(&prepared.dataset);
        problem.SetTrainingRange({ 0, 1 });
        problem.SetTestRange({ 0, 1 });
        problem.SetTarget("Y");
        problem.SetLinearScalingEnabled(false);
        DispatchTable<S> dtable;
        Evaluator<DispatchTable<S>> metric(&problem, &dtable, NMSE {});
        ShapeConstraintSet constraints;
        for (auto const& domain : oracle.domains) {
            constraints.Domains.emplace(domain.name, std::pair { domain.lo, domain.hi });
        }
        constraints.Constraints.push_back({ .Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair { S { -1e6 }, S { 1e6 } } });
        ShapeConstrainedEvaluator evaluator(&metric, &dtable, constraints);
        evaluator.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
        evaluator.SetBoundOptions({ .BisectionDepth = 5 });
        auto const result = evaluator.Measure(prepared.tree);
        REQUIRE(result.Measurements.size() == 1);
        REQUIRE(result.Measurements.front().Bound.has_value());
        auto const [lo, hi] = *result.Measurements.front().Bound;
        return { lo, hi };
    }

    void CheckContains(RangeOracle const& oracle, Bounds bound)
    {
        CAPTURE(oracle.id);
        INFO("expected: [" << oracle.expected.lo << ", " << oracle.expected.hi << "]");
        CHECK(bound.lo <= oracle.expected.lo + oracle.tolerance);
        CHECK(bound.hi >= oracle.expected.hi - oracle.tolerance);
    }
} // namespace

TEST_CASE("Range-oracle corpus: interval and bisection enclosures", "[range-oracles][shape-constraints]")
{
    for (auto const& oracle : Oracles()) {
        auto prepared = Prepare(oracle);
        auto const direct = IE(&prepared.tree, prepared.domains).Evaluate(prepared.tree.GetCoefficients());
        auto const bisected = BisectedBound(prepared, oracle);
        CheckContains(oracle, { direct.inf(), direct.sup() });
        CheckContains(oracle, bisected);
        CHECK(bisected.lo >= direct.inf() - oracle.tolerance);
        CHECK(bisected.hi <= direct.sup() + oracle.tolerance);
    }
}
} // namespace Operon::Test
