// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "capabilities.hpp"
#include "cli_args.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/version.hpp"
#include "operon/operators/evaluator.hpp"
#include "population_stats.hpp"

namespace Operon::Test {

TEST_CASE("Cli::ScanExact accepts one whole number and rejects everything else", "[cli]")
{
    using Operon::Cli::ScanExact;

    CHECK(ScanExact<std::size_t>("100") == 100);
    CHECK(ScanExact<std::size_t>("0") == 0);
    CHECK(ScanExact<double>("0.25") == 0.25);
    CHECK(ScanExact<double>("1e-3") == 1e-3);

    CHECK_FALSE(ScanExact<std::size_t>(""));
    CHECK_FALSE(ScanExact<std::size_t>("abc"));
    CHECK_FALSE(ScanExact<std::size_t>("5x")); // trailing characters
    CHECK_FALSE(ScanExact<std::size_t>("5:6"));
    CHECK_FALSE(ScanExact<std::size_t>("-1")); // negative value for an unsigned type
    CHECK_FALSE(ScanExact<std::size_t>("1.5"));
    CHECK_FALSE(ScanExact<double>("x"));
    CHECK_FALSE(ScanExact<double>("nan")); // non-finite floating-point values are never valid parameters
    CHECK_FALSE(ScanExact<double>("inf"));
    CHECK_FALSE(ScanExact<std::size_t>("99999999999999999999999")); // overflow
}

TEST_CASE("Cli::ParseLimit: integers, 'unlimited' and values >= 2^62", "[cli]")
{
    using Operon::Cli::ErrorCode;
    using Operon::Cli::ParseLimit;
    constexpr auto unlimited = std::numeric_limits<std::size_t>::max();

    SECTION("plain integers pass through")
    {
        CHECK(*ParseLimit("evaluations", "0") == 0);
        CHECK(*ParseLimit("evaluations", "1000000") == 1'000'000);
        CHECK(*ParseLimit("generations", "4611686018427387903") == 4611686018427387903ULL); // 2^62 - 1
    }

    SECTION("'unlimited' and values at or above 2^62 mean no limit and normalise to SIZE_MAX")
    {
        CHECK(*ParseLimit("evaluations", "unlimited") == unlimited);
        CHECK(*ParseLimit("generations", "4611686018427387904") == unlimited); // 2^62
        CHECK(*ParseLimit("generations", "18446744073709551615") == unlimited); // UINT64_MAX
    }

    SECTION("anything else is an error naming the option and the offending text")
    {
        for (std::string const text : { "", "Unlimited", "inf", "-1", "1e6", "10x", "1 000", "18446744073709551616" }) {
            INFO("text = '" << text << "'");
            auto const parsed = ParseLimit("evaluations", text);
            REQUIRE_FALSE(parsed);
            CHECK(parsed.error().Code == ErrorCode::InvalidArguments);
            CHECK(parsed.error().Context == "--evaluations");
            CHECK(parsed.error().Message.find("unlimited") != std::string::npos);
        }
    }
}

TEST_CASE("Cli::ParseThreads rejects zero", "[cli]")
{
    using Operon::Cli::ParseThreads;

    CHECK(*ParseThreads(1) == 1);
    CHECK(*ParseThreads(64) == 64);

    auto const zero = ParseThreads(0);
    REQUIRE_FALSE(zero);
    CHECK(zero.error().Context == "--threads");
    CHECK(zero.error().Message.find("at least 1") != std::string::npos);
    CHECK(zero.error().Message.find("omit") != std::string::npos);
}

namespace {
    auto MakeIndividual(std::size_t length, Operon::Scalar fitness) -> Operon::Individual
    {
        Operon::Individual ind { 1 };
        Operon::Vector<Operon::Node> nodes(length, Operon::Node::Constant(1.0));
        ind.Genotype = Operon::Tree(std::move(nodes));
        ind.Fitness[0] = fitness;
        return ind;
    }
} // namespace

TEST_CASE("ComputePopulationMeans excludes rejected and non-finite individuals from the fitness mean", "[cli]")
{
    constexpr auto errMax = Operon::EvaluatorBase::ErrMax;
    constexpr auto inf = std::numeric_limits<Operon::Scalar>::infinity();
    constexpr auto nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();

    SECTION("a healthy population: plain arithmetic means")
    {
        std::vector pop { MakeIndividual(3, 1.0F), MakeIndividual(5, 2.0F), MakeIndividual(7, 6.0F) };
        auto const means = Operon::ComputePopulationMeans(pop, 0);
        CHECK(means.Valid == 3);
        CHECK(means.Fitness == 3.0);
        CHECK(means.Length == 5.0);
    }

    SECTION("ErrMax sentinels, infinities and NaN are not averaged in")
    {
        std::vector pop { MakeIndividual(3, 1.0F), MakeIndividual(5, 3.0F), MakeIndividual(1, errMax),
            MakeIndividual(1, inf), MakeIndividual(1, nan), MakeIndividual(1, -inf) };
        auto const means = Operon::ComputePopulationMeans(pop, 0);
        CHECK(means.Valid == 2);
        CHECK(means.Fitness == 2.0);
        CHECK(means.Length == 2.0); // length is averaged over every individual: (3 + 5 + 4 * 1) / 6
    }

    SECTION("the sum does not overflow the scalar type")
    {
        // Two huge but valid values sum past FLT_MAX in a float accumulator; the double accumulator keeps them finite.
        constexpr auto huge = errMax / 2;
        std::vector pop { MakeIndividual(1, huge), MakeIndividual(1, huge), MakeIndividual(1, huge) };
        auto const means = Operon::ComputePopulationMeans(pop, 0);
        CHECK(means.Valid == 3);
        CHECK(std::isfinite(means.Fitness));
        // Each term is divided by the count before it is summed, so the mean is only accurate to a few ulp.
        CHECK(std::abs(means.Fitness / static_cast<double>(huge) - 1.0) < 1e-12);
    }

    SECTION("a fully failed population reports ErrMax instead of NaN or infinity")
    {
        std::vector pop { MakeIndividual(2, errMax), MakeIndividual(4, nan) };
        auto const means = Operon::ComputePopulationMeans(pop, 0);
        CHECK(means.Valid == 0);
        CHECK(means.Fitness == static_cast<double>(errMax));
        CHECK(means.Length == 3.0);
    }

    SECTION("an empty population is all zeros")
    {
        std::vector<Operon::Individual> pop;
        auto const means = Operon::ComputePopulationMeans(pop, 0);
        CHECK(means.Valid == 0);
        CHECK(means.Fitness == 0.0);
        CHECK(means.Length == 0.0);
    }
}

TEST_CASE("Cli::CapabilitiesJson is valid JSON, escapes strings and lists the options sorted", "[cli]")
{
    Operon::BuildInfo info;
    info.Revision = "abc123";
    info.BuildType = "Release";
    info.Platform = "Linux-6";
    info.Arch = "x86_64";
    info.CompilerId = "Clang";
    info.CompilerVersion = "21.1.8";
    info.CompilerFlags = "-O3 -DNAME=\"a\\b\" -march=x86-64-v3";
    info.Precision = "single";
    info.EigenVersion = "3.4.1";
    info.TaskflowVersion = "4.1.0";
    info.IsaLevel = "x86-64-v3";
    info.IsaFeatures = { "avx2", "fma" };
    info.Jit = true;

    SECTION("a GP-like program with --report-json")
    {
        auto const json = Operon::Cli::CapabilitiesJson("operon_gp", { "seed", "report-json", "dataset" }, info);
        INFO(json);
        CHECK_FALSE(glz::validate_json(json));
        CHECK(json.find("\"schema\":\"operon.capabilities\"") != std::string::npos);
        CHECK(json.find("\"schema_version\":1") != std::string::npos);
        CHECK(json.find("\"program\":\"operon_gp\"") != std::string::npos);
        CHECK(json.find("\"revision\":\"abc123\"") != std::string::npos);
        CHECK(json.find("\"build_type\":\"Release\"") != std::string::npos);
        CHECK(json.find("\"precision\":\"single\"") != std::string::npos);
        CHECK(json.find("\"isa\":{\"level\":\"x86-64-v3\",\"features\":[\"avx2\",\"fma\"]}") != std::string::npos);
        CHECK(json.find("\"jit\":true") != std::string::npos);
        CHECK(json.find("\"machine_report_schema_versions\":[1]") != std::string::npos);
        CHECK(json.find("\"options\":[\"dataset\",\"report-json\",\"seed\"]") != std::string::npos);
        // the quote and backslash in the compiler flags are escaped
        CHECK(json.find("-DNAME=\\\"a\\\\b\\\"") != std::string::npos);
    }

    SECTION("a program without --report-json and on a non-x86 build")
    {
        info.IsaLevel.clear();
        info.IsaFeatures = { "neon" };
        info.Jit = false;
        auto const json = Operon::Cli::CapabilitiesJson("operon_nsgp", { "sorter" }, info);
        INFO(json);
        CHECK_FALSE(glz::validate_json(json));
        CHECK(json.find("\"machine_report_schema_versions\":[]") != std::string::npos);
        CHECK(json.find("\"isa\":{\"level\":null,\"features\":[\"neon\"]}") != std::string::npos);
        CHECK(json.find("\"jit\":false") != std::string::npos);
    }
}

TEST_CASE("Operon::GetBuildInfo describes this build", "[cli]")
{
    auto const info = Operon::GetBuildInfo();
    CHECK(info.Precision == (sizeof(Operon::Scalar) == sizeof(float) ? "single" : "double"));
    CHECK_FALSE(info.Revision.empty());
    CHECK_FALSE(info.BuildType.empty());
    CHECK_FALSE(info.CompilerId.empty());
    CHECK_FALSE(info.EigenVersion.empty());
#if defined(__AVX2__)
    CHECK(std::ranges::find(info.IsaFeatures, "avx2") != info.IsaFeatures.end());
#endif
#if defined(__x86_64__)
    CHECK(info.IsaLevel.starts_with("x86-64-v"));
#endif
#if defined(HAVE_ASMJIT)
    CHECK(info.Jit);
#else
    CHECK_FALSE(info.Jit);
#endif
}

} // namespace Operon::Test
