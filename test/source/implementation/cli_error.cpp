// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include "cli_error.hpp"

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace Operon::Test {

TEST_CASE("Cli::Invoke converts exceptions and passes values through", "[cli]")
{
    using Operon::Cli::ErrorCode;
    using Operon::Cli::Invoke;

    SECTION("throwing lambda becomes a Cli::Error instead of escaping") {
        auto const result = Invoke([]() -> int { throw std::runtime_error("boom"); },
            ErrorCode::Input, "infix expression");
        CHECK_FALSE(result);
        CHECK(result.error().Code == ErrorCode::Input);
        CHECK(result.error().Context == "infix expression");
        CHECK(result.error().Message == "boom");
    }

    SECTION("non-throwing lambda's return value passes through unchanged") {
        auto const result = Invoke([] { return 42; }, ErrorCode::Runtime, "unused");
        REQUIRE(result);
        CHECK(*result == 42);
    }

    SECTION("void-returning lambda yields a successful Result<void>") {
        // Invoke's return type is Result<decltype(fn())> = tl::expected<void,
        // Error> here: the void instantiation must compile and report success.
        bool ran = false;
        auto const result = Invoke([&] { ran = true; }, ErrorCode::Runtime, "unused");
        CHECK(result);
        CHECK(ran);
    }
}

TEST_CASE("Cli::Report prints the error and returns EXIT_FAILURE", "[cli]")
{
#if !defined(_WIN32)
    // Capture stderr via dup2 so the format contract ('error: <context>:
    // <message>', and 'error: <message>' when the context is empty) is
    // asserted against what Report actually wrote.
    auto const capture = [](Operon::Cli::Error const& error) -> std::pair<int, std::string> {
        auto const path = std::filesystem::temp_directory_path() / "operon_cli_error_report_stderr.txt";
        std::fflush(stderr);
        auto const saved = ::dup(STDERR_FILENO);
        REQUIRE(saved != -1);
        auto const fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        REQUIRE(fd != -1);
        REQUIRE(::dup2(fd, STDERR_FILENO) != -1);
        auto const code = Operon::Cli::Report(error);
        std::fflush(stderr);
        ::dup2(saved, STDERR_FILENO);
        ::close(saved);
        ::close(fd);
        std::ifstream in(path);
        std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        return {code, text};
    };

    auto const [codeWithContext, textWithContext] =
        capture({Operon::Cli::ErrorCode::Configuration, "shape-constraints config", "bad json"});
    CHECK(codeWithContext == EXIT_FAILURE);
    CHECK(textWithContext.find("error: shape-constraints config: bad json") != std::string::npos);

    auto const [codeBare, textBare] =
        capture({Operon::Cli::ErrorCode::Input, "", "no context"});
    CHECK(codeBare == EXIT_FAILURE);
    CHECK(textBare.find("error: no context") != std::string::npos);
#else
    CHECK(Operon::Cli::Report({Operon::Cli::ErrorCode::Input, "", "no context"}) == EXIT_FAILURE);
#endif
}

} // namespace Operon::Test
