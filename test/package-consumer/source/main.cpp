// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Package-consumer contract fixture (see test/package-consumer/CMakeLists.txt).
// Intentionally standalone: it must compile and link using ONLY the public
// headers and operon::operon target as they appear after `find_package(operon
// CONFIG REQUIRED)` against an *installed, relocated* package -- it never
// sees the operon source or build tree directly.

#include <cstdlib>
#include <iostream>

#include <operon/core/node.hpp>
#include <operon/core/tree.hpp>
#include <operon/core/version.hpp>

auto main() -> int {
    // A one-node tree is enough to prove that the installed public headers
    // are self-sufficient (no missing transitive includes) and that
    // liboperon's implementation is actually reachable through the
    // operon::operon imported target (UpdateNodes() is defined in
    // source/core/tree.cpp, not header-only).
    auto node = Operon::Node::Constant(2.0);
    Operon::Tree tree({node});
    tree.UpdateNodes();

    if (tree.Length() != 1) {
        std::cerr << "package-consumer: unexpected tree length " << tree.Length() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "package-consumer: " << Operon::Version();
    std::cout << "package-consumer: OK\n";
    return EXIT_SUCCESS;
}
