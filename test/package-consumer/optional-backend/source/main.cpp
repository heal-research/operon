// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Links operon::core only, against a package whose backend adapter is
// reported as not found (see ../CMakeLists.txt).

#include <cstdlib>

#include <operon/core/tree.hpp>

auto main() -> int
{
    Operon::Tree const tree;
    return tree.Length() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
