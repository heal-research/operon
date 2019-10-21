/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#include <fmt/core.h>
#include "core/dataset.hpp"

#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#include "csv.hpp"
#pragma GCC diagnostic warning "-Wreorder"
#pragma GCC diagnostic warning "-Wignored-qualifiers"
#pragma GCC diagnostic warning "-Wunknown-pragmas"

namespace Operon {
Dataset::Dataset(const std::string& file, bool hasHeader)
{
    auto info = csv::get_file_info(file);
    auto nrows = info.n_rows;
    auto ncols = info.n_cols;
    variables.resize(ncols);

    if (hasHeader) {
        for (auto i = 0; i < ncols; ++i) {
            variables[i].Name = info.col_names[i];
            variables[i].Index = i;
        }
    } else {
        for (auto i = 0; i < ncols; ++i) {
            variables[i].Name = fmt::format("X{}", i + 1);
            variables[i].Index = i;
        }
        variables.back().Name = "Y";
    }

    csv::CSVReader reader(file);

    values = MatrixType(nrows, ncols);
    gsl::index i = 0;
    for (auto& row : reader) {
        gsl::index j = 0;
        for (auto& field : row) {
            if (field.is_num()) {
                values(i, j) = field.get<double>();
                ++j;
            } else {
                throw new std::runtime_error(fmt::format("Could not cast {} as a floating-point type.", field.get()));
            }
        }
        ++i;
    }

    std::sort(variables.begin(), variables.end(), [&](const Variable& a, const Variable& b) { return CompareWithSize(a.Name, b.Name); });
    // fill in variable hash values using a fixed seed
    operon::rand_t random(1234);
    std::vector<operon::hash_t> hashes(ncols);
    std::generate(hashes.begin(), hashes.end(), [&]() { return random(); });
    std::sort(hashes.begin(), hashes.end());
    for (auto i = 0; i < ncols; ++i) {
        variables[i].Hash = hashes[i];
    }
}
}
