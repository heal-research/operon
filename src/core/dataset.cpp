/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
 */

#include "core/dataset.hpp"
#include "core/jsf.hpp"
#include <fmt/core.h>

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
    Random::JsfRand<64> jsf(1234);
    std::vector<operon::hash_t> hashes(ncols);
    std::generate(hashes.begin(), hashes.end(), [&]() { return jsf(); });
    std::sort(hashes.begin(), hashes.end());
    for (auto i = 0; i < ncols; ++i) {
        variables[i].Hash = hashes[i];
    }

    auto [r, c] = Dimensions();
    if (r == 0 || c == 0) {
        throw std::runtime_error(fmt::format("Invalid matrix dimensions {} x {}\n", r, c));
    }
}
}
