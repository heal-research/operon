#ifndef DATASET_H
#define DATASET_H

#include <unordered_map>
#include <vector>
#include <exception>
#include <fstream>
#include <charconv>
#include <utility>
#include <fmt/core.h>
#include <type_traits>
#include <map>
#include <algorithm>
#include <numeric>

#include "stats.hpp"

#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#include "csv-parser/include/csv.hpp"
#pragma GCC diagnostic warning "-Wreorder"
#pragma GCC diagnostic warning "-Wignored-qualifiers"
#pragma GCC diagnostic warning "-Wunknown-pragmas"
#include "jsf.hpp"

namespace Operon {

    namespace detail {
        struct Variable
        {
            std::string         name;
            uint64_t            hash;
            size_t              index;
        };

        // compare strings size first, as an attempt to have eg X1, X2, X10 in this order and not X1, X10, X2
        inline bool CompareWithSize(const std::string& lhs, const std::string& rhs) 
        {
            auto lSize = lhs.size(), rSize = rhs.size();
            return lSize == rSize ? lhs < rhs : lSize < rSize;
        }
    }

    struct Range 
    {
        size_t Start;
        size_t End;

        inline int Size() const { return End - Start; }
    };

    class Dataset
    {
        private:
            using Variable = detail::Variable;
            std::vector<Variable> variables;
            std::vector<std::vector<double>> values;

            Dataset();

        public:
            Dataset(const std::string& file, bool hasHeader = false) 
            {
                csv::CSVReader reader(file);
                auto ncol = reader.get_col_names().size();
                variables.resize(ncol);
                values.resize(ncol);

                // fill in variable names
                std::vector<std::string> names;
                if (hasHeader)
                {
                    names = reader.get_col_names();
                }
                else
                {
                    int i = 0;
                    std::generate(names.begin(), names.end(), [&]() { return fmt::format("X{}", ++i); });
                }

                for (auto& row : reader)
                {
                    int i = 0;
                    for (auto& field : row) 
                    {
                        if (field.is_num())
                        {
                            auto value = field.get<double>();
                            values[i++].push_back(value); 
                        }
                        else 
                        {
                            throw std::runtime_error(fmt::format("The field {} could not be parsed as a number.", field.get()));
                        }
                    }
                }

                // the code below sorts the variables vector by variable name, then assigns hash values (also sorted, increasing order) and indices
                // this is done for the purpose of enabling searching with std::equal_range so we can retrieve data using name, hash value or index

                // fill in variable names 
                for (size_t i = 0; i < ncol; ++i) 
                {
                    variables[i].name = names[i];
                    variables[i].index = i;
                }

                std::sort(variables.begin(), variables.end(), [&](const Variable& a, const Variable& b) { return detail::CompareWithSize(a.name, b.name); });
                // fill in variable hash values
                Random::JsfRand<64> jsf(1234);
                // generate some hash values 
                std::vector<uint64_t> hashes(ncol);
                std::generate(hashes.begin(), hashes.end(), [&]() { return jsf(); });
                std::sort(hashes.begin(), hashes.end());
                for (size_t i = 0; i < ncol; ++i)
                {
                    variables[i].hash = hashes[i];
                }
            }

            Dataset(const Dataset& rhs) : variables(rhs.variables), values(rhs.values) { } 
            Dataset(Dataset&& rhs) noexcept : variables(rhs.variables), values(std::move(rhs.values)){ }
            Dataset(const std::vector<Variable> vars, const std::vector<std::vector<double>> vals) : variables(vars), values(std::move(vals)) {} 

            Dataset& operator=(Dataset rhs)
            {
                swap(rhs);
                return *this;
            }

            Dataset& operator=(Dataset&& rhs)
            {
                swap(rhs);
                return *this;
            }

            void swap(Dataset& rhs) noexcept
            {
                std::swap(variables, rhs.variables); 
                std::swap(values, rhs.values);
            }

            size_t Rows() const { return values[0].size(); }
            size_t Cols() const { return variables.size(); }

            const std::vector<std::string> VariableNames() const 
            {
                std::vector<std::string> names;
                std::transform(variables.begin(), variables.end(), std::back_inserter(names), [](const Variable& v) { return v.name; }); 
                return names;
            }

            const std::vector<double>& GetValues(const std::string& name) const 
            {
                auto needle = Variable { name, 0, 0 }; 
                auto record = std::equal_range(variables.begin(), variables.end(), needle, [&](const Variable& a, const Variable& b) { return detail::CompareWithSize(a.name, b.name); }); 
                return values[record.first->index];            
            }

            const std::vector<double>& GetValues(uint64_t hashValue) const 
            {
                auto needle = Variable { "", hashValue, 0 };
                auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return a.hash < b.hash; }); 
                return values[record.first->index];
            }

            const std::vector<double>& GetValues(int index) const 
            {
                return values[index]; 
            }

            const std::string& GetName(uint64_t hashValue) const
            {
                auto needle = Variable { "", hashValue, 0 };
                auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return a.hash < b.hash; }); 
                return record.first->name;
            }

            uint64_t GetHashValue(const std::string& name) const
            {
                auto needle = Variable { name, 0 /* hash value */, 0 /* index */ };
                auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return detail::CompareWithSize(a.name, b.name); }); 
                return record.first->hash;
            }

            const std::string& GetName(int index) const { return variables[index].name; }
            const std::vector<Variable>& Variables() const { return variables; }
    };
}

#endif
