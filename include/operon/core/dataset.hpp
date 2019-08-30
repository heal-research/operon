#ifndef DATASET_H
#define DATASET_H

#include <unordered_map>
#include <vector>
#include <exception>
#include <algorithm>
#include <numeric>

#include "common.hpp"

namespace Operon {
    // compare strings size first, as an attempt to have eg X1, X2, X10 in this order and not X1, X10, X2
    namespace 
    {
        inline bool CompareWithSize(const std::string& lhs, const std::string& rhs) 
        {
            return std::make_tuple(lhs.size(), lhs) < std::make_tuple(rhs.size(), rhs);            
        }
    }

    class Dataset
    {
        private:
            std::vector<Variable> variables;
            std::vector<std::vector<double>> values;

            Dataset();

        public:
            Dataset(const std::string& file, bool hasHeader = false); 
            Dataset(const Dataset& rhs) : variables(rhs.variables), values(rhs.values) { } 
            Dataset(Dataset&& rhs) noexcept : variables(rhs.variables), values(std::move(rhs.values)){ }
            Dataset(const std::vector<Variable> vars, const std::vector<std::vector<double>> vals) : variables(vars), values(std::move(vals)) {} 

            Dataset& operator=(Dataset rhs)
            {
                swap(rhs);
                return *this;
            }

            void swap(Dataset& rhs) noexcept
            {
                variables.swap(rhs.variables);
                values.swap(rhs.values);
            }

            size_t Rows() const { return values[0].size(); }
            size_t Cols() const { return variables.size(); }
            std::pair<size_t, size_t> Dimensions() const { return { Rows(), Cols() }; }

            const std::vector<std::string> VariableNames() const 
            {
                std::vector<std::string> names;
                std::transform(variables.begin(), variables.end(), std::back_inserter(names), [](const Variable& v) { return v.Name; }); 
                return names;
            }

            const gsl::span<const double> GetValues(const std::string& name) const 
            {
                auto needle = Variable { name, 0, 0 }; 
                auto record = std::equal_range(variables.begin(), variables.end(), needle, [&](const Variable& a, const Variable& b) { return CompareWithSize(a.Name, b.Name); }); 
                return gsl::span<const double>(values[record.first->Index]);
            }

            const gsl::span<const double> GetValues(operon::hash_t hashValue) const 
            {
                auto needle = Variable { "", hashValue, 0 };
                auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return a.Hash < b.Hash; }); 
                return gsl::span<const double>(values[record.first->Index]);
            }

            const gsl::span<const double> GetValues(gsl::index index) const 
            {
                return gsl::span<const double>(values[index]); 
            }

            const std::string& GetName(operon::hash_t hashValue) const
            {
                auto needle = Variable { "", hashValue, 0 };
                auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return a.Hash < b.Hash; }); 
                return record.first->Name;
            }

            operon::hash_t GetHashValue(const std::string& name) const
            {
                auto needle = Variable { name, 0 /* hash value */, 0 /* index */ };
                auto record = std::equal_range(variables.begin(), variables.end(), needle, [](const Variable& a, const Variable& b) { return CompareWithSize(a.Name, b.Name); }); 
                return record.first->Hash;
            }

            const std::string& GetName(int index) const { return variables[index].Name; }
            const gsl::span<const Variable> Variables() const { return gsl::span<const Variable>(variables); }
    };
}

#endif

