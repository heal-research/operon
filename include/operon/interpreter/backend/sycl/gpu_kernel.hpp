// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GPU_KERNEL_HPP
#define OPERON_GPU_KERNEL_HPP

#include "gpu_encoded.hpp"

namespace Operon::Sycl {

// Run the stack-machine kernel over the encoded population.
// Returns a [PopSize × NRows] float buffer of raw model outputs, individual-major.
auto RunKernel(EncodedPopulation const& enc) -> std::vector<float>;

} // namespace Operon::Sycl

#endif // OPERON_GPU_KERNEL_HPP
