#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads(); // workaround __syncthreads warning
#    define KERNEL_ARG2(grid, block)
#    define KERNEL_ARG3(grid, block, sh_mem)
#    define KERNEL_ARG4(grid, block, sh_mem, stream)
#else
#    define KERNEL_ARG2(grid, block) <<< grid, block >>>
#    define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#    define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif

// Reference: https://stackoverflow.com/questions/51959774/cuda-in-vscode
