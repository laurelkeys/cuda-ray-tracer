#pragma once

#ifdef _VSCODE
#    define _PARALLEL
#    include "cudaHeaders.cuh"
#endif

//
// Common headers
//

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

//
// Typedefs
//

typedef int8_t i8;
typedef uint8_t u8;

typedef std::size_t usize;
typedef unsigned int uint;

//
// Macros
//

#define Unused(x) ((void) (x))

#define DivUp(dividend, divisor) ((dividend + divisor - 1) / divisor)

#ifdef _PARALLEL
#    define CudaHostFn __host__
#    define CudaDeviceFn __device__
#    define CudaGlobalFn __global__
#    define ThreadZero threadIdx.x == 0 && blockIdx.x == 0
#else
#    define CudaHostFn
#    define CudaDeviceFn
#    define CudaGlobalFn
#    define ThreadZero true
#endif

/* #define USE_PBRT_RNG */

#ifdef _PARALLEL
#    define Rng curandState
#else
#    ifdef USE_PBRT_RNG
#        define Rng pbrt::RNG
#    else
#        define Rng int
#    endif
#endif

//
// Constants
//

#define RNG_SEED 970

#define NEAR_ZERO 0.0001f
#define NEAR_ZERO_SQRT 0.01f

//
// Notes
//

/* Links:
 *
 *  Docker
 *      Installing on Windows Home + using it with WSL 2
 *          https://docs.docker.com/docker-for-windows/install-windows-home/
 *          https://docs.docker.com/docker-for-windows/wsl/
 *      Working with containers in VS Code
 *          https://code.visualstudio.com/docs/containers/overview
 *
 *  WSL 2
 *      GPU support
 *          https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
 *
 *  CMake
 *      Documentation for 3.10 and the latest version (currently 3.19)
 *          https://cmake.org/cmake/help/v3.10/
 *          https://cmake.org/cmake/help/latest/
 *      "An Introduction to Modern CMake" and other resources
 *          https://cliutils.gitlab.io/modern-cmake/
 *
 *  VS Code
 *      WSL + Containers
 *          https://code.visualstudio.com/docs/cpp/config-wsl
 *          https://code.visualstudio.com/docs/containers/overview
 *
 * CUDA
 *      Function execution space specifies
 *          https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers
 *
 * Further improvements
 *      Other ray tracer implementations using CUDA (to look into in the future)
 *          https://github.com/mikoro/valo
 *          https://github.com/MeirBon/CudaTracer
 *          https://github.com/Khrylx/DSGPURayTracing
 *          https://github.com/HardCoreCodin/SlimTracin
 *          https://github.com/sergeneren/volumetric-bvh
 *          https://github.com/henrikglass/cuda-path-tracer
 *          https://github.com/jan-van-bergen/GPU-Pathtracer
 *          https://github.com/TheSandvichMaker/BUAS-Pathtracer
 *          https://github.com/lukedan/Project3-CUDA-Path-Tracer
 *          https://github.com/Clement-Pirelli/ROIW_based-Raytracer
 *          https://github.com/ZheyuanXie/CUDA-Path-Tracer-Denoising
 *          https://github.com/voxel-tracer/raytracinginoneweekendincuda
 *
 * */
