# ----------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)

project(
    MO644-Final-Project
    VERSION     1.0
    LANGUAGES   CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Set a default build type if none was specified.
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")

  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)

  set_property(
      CACHE CMAKE_BUILD_TYPE
      PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Make sure to have two executables: one for the serial and one for the parallel version.
set(IRIS_HEADERS
    src/iris/camera.hpp
    src/iris/color.hpp
    src/iris/error.hpp
    src/iris/hit.hpp
    src/iris/hittable.hpp
    src/iris/image.hpp
    src/iris/linalg.hpp
    src/iris/material.hpp
    src/iris/math.hpp
    src/iris/random.hpp
    src/iris/ray.hpp
    src/iris/rng.hpp
    src/iris/scattering.hpp
    src/iris/schlick.hpp
    src/iris/sphere.hpp
    src/iris/prelude.hpp)

add_executable(
    serial
    ${IRIS_HEADERS}
    src/render.hpp
    src/scenes.hpp
    src/main.cpp)

add_executable(
    parallel
    ${IRIS_HEADERS}
    src/render.hpp
    src/scenes.hpp
    src/main.cu)

target_include_directories(serial PRIVATE src/)
target_include_directories(parallel PRIVATE src/)

set_target_properties(serial PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(parallel PROPERTIES LINKER_LANGUAGE CUDA)
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Define "_PARALLEL" when building the parallel executable.
target_compile_definitions(parallel PRIVATE _PARALLEL)

# Specify the GPU architecture.
set(ARCH 61 CACHE STRING "Set the NVIDIA GPU architecture")
message(STATUS "Using -arch=sm_${ARCH}")

target_compile_options(parallel
    PRIVATE "-arch=sm_${ARCH}")

set_property(
    TARGET parallel
    PROPERTY CUDA_ARCHITECTURES OFF)
# ----------------------------------------------------------------------------------------
