#!/bin/bash -exu

export CC=clang
export CXX=clang++

# Create a clean build environment
cmake -E remove -f build
cmake -E make_directory build

# Configure CMake and set CUDA arch
cmake -S . -B build -DARCH=35 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Compile
cmake --build build

set +x

# Run serial
echo "::: Running (serial) :::"
for i in 1 2 3 4 5 6; do
    ./build/serial $i output/ser.$i.ppm
    echo "----------------------------------"
done

# Run parallel
echo "::: Running (parallel) :::"
for i in 1 2 3 4 5 6; do
    ./build/parallel $i output/par.$i.ppm
    echo "----------------------------------"
done


# Compare serial results
echo "::: Comparing (serial) :::"
for i in 1 2 3 4 5 6; do
    python3 compare.py $i output/ser.$i.ppm tests/ser.$i.ref.ppm
    echo "-------------------------"
done

# Compare parallel results
echo "::: Comparing (parallel) :::"
for i in 1 2 3 4 5 6; do
    python3 compare.py $i output/par.$i.ppm tests/par.$i.ref.ppm
    echo "-------------------------"
done

# exec /bin/bash