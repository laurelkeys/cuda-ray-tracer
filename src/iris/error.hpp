#pragma once

#include "prelude.hpp"

#include <cstdio>

namespace iris {

enum class Error {
    None = 0,
    /*EXIT_FAILURE = 1*/
    MissingArguments = 2,
    InvalidArgumentValue,
    WrongIrisExecutable = 42,
    CUDA = 99,
};

#ifdef _PARALLEL
#    define CUDACHECK(cudaCmd) cudaCheckError((cudaCmd), __FILE__, __LINE__)
void cudaCheckError(cudaError_t err, char const *const filename, int const line) {
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "CUDA error (%d) at %s:%d: %s: \"%s\"\n",
            static_cast<uint>(err),
            filename,
            line,
            cudaGetErrorName(err),
            cudaGetErrorString(err));
        cudaDeviceReset();
        exit(static_cast<int>(Error::CUDA));
    }
}
#endif

} // namespace iris
