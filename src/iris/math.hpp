#pragma once

#include "prelude.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace iris {

//
// Constants
//

constexpr float Infinity = std::numeric_limits<float>::infinity(); // +∞
constexpr float Eps = std::numeric_limits<float>::epsilon(); // The machine epsilon
constexpr float Tau = static_cast<float>(6.28318530717958647692528676655900577); // τ
constexpr float Pi = static_cast<float>(3.14159265358979323846264338327950288); // π

//
// Floating-point
//

CudaDeviceFn constexpr float deg2rad(float angle_in_degrees) {
    return angle_in_degrees * (Tau / 360.0f);
}
CudaDeviceFn constexpr float rad2deg(float angle_in_radians) {
    return angle_in_radians * (360.0f / Tau);
}

CudaHostFn CudaDeviceFn constexpr bool approx_eq(float lhs, float rhs, float eps = Eps) {
    if (lhs == rhs) { return true; }

    float const abs_diff = abs(lhs - rhs);

    // @Note: by using a single epsilon we're assuming equal values
    // for the absolute and relative tolerances. If we wanted to have
    // different values for them, we could use instead:
    //  |
    //  |   return abs_diff <= max(abs_eps, rel_eps * max(abs(lhs), abs(rhs)));
    //

    return abs_diff <= eps || abs_diff <= eps * fmax(abs(lhs), abs(rhs));
}

//
// Template
//

// clang-format off
template <typename T> CudaDeviceFn constexpr T
square(T x) { return x * x; }
template <typename T> CudaDeviceFn constexpr T
cube(T x) { return x * x * x; }
template <typename T> CudaDeviceFn constexpr T
pow5(T x) { return square(square(x)) * x; }

template <typename T> CudaHostFn CudaDeviceFn constexpr T
clamp(T x, T min, T max) {
    if (x < min) { return min; }
    if (x > max) { return max; }
    return x;
}
template <typename T> CudaHostFn CudaDeviceFn constexpr T
saturate(T x) { return clamp(x, T(0.0), T(1.0)); }
// clang-format on

} // namespace iris
