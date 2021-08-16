#pragma once

#include "prelude.hpp"

#include "linalg.hpp"
#include "math.hpp"

namespace iris {
#define Fn CudaDeviceFn

// clang-format off

// Reflects an incomming vector `v` around a surface normal `n`.
Fn constexpr Vec3
reflect(Vec3 const &v, UnitVec3 const &n) {
    float const cos_theta_i = dot(-v, n);
    assert(cos_theta_i >= 0.0f);

    return v + 2.0f * cos_theta_i * n;
}
Fn constexpr UnitVec3
reflect(UnitVec3 const &v, UnitVec3 const &n) {
    // @Safety: the reflection of a unit vector is also a unit vector.
    return reflect(v.as_vec(), n).as_unit();
}

// Refracts an incoming vector `v` into a material with surface normal `n`, where
// `ni_over_nt` is the ratio of the index of refraction of the incident medium (ηi)
// to the one of the transmitted medium (ηt). No vector is returned in case of total
// internal reflection.
Fn inline bool
refract(UnitVec3 const &v, UnitVec3 const &n, float ni_over_nt, float cos_theta_i, Vec3 &maybe_refracted) {
    float const sin2_theta_i = 1.0f - square(cos_theta_i);
    float const sin2_theta_t = square(ni_over_nt) * sin2_theta_i;
    if (sin2_theta_t >= 1.0f) { return false; } // total internal reflection
    float const cos_theta_t = std::sqrt(1.0f - sin2_theta_t);

    maybe_refracted = ni_over_nt * v + (ni_over_nt * cos_theta_i - cos_theta_t) * n;
    return true;
}
Fn inline bool
refract(UnitVec3 const &v, UnitVec3 const &n, float ni_over_nt, Vec3 &maybe_refracted) {
    float const cos_theta_i = dot(-v, n);
    assert(cos_theta_i >= 0.0f);

    return refract(v, n, ni_over_nt, cos_theta_i, maybe_refracted);
}

// clang-format on

#undef Fn
} // namespace iris
