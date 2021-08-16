#pragma once

#include "prelude.hpp"

#include "math.hpp"

namespace iris {
#define Fn CudaDeviceFn

// Compute Schlick's approximation formula for R(θ), given R0 and cos(θ).
Fn constexpr float schlicks_approximation(float r0, float cos_theta_i) {
    assert(cos_theta_i >= 0.0f);
    return r0 + (1.0f - r0) * pow5(1.0f - cos_theta_i);
}

// Compute the reflection coefficient for light incoming parallel to the surface normal
// between two dieletric materials (i.e. the reflectance at normal incidence; the value
// of R0 used in Schlick's Fresnel reflectance approximation).
Fn constexpr float schlicks_normal_reflectance(float ni_over_nt) {
    return square((1.0f - ni_over_nt) / (1.0f + ni_over_nt));
}

// Compute Schlick's approximation of Fresnel reflectance for a dielectric material.
Fn constexpr float schlicks_fresnel_reflectance(float ni_over_nt, float cos_theta_i) {
    // @Note: reflectance (R) is the fraction of power of an incident wave of light
    // that is reflected from an interface between two media, while the fraction of
    // power that is refracted into the second medium is called the transmittance.
    float const r0 = schlicks_normal_reflectance(ni_over_nt);
    return schlicks_approximation(r0, cos_theta_i);
}

#undef Fn
} // namespace iris
