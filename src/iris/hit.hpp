#pragma once

#include "prelude.hpp"

#include "linalg.hpp"
#include "ray.hpp"

namespace iris {
#define Fn CudaDeviceFn

class Material; // @Note: forward declaration

// Stores information from an object hit by a ray at point P(t).
// Note: the stored normal is a unit vector and points out from the object.
struct Hit {
    float t; // Note: `ray.at(t) == point`
    Point3 point; // Hit point
    UnitVec3 surface_normal; // Object's surface normal at the hit point
    Material *material; // Material at the hit point

    // Returns the surface normal at the hit point, but with direction against the ray.
    // Note: if the ray is entering the material, then this is simply the surface normal,
    // otherwise, i.e. if the ray is exiting, it is the normal with its direction opposed.
    Fn constexpr UnitVec3 normal_against(Ray const &ray) const {
        // Since the stored normal always points outwards (by convention), we check
        // whether the ray intersects the object from the inside or from the outside
        // by comparing its direction with respect to the surface normal:
        return dot(ray.direction, surface_normal) < 0.0f ? +surface_normal
                                                         : -surface_normal;
    }
};

// Defines the valid interval for hits (t_min <= t < t_max).
struct HitInterval {
    float t_min; // (inclusive)
    float t_max; // (exclusive)

    Fn constexpr bool contains(float t) const { return t < t_max && t_min <= t; }
};

#undef Fn
} // namespace iris
