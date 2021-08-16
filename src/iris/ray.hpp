#pragma once

#include "prelude.hpp"

#include "linalg.hpp"

namespace iris {
#define Fn CudaDeviceFn

struct Ray {
public:
    Point3 origin;
    UnitVec3 direction;

    // P(t) = origin + t * direction
    Fn constexpr Point3 at(float t) const { return origin + t * direction; }
};

#undef Fn
} // namespace iris
