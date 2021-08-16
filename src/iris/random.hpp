#pragma once

#include "prelude.hpp"

#include "linalg.hpp"
#include <cstdlib>

#ifdef _PARALLEL
#    include <curand_kernel.h>
#endif

namespace iris {
#define Fn CudaDeviceFn

namespace Random {

    //
    // Scalars
    //

    // Returns a random real number in [0.0, 1.0).
    Fn inline float scalar(Rng *rng) {
#ifdef _PARALLEL
        // @Note: curand_uniform returns a float value in (0.0, 1.0].
        return 1.0f - curand_uniform(rng);
#else
#    ifdef USE_PBRT_RNG
        // @Note: pbrt::RNG generates returns a float value in [0.0, 1.0).
        return rng->UniformFloat();
#    else
        // @Note: rand returns an int value in [0, RAND_MAX].
        Unused(rng);
        return rand() / (static_cast<float>(RAND_MAX) + 1.0f);
#    endif
#endif
    }

    // Returns a random real number in [min, max).
    Fn inline float scalar_in(float min, float max, Rng *rng) {
        return min + (max - min) * scalar(rng);
    }

    //
    // Vectors
    //

    // Returns a 3D vector with random coordinates in [0.0, 1.0).
    Fn inline Vec3 vector(Rng *rng) {
        return Vec3 { scalar(rng), scalar(rng), scalar(rng) };
    }

    // Returns a 3D vector with random coordinates in [min, max).
    Fn inline Vec3 vector_in(float min, float max, Rng *rng) {
        return Vec3 {
            scalar_in(min, max, rng),
            scalar_in(min, max, rng),
            scalar_in(min, max, rng),
        };
    }

    // Returns a vector to a random point inside a sphere with radius 1.
    Fn inline Vec3 vector_in_unit_sphere(Rng *rng) {
        // Pick a random point by rejection sampling.
        while (true) {
            Vec3 const sample = vector_in(-1.0f, 1.0f, rng);
            if (sample.length_squared() <= 1.0f) { return sample; }
        }
    }

    // Returns a vector to a random point on the surface of a sphere with radius 1.
    Fn inline UnitVec3 vector_on_unit_sphere(Rng *rng) {
        // Pick a random point inside the unit sphere and normalize it to length 1.
        return vector_in_unit_sphere(rng).into_unit();
    }

} // namespace Random

#undef Fn
} // namespace iris
