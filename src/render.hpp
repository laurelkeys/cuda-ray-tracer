#pragma once

#include "iris/prelude.hpp"

#include "iris/camera.hpp"
#include "iris/color.hpp"
#include "iris/hit.hpp"
#include "iris/hittable.hpp"
#include "iris/linalg.hpp"
#include "iris/material.hpp"
#include "iris/random.hpp"
#include "iris/ray.hpp"
#include "iris/sphere.hpp"

#ifdef _PARALLEL
#    include <curand_kernel.h>
#endif

using namespace iris;

#ifdef _PARALLEL
__global__ void init_render_rng(int width, int height, curandState *rand_state) {
    int const i = threadIdx.x + blockIdx.x * blockDim.x;
    int const j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < width && j < height) {
        int const idx = j * width + i;
        curand_init(RNG_SEED + idx, 0, 0, &rand_state[idx]);
    }
}
#endif

CudaDeviceFn Rgb
color_at(Ray const &ray, int depth, HittableList const &world, Rng *rng) {
    Ray bounced_ray = ray;
    Rgb attenuation = Rgb::white();
    Sky const sky { Rgb::white(), Rgb(0.5f, 0.7f, 1.0f) };

    // Either return the sky color if the ray didn't hit anything,
    // or attenuate the color from ray bounces until one hits the sky.
    for (int ray_bounce = 0; ray_bounce < depth; ++ray_bounce) {
        Hit hit;
        MaybeScatter scattered;
        if (world.hit(bounced_ray, HitInterval { NEAR_ZERO, Infinity }, hit)
            && hit.material->scatter(bounced_ray, hit, scattered, rng)) {
            attenuation *= scattered.attenuation;
            bounced_ray = scattered.ray;
        } else {
            return attenuation * sky.color_at(bounced_ray);
        }
    }

    // No light is gathered if we exceed the ray bounce limit.
    return Rgb::black();
}

CudaGlobalFn void render(
    Rgb *fb,
    int width,
    int height,
    int spp, // samples per pixel
    int depth, // max ray bounces
    Camera **camera,
    HittableList **world,
    Rng *rand_state) {
#define DO_RENDER                                                                \
    Rgb color_sum { 0.0f, 0.0f, 0.0f };                                          \
    for (int sample = 0; sample < spp; ++sample) {                               \
        float const u = (x + Random::scalar(&rng)) / (width - 1.0f);             \
        float const v = (y + Random::scalar(&rng)) / (height - 1.0f);            \
        color_sum += color_at((*camera)->ray(u, v, &rng), depth, **world, &rng); \
    }                                                                            \
    fb[idx] = color_sum / static_cast<float>(spp);

#ifdef _PARALLEL
    int const x = threadIdx.x + blockIdx.x * blockDim.x;
    int const y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        int const idx = y * width + x;
        curandState rng = rand_state[idx];
        DO_RENDER
    }
#else
#    ifdef USE_PBRT_RNG
    Rng rng = *rand_state;
#    else
    Unused(rand_state);
    Rng rng = 0;
#    endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int const idx = y * width + x;
            DO_RENDER
        }
    }
#endif
}
