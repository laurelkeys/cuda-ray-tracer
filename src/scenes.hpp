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
__global__ void init_scene_rng(curandState *rand_state) {
    if (ThreadZero) { curand_init(RNG_SEED, 0, 0, rand_state); }
}
#endif

CudaGlobalFn void free_scene(HittableList **d_world, Camera **d_camera) {
    for (int obj = 0; obj < (*d_world)->list_size; ++obj) {
        delete ((Sphere *) (*d_world)->list[obj])->material;
        delete (*d_world)->list[obj];
    }
    delete *d_world;
    delete *d_camera;
}

namespace Preset {

// Reference:
// https://raytracing.github.io/books/RayTracingInOneWeekend.html#wherenext?/afinalrender

CudaGlobalFn void rtiow_final_scene(
    HittableList **d_world, Camera **d_camera, float aspect_ratio, Rng *rand_state) {
    if (ThreadZero) {
        Rng rng = *rand_state;

        // clang-format off
        int const list_size = 22 * 22 + 1 + 3;
        Hittable **objects = new Hittable *[list_size];
        int obj = 0;

        objects[obj++] = new Sphere(
            Point3 { 0, -1000.0, 0 }, 1000, new Lambertian(Rgb(0.5, 0.5, 0.5)));

        for (int a = -11; a < 11; ++a) {
            for (int b = -11; b < 11; ++b) {
                float const x = a + 0.9 * Random::scalar(&rng);
                float const z = b + 0.9 * Random::scalar(&rng);
                Point3 const center { x, 0.2, z };

                float const choose_material = Random::scalar(&rng);
                if (choose_material < 0.8) {
                    // diffuse
                    objects[obj++] = new Sphere(center, 0.2,
                        new Lambertian(
                            Rgb::from_vec(Random::vector(&rng) * Random::vector(&rng))));

                } else if (choose_material < 0.95) {
                    // metal
                    objects[obj++] = new Sphere(center, 0.2,
                        new Metal(
                            Rgb::from_vec(Random::vector_in(0.0, 0.5, &rng)),
                            Random::scalar_in(0.0, 0.5, &rng)));

                } else {
                    // glass
                    objects[obj++] = new Sphere(center, 0.2,
                        new Dielectric(1.5));
                }
            }
        }

        objects[obj++] = new Sphere( // middle glass sphere
            Point3 {  0, 1, 0 }, 1.0, new Dielectric(1.5));
        objects[obj++] = new Sphere( // back diffuse sphere
            Point3 { -4, 1, 0 }, 1.0, new Lambertian(Rgb(0.4, 0.2, 0.1)));
        objects[obj++] = new Sphere( // front metal sphere
            Point3 {  4, 1, 0 }, 1.0, new Metal(Rgb(0.7, 0.6, 0.5), 0.0));

        assert(obj == list_size);
        *d_world = new HittableList(objects, list_size);
        // clang-format on

        *rand_state = rng;

        Point3 const look_from { 13, 2, 3 };
        Point3 const look_at { 0, 0, 0 };
        Vec3 const view_up { 0, 1, 0 };

        float const vfov = 30.0f; // 25.0f;
        float const aperture = 0.1f; // 0.0f;
        float const focus_distance = 10.0f; // (look_from - look_at).length();

        *d_camera = new Camera({
            look_from,
            look_at,
            view_up,
            vfov,
            aspect_ratio,
            aperture,
            focus_distance,
        });
    }
}

// Reference:
// https://github.com/define-private-public/PSRayTracing/tree/master/src/Scenes

CudaGlobalFn void three_spheres(
    HittableList **d_world, Camera **d_camera, float aspect_ratio, Rng *rand_state) {
    if (ThreadZero) {
        Unused(rand_state);

        // clang-format off
        int const list_size = 9;
        Hittable **objects = new Hittable *[list_size];

        float const tiny_y = 0.005;
        objects[0] = new Sphere(Point3 { -1, tiny_y, -1 },  0.5,  new Dielectric(1.5));
        objects[1] = new Sphere(Point3 { -1, tiny_y, -1 }, -0.35, new Dielectric(1.5));
        objects[2] = new Sphere(Point3 { -1, tiny_y, -1 },  0.3,  new Dielectric(1.5));
        objects[3] = new Sphere(Point3 { -1, tiny_y, -1 }, -0.2,  new Dielectric(1.5));
        objects[4] = new Sphere(Point3 { -1, tiny_y, -1 },  0.15, new Dielectric(1.5));
        objects[5] = new Sphere(Point3 { -1, tiny_y, -1 },  0.1,  new Lambertian(Rgb(0.1, 0.2, 1)));
        objects[6] = new Sphere(Point3 {  0,   0.15, -1 },  0.5,  new Metal(Rgb(0.7, 0.7, 0.75), 0.5));
        objects[7] = new Sphere(Point3 {  1, tiny_y, -1 },  0.5,  new Metal(Rgb(0.8, 0.6, 0.2), 1.0));
        objects[8] = new Sphere(Point3 {  0, -100.5, -1 },  100,  new Lambertian(Rgb(0.8, 0.8, 0)));

        *d_world = new HittableList(objects, list_size);
        // clang-format on

        Point3 const look_from { 0, 0.25, 3.25 };
        Point3 const look_at { 0, 0, -1 };
        Vec3 const view_up { 0, 1, 0 };

        float const vfov = 25.0f;
        float const aperture = 0.5f;
        float const focus_distance = (look_from - look_at).length();

        *d_camera = new Camera({
            look_from,
            look_at,
            view_up,
            vfov,
            aspect_ratio,
            aperture,
            focus_distance,
        });
    }
}

namespace _ {

    CudaDeviceFn inline Camera *rtiow_default_camera(float aspect_ratio) {
        Point3 const look_from { 0, 0.25, 3.25 };
        Point3 const look_at { 0, 0, -1 };
        Vec3 const view_up { 0, 1, 0 };

        float const vfov = 25.0f;
        float const aperture = 0.0f;
        float const focus_distance = (look_from - look_at).length();

        return new Camera({
            look_from,
            look_at,
            view_up,
            vfov,
            aspect_ratio,
            aperture,
            focus_distance,
        });
    }

    CudaDeviceFn inline HittableList *
    rtiow_single_sphere(Material *sphere_mat, Material *ground_mat) {
        int const list_size = 2;
        Hittable **objects = new Hittable *[list_size];
        objects[0] = new Sphere(Point3 { 0, 0, -1 }, 0.5, sphere_mat);
        objects[1] = new Sphere(Point3 { 0, -100.5, -1 }, 100, ground_mat);
        return new HittableList(objects, list_size);
    }

    CudaDeviceFn inline HittableList *rtiow_three_spheres(
        Material *left_mat,
        Material *right_mat,
        Material *center_mat,
        Material *ground_mat) {
        int const list_size = 4;
        Hittable **objects = new Hittable *[list_size];
        objects[0] = new Sphere(Point3 { -1, 0, -1 }, 0.5, left_mat);
        objects[1] = new Sphere(Point3 { 1, 0, -1 }, 0.5, right_mat);
        objects[2] = new Sphere(Point3 { 0, 0, -1 }, 0.5, center_mat);
        objects[3] = new Sphere(Point3 { 0, -100.5, -1 }, 100, ground_mat);
        return new HittableList(objects, list_size);
    }

    CudaDeviceFn inline HittableList *
    rtiow_metal_spheres(float left_sphere_fuzz, float right_sphere_fuzz) {
        return rtiow_three_spheres(
            new Metal(Rgb(0.8, 0.8, 0.8), left_sphere_fuzz),
            new Metal(Rgb(0.8, 0.6, 0.2), right_sphere_fuzz),
            new Lambertian(Rgb(0.7, 0.3, 0.3)),
            new Lambertian(Rgb(0.8, 0.8, 0)));
    }

} // namespace _

CudaGlobalFn void grey_sphere(
    HittableList **d_world, Camera **d_camera, float aspect_ratio, Rng *rand_state) {
    if (ThreadZero) {
        Unused(rand_state);

        *d_world = _::rtiow_single_sphere(
            new Lambertian(Rgb(0.5, 0.5, 0.5)), new Lambertian(Rgb(0.5, 0.5, 0.5)));

        *d_camera = _::rtiow_default_camera(aspect_ratio);
    }
}

CudaGlobalFn void shiny_metal_spheres(
    HittableList **d_world, Camera **d_camera, float aspect_ratio, Rng *rand_state) {
    if (ThreadZero) {
        Unused(rand_state);

        *d_world = _::rtiow_metal_spheres(0.0, 0.0);

        *d_camera = _::rtiow_default_camera(aspect_ratio);
    }
}

CudaGlobalFn void fuzzy_metal_spheres(
    HittableList **d_world, Camera **d_camera, float aspect_ratio, Rng *rand_state) {
    if (ThreadZero) {
        Unused(rand_state);

        *d_world = _::rtiow_metal_spheres(0.3, 1.0);

        *d_camera = _::rtiow_default_camera(aspect_ratio);
    }
}

CudaGlobalFn void two_glass_one_metal_spheres(
    HittableList **d_world, Camera **d_camera, float aspect_ratio, Rng *rand_state) {
    if (ThreadZero) {
        Unused(rand_state);

        *d_world = _::rtiow_three_spheres(
            new Dielectric(1.5),
            new Dielectric(1.5),
            new Metal(Rgb(0.8, 0.6, 0.2), 1.0),
            new Lambertian(Rgb(0.8, 0.8, 0)));

        *d_camera = _::rtiow_default_camera(aspect_ratio);
    }
}

CudaGlobalFn void glass_blue_metal_spheres(
    HittableList **d_world, Camera **d_camera, float aspect_ratio, Rng *rand_state) {
    if (ThreadZero) {
        Unused(rand_state);

        *d_world = _::rtiow_three_spheres(
            new Dielectric(1.5),
            new Lambertian(Rgb(0.1, 0.2, 0.5)),
            new Metal(Rgb(0.8, 0.6, 0.2), 0.0),
            new Lambertian(Rgb(0.8, 0.8, 0)));

        *d_camera = _::rtiow_default_camera(aspect_ratio);
    }
}

} // namespace Preset
