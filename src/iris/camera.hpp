#pragma once

#include "prelude.hpp"

#include "linalg.hpp"
#include "math.hpp"
#include "random.hpp"
#include "ray.hpp"

namespace iris {
#define Fn CudaDeviceFn

struct CameraSpecs {
    Point3 look_from;
    Point3 look_at;
    Vec3 view_up;
    float vfov; // Vertical field-of-view in degrees
    float aspect_ratio;
    float aperture;
    float focus_distance;
};

class Camera {
public:
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    struct OrthonormalBasis {
        UnitVec3 u;
        UnitVec3 v;
        UnitVec3 w;
    } basis;
    float lens_radius;

    Fn Camera(CameraSpecs const &camera) {
        float const theta = deg2rad(camera.vfov);
        float const h = tan(theta / 2.0f);
        float const viewport_height = 2.0f * h;
        float const viewport_width = viewport_height * camera.aspect_ratio;

        // Compute an orthonormal basis (u,v,w) to describe the camera's orientation.
        UnitVec3 const w = (camera.look_from - camera.look_at).into_unit();
        UnitVec3 const u = cross(camera.view_up, w).into_unit();
        UnitVec3 const v = cross(w, u);
        basis = OrthonormalBasis { u, v, w };

        origin = camera.look_from;
        horizontal = camera.focus_distance * viewport_width * u;
        vertical = camera.focus_distance * viewport_height * v;
        lower_left_corner =
            origin - horizontal / 2.0f - vertical / 2.0f - camera.focus_distance * w;

        lens_radius = camera.aperture / 2.0f;
    }

    Fn Ray ray(float u, float v, Rng *rng) const {
        // Fake defocus blur (depth of field).
        Vec3 const rd = lens_radius * Random::vector_on_unit_sphere(rng);
        Vec3 const offset = basis.u * rd.x + basis.v * rd.y;
        Point3 const offset_origin = origin + offset;

        Vec3 const direction =
            lower_left_corner + u * horizontal + v * vertical - offset_origin;

        return Ray { offset_origin, direction.into_unit() };
    }
};

#undef Fn
} // namespace iris
