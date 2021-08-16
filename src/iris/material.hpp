#pragma once

#include "prelude.hpp"

#include "color.hpp"
#include "hittable.hpp"
#include "linalg.hpp"
#include "random.hpp"
#include "ray.hpp"
#include "scattering.hpp"
#include "schlick.hpp"

namespace iris {
#define Fn CudaDeviceFn

// Linearly interpolates between the `zenith` and `horizon` colors,
// creating a color gradient in the skydome.
struct Sky {
    Rgb horizon; // Skydome's bottom color
    Rgb zenith; // Skydome's top color

    Fn Rgb color_at(Ray const &ray) const {
        float const t = 0.5f * (ray.direction.y() + 1.0f); // [-1, 1] -> [0, 1]

        // Blend the skydome colors based on the ray's y coordinate.
        return (1.0f - t) * horizon + t * zenith;
    }
};

struct Hit; // @Note: forward declaration

// Represents an object's material properties (i.e. how rays interact with the surface).
class Material {
public:
#ifndef _PARALLEL
    virtual ~Material() = default;
#endif

    struct ScatterResult {
        Rgb attenuation; // Amount by which the ray should be attenuated for each color
        Ray ray; // Scattered ray with the reflected / refracted direction of the light
    };

    // Returns a scattered ray when the incident ray isn't completely absorbed.
    Fn virtual bool scatter(
        Ray const &incident_ray,
        Hit const &hit,
        ScatterResult &maybe_scattered,
        Rng *rng) const = 0;
};

using MaybeScatter = Material::ScatterResult;

//
// Materials
//

class Lambertian : public Material {
public:
    Rgb albedo;

    Fn Lambertian() = delete;
    Fn Lambertian(Rgb const &albedo)
        : albedo(albedo) { }

    Fn bool scatter(
        Ray const &incident_ray,
        Hit const &hit,
        MaybeScatter &maybe_scattered,
        Rng *rng) const override;
};

class Metal : public Material {
public:
    Rgb albedo; // Amount of light energy reflected in each color component
    float fuzz; // Amount of randomness introduced into reflected rays, in range [0, 1].

    Fn Metal() = delete;
    Fn Metal(Rgb const &albedo, float fuzz)
        : albedo(albedo)
        , fuzz(saturate(fuzz)) { }

    Fn bool scatter(
        Ray const &incident_ray,
        Hit const &hit,
        MaybeScatter &maybe_scattered,
        Rng *rng) const override;
};

class Dielectric : public Material {
public:
    float ior; // Describes how fast light travels through the material

    Fn Dielectric() = delete;
    Fn Dielectric(float index_of_refraction)
        : ior(index_of_refraction) { }

    Fn bool scatter(
        Ray const &incident_ray,
        Hit const &hit,
        MaybeScatter &maybe_scattered,
        Rng *rng) const override;
};

//
// Material::scatter impl
//

Fn bool Lambertian::scatter(
    Ray const &incident_ray,
    Hit const &hit,
    MaybeScatter &maybe_scattered,
    Rng *rng) const {
    UnitVec3 const normal = hit.normal_against(incident_ray);
    UnitVec3 const random_vector = Random::vector_on_unit_sphere(rng);

    // Catch degeneracy when the random unit vector is exactly opposite to the normal.
    UnitVec3 const scatter_direction = UnitVec3::all_close(normal, -random_vector)
                                           ? normal
                                           : (normal + random_vector).into_unit();

    maybe_scattered.attenuation = albedo;
    maybe_scattered.ray = Ray { hit.point, scatter_direction };
    return true;
}

Fn bool Metal::scatter(
    Ray const &incident_ray,
    Hit const &hit,
    MaybeScatter &maybe_scattered,
    Rng *rng) const {
    UnitVec3 const normal = hit.normal_against(incident_ray);

    UnitVec3 const reflection_direction = (reflect(incident_ray.direction, normal)
                                           + fuzz * Random::vector_in_unit_sphere(rng))
                                              .into_unit();

    maybe_scattered.attenuation = albedo;
    maybe_scattered.ray = Ray { hit.point, reflection_direction };
    return true;
}

Fn bool Dielectric::scatter(
    Ray const &incident_ray,
    Hit const &hit,
    MaybeScatter &maybe_scattered,
    Rng *rng) const {
    UnitVec3 normal;
    float ni_over_nt;
    float cos_theta_i = dot(incident_ray.direction, hit.surface_normal);
    if (cos_theta_i < 0.0f) {
        // The ray is entering the material (ior == ηt).
        normal = +hit.surface_normal;
        ni_over_nt = 1.0f / ior;
        cos_theta_i = -cos_theta_i;
    } else {
        // The ray is exiting the material (ior == ηi).
        normal = -hit.surface_normal;
        ni_over_nt = ior / 1.0f;
        cos_theta_i = +cos_theta_i;
    }

    Vec3 refracted;
    bool reflect_ray = true;

    // Snell's law: sin(θt) = (ηi / ηt) * sin(θi)
    if (ni_over_nt * sqrt(1.0f - square(cos_theta_i)) <= 1.0f
        && refract(incident_ray.direction, normal, ni_over_nt, cos_theta_i, refracted)) {
        reflect_ray =
            Random::scalar(rng) < schlicks_fresnel_reflectance(ior, cos_theta_i);
    }

    maybe_scattered.attenuation = Rgb::white(); // absorb nothing
    maybe_scattered.ray = Ray {
        hit.point,
        reflect_ray ? reflect(incident_ray.direction, normal) : refracted.into_unit(),
    };
    return true;
}

#undef Fn
} // namespace iris
