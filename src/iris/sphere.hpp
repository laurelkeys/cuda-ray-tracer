#pragma once

#include "prelude.hpp"

#include "hittable.hpp"
#include "linalg.hpp"
#include "ray.hpp"

namespace iris {
#define Fn CudaDeviceFn

class Sphere : public Hittable {
public:
    Point3 center;
    float radius;
    Material *material;

    Fn Sphere() = delete;
    Fn Sphere(Point3 const &center, float radius, Material *material)
        : center(center)
        , radius(radius)
        , material(material) { }

    Fn virtual bool
    hit(Ray const &ray, HitInterval const &interval, Hit &maybe_hit) const;
};

Fn bool Sphere::hit(Ray const &ray, HitInterval const &interval, Hit &maybe_hit) const {
    // Consider a sphere with center C and radius r. Any point P that satisfies
    // the following equation is on the surface of such sphere:
    //   <P - C, P - C> = r*r,
    //   i.e. the squared length of the vector P - C is equal to r*r
    //
    // Thus, for a ray to hit the sphere, there must be a point P(t) = A + t*b along
    // the ray (with origin A and direction b) for which the equation is true:
    //   <P(t) - C, P(t) - C> = r*r
    //   => <A + t*b - C, A + t*b - C> = r*r
    //   => t*t*<b, b> + 2*t*<b, A - C> + <A - C, A - C> - r*r = 0
    //
    // We can now directly solve for t, as this is a quadratic equation, but since
    // we're only interested in whether or not the sphere is hit, we can simply look
    // at the number of solutions (0, 1, 2) by computing the sign of the discriminant:
    //   t*t*a' + t*b' + c' = 0, where:
    //   | a' = <b, b>
    //   | b' = 2*<b, A - C>
    //   | c' = <A - C, A - C> - r*r

    // Note: since b' has a factor of 2 in it, we can simplify the solution of the
    // quadratic equation above in terms of h' = b' / 2:
    //   (-b' +- sqrt(b'*b' - 4*a'*c')) / (2*a')
    //   => (-2*h' +- sqrt(2*h'*2*h' - 4*a'*c')) / (2*a')
    //   => (-2*h' +- sqrt(4*(h'*h' - a'*c'))) / (2*a')
    //   => (-2*h' +- 2*sqrt(h'*h' - a'*c')) / (2*a')
    //   => (-h' +- sqrt(h'*h' - a'*c')) / a'
    //
    // where h'*h' - a'*c' is the "half" discriminant.

    Vec3 const center_to_ray_origin = ray.origin - center; // A - C

    float const a = dot(ray.direction, ray.direction); // <b, b>
    float const h = dot(ray.direction, center_to_ray_origin); // <b, A - C>
    float const c = dot(center_to_ray_origin, center_to_ray_origin) // <A - C, A - C>
                    - radius * radius; // - r*r

    float const discriminant = h * h - a * c;

    if (discriminant >= 0.0f) {
        float const sqrt_discriminant = sqrt(discriminant);

        // Check if one of the roots lies within the valid range for t.
        float const t_minus = (-h - sqrt_discriminant) / a;
        float const t_plus = (-h + sqrt_discriminant) / a;

        // @Note: the surface normal point outwards (by convention).
        if (interval.contains(t_minus)) {
            maybe_hit.t = t_minus;
            maybe_hit.point = ray.at(t_minus);
            maybe_hit.surface_normal = ((maybe_hit.point - center) / radius).into_unit();
            maybe_hit.material = material;
            return true;
        } else if (interval.contains(t_plus)) {
            maybe_hit.t = t_plus;
            maybe_hit.point = ray.at(t_plus);
            maybe_hit.surface_normal = ((maybe_hit.point - center) / radius).into_unit();
            maybe_hit.material = material;
            return true;
        }
    }

    return false; // the ray doesn't intersect the sphere
}

#undef Fn
} // namespace iris
