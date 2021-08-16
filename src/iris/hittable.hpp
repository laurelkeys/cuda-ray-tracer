#pragma once

#include "prelude.hpp"

#include "hit.hpp"
#include "linalg.hpp"
#include "ray.hpp"

namespace iris {
#define Fn CudaDeviceFn

// Interface for objects that can be intersected by rays (i.e. that are hittable).
class Hittable {
public:
#ifndef _PARALLEL
    virtual ~Hittable() = default;
#endif

    // If the ray hits the object, returns a record wrapping the object's surface normal
    // at the hit point, the hit point itself, and the value of t that parameterizes it.
    Fn virtual bool
    hit(Ray const &ray, HitInterval const &interval, Hit &maybe_hit) const = 0;
};

// A list of hittable objects.
class HittableList : public Hittable {
public:
    Hittable **list;
    int list_size;

    Fn HittableList() = delete;
    Fn HittableList(Hittable **list, int list_size)
        : list(list)
        , list_size(list_size) { }

    Fn virtual bool
    hit(const Ray &ray, HitInterval const &interval, Hit &maybe_hit) const;
};

Fn bool
HittableList::hit(const Ray &ray, HitInterval const &interval, Hit &maybe_hit) const {
    HitInterval hit_interval = interval;
    bool hit_anything = false;

    for (int obj = 0; obj < list_size; ++obj) {
        Hit closest_hit;
        if (list[obj]->hit(ray, hit_interval, closest_hit)) {
            hit_anything = true;
            hit_interval.t_max = closest_hit.t;
            maybe_hit = closest_hit;
        }
    }

    return hit_anything;
}

#undef Fn
} // namespace iris
