#pragma once

#include "prelude.hpp"

#include "math.hpp"

#include <iostream>
#include <math.h>
#include <stdlib.h>

namespace iris {
#define Fn CudaHostFn CudaDeviceFn

// clang-format off

// A wrapper used to denote that the underlying vector is normalized, i.e., that
// it has unit-length. Used to enforce this requirement in the type system itself.
template <typename V> struct Unit {
    V vec_; // @Todo: make this inner value private.

    Fn constexpr float x() const { return vec_.x; }
    Fn constexpr float y() const { return vec_.y; }
    Fn constexpr float z() const { return vec_.z; }

    Fn constexpr Unit<V> const& operator+() const { return *this; }
    Fn constexpr Unit<V> operator-() const { return Unit<V> { -vec_ }; }

    // Retrieves the inner vector.
    Fn constexpr V const& as_vec() const { return vec_; }
    Fn constexpr V into_vec() const { return vec_; }

    // Utility methods.
    Fn static inline Unit<V> from(const V& v) {
        assert(!approx_eq(v.length_squared(), 0.0f, NEAR_ZERO * NEAR_ZERO));
        return Unit<V> { v / v.length() };
    }

    Fn static constexpr bool all_close(Unit<V> const& lhs, Unit<V> const& rhs, float eps = Eps) {
        return V::all_close(lhs.vec_, rhs.vec_, eps);
    }
};

template <typename V> Fn constexpr V operator+(Unit<V> const& lhs, Unit<V> const& rhs) { return lhs.vec_ + rhs.vec_; }
template <typename V> Fn constexpr V operator+(     V  const& lhs, Unit<V> const& rhs) { return lhs      + rhs.vec_; }
template <typename V> Fn constexpr V operator+(Unit<V> const& lhs,      V  const& rhs) { return lhs.vec_ + rhs; }
template <typename V> Fn constexpr V operator-(Unit<V> const& lhs, Unit<V> const& rhs) { return lhs.vec_ - rhs.vec_; }
template <typename V> Fn constexpr V operator-(     V  const& lhs, Unit<V> const& rhs) { return lhs      - rhs.vec_; }
template <typename V> Fn constexpr V operator-(Unit<V> const& lhs,      V  const& rhs) { return lhs.vec_ - rhs; }
template <typename V> Fn constexpr V operator*(Unit<V> const& lhs, Unit<V> const& rhs) { return lhs.vec_ * rhs.vec_; }
template <typename V> Fn constexpr V operator*(     V  const& lhs, Unit<V> const& rhs) { return lhs      * rhs.vec_; }
template <typename V> Fn constexpr V operator*(Unit<V> const& lhs,      V  const& rhs) { return lhs.vec_ * rhs; }
template <typename V> Fn constexpr V operator/(Unit<V> const& lhs, Unit<V> const& rhs) { return lhs.vec_ / rhs.vec_; }
template <typename V> Fn constexpr V operator/(     V  const& lhs, Unit<V> const& rhs) { return lhs      / rhs.vec_; }
template <typename V> Fn constexpr V operator/(Unit<V> const& lhs,      V  const& rhs) { return lhs.vec_ / rhs; }

template <typename V> Fn constexpr V operator*(float t, Unit<V> const& v) { return t * v.vec_; }
template <typename V> Fn constexpr V operator*(Unit<V> const& v, float t) { return t * v; }
template <typename V> Fn constexpr V operator/(Unit<V> const& v, float t) { return (1.0f / t) * v; }

template <typename V> Fn constexpr float dot(Unit<V> const& lhs, Unit<V> const& rhs) { return dot(lhs.vec_, rhs.vec_); }
template <typename V> Fn constexpr float dot(     V  const& lhs, Unit<V> const& rhs) { return dot(lhs,      rhs.vec_); }
template <typename V> Fn constexpr float dot(Unit<V> const& lhs,      V  const& rhs) { return dot(lhs.vec_, rhs); }

// A 3D Euclidean vector.
struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Fn constexpr Vec3 const &operator+() const { return *this; }
    Fn constexpr Vec3 operator-() const { return Vec3 { -x, -y, -z }; }

    // This is witchcraft. Good luck if you decide to uncomment it.
    Fn inline float operator[](usize i) const { return reinterpret_cast<float const *>(this)[i]; }
    Fn inline float &operator[](usize i) { return reinterpret_cast<float *>(this)[i]; }

    Fn constexpr Vec3 &operator+=(Vec3 const &other);
    Fn constexpr Vec3 &operator-=(Vec3 const &other);

    // Element-wise multiplication/division.
    Fn constexpr Vec3 &operator*=(Vec3 const &other);
    Fn constexpr Vec3 &operator/=(Vec3 const &other);

    Fn constexpr Vec3 &operator*=(float t);
    Fn constexpr Vec3 &operator/=(float t);

    Fn constexpr float length_squared() const { return x * x + y * y + z * z; }
    Fn inline float length() const { return sqrt(length_squared()); }

    using UnitVec3 = Unit<Vec3>;
    // Normalizes the vector and then wraps it in `Unit`.
    Fn inline UnitVec3 into_unit() const { return UnitVec3::from(*this); }
    // Assumes the vector is normalized and wraps it in `Unit`.
    Fn constexpr UnitVec3 as_unit() const {
        assert(approx_eq(length_squared(), 1.0f, NEAR_ZERO_SQRT));
        // @Note: this is by no means safe, this function is simply
        // an unchecked (and, thus, unsafe) version of into_unit().
        return UnitVec3 { *this };
    }

    // Utility methods.
    Fn static constexpr UnitVec3 unit_x() { return UnitVec3 { 1, 0, 0 }; }
    Fn static constexpr UnitVec3 unit_y() { return UnitVec3 { 0, 1, 0 }; }
    Fn static constexpr UnitVec3 unit_z() { return UnitVec3 { 0, 0, 1 }; }

    Fn static constexpr Vec3 of(float val) { return Vec3 { val, val, val }; }

    Fn static constexpr bool all_close(Vec3 const& lhs, Vec3 const& rhs, float eps = Eps);

    // Smallest/largest component of a vector.
    Fn static inline float min(Vec3 const& v);
    Fn static inline float max(Vec3 const& v);

    // Component-wise min/max of two vectors.
    Fn static inline Vec3 min(Vec3 const& lhs, Vec3 const& rhs);
    Fn static inline Vec3 max(Vec3 const& lhs, Vec3 const& rhs);
};

using UnitVec3 = Unit<Vec3>;

Fn constexpr Vec3& Vec3::operator+=(Vec3 const& other) { x += other.x; y += other.y; z += other.z; return *this; }
Fn constexpr Vec3& Vec3::operator-=(Vec3 const& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
Fn constexpr Vec3& Vec3::operator*=(Vec3 const& other) { x *= other.x; y *= other.y; z *= other.z; return *this; }
Fn constexpr Vec3& Vec3::operator/=(Vec3 const& other) { x /= other.x; y /= other.y; z /= other.z; return *this; }
Fn constexpr Vec3& Vec3::operator*=(float t) { x *= t; y *= t; z *= t; return *this; }
Fn constexpr Vec3& Vec3::operator/=(float t) { *this *= 1.0f / t; return *this; }

Fn constexpr Vec3 operator+(Vec3 const& lhs, Vec3 const& rhs) { return Vec3 { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z }; }
Fn constexpr Vec3 operator-(Vec3 const& lhs, Vec3 const& rhs) { return Vec3 { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z }; }
Fn constexpr Vec3 operator*(Vec3 const& lhs, Vec3 const& rhs) { return Vec3 { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z }; }
Fn constexpr Vec3 operator/(Vec3 const& lhs, Vec3 const& rhs) { return Vec3 { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z }; }

Fn constexpr Vec3 operator*(float t, Vec3 const& v) { return Vec3 { v.x * t, v.y * t, v.z * t }; }
Fn constexpr Vec3 operator*(Vec3 const& v, float t) { return t * v; }
Fn constexpr Vec3 operator/(Vec3 const& v, float t) { return (1.0f / t) * v; }

Fn constexpr bool Vec3::all_close(Vec3 const& lhs, Vec3 const& rhs, float eps) {
    return approx_eq(lhs.x, rhs.x, eps) && approx_eq(lhs.y, rhs.y, eps) && approx_eq(lhs.z, rhs.z, eps);
}

Fn inline float Vec3::min(Vec3 const& v) { return fmin(v.x, fmin(v.y, v.z)); }
Fn inline float Vec3::max(Vec3 const& v) { return fmax(v.x, fmax(v.y, v.z)); }
Fn inline Vec3 Vec3::min(Vec3 const& lhs, Vec3 const& rhs) { return Vec3 { fmin(lhs.x, rhs.x), fmin(lhs.y, rhs.y), fmin(lhs.z, rhs.z) }; }
Fn inline Vec3 Vec3::max(Vec3 const& lhs, Vec3 const& rhs) { return Vec3 { fmax(lhs.x, rhs.x), fmax(lhs.y, rhs.y), fmax(lhs.z, rhs.z) }; }

Fn constexpr float dot(Vec3 const& lhs, Vec3 const& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Fn constexpr Vec3 cross(Vec3 const& lhs, Vec3 const& rhs) {
    return Vec3 {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

Fn constexpr Vec3 cross(Vec3 const& lhs, Unit<Vec3> const& rhs) { return cross(lhs, rhs.vec_); }
Fn constexpr Vec3 cross(Unit<Vec3> const& lhs, Vec3 const& rhs) { return cross(lhs.vec_, rhs); }
Fn constexpr Unit<Vec3> cross(Unit<Vec3> const& lhs, Unit<Vec3> const& rhs) {
    // @Safety: the cross product between two unit vectors is also a unit vector.
    return cross(lhs.vec_, rhs.vec_).as_unit();
}

// A point in 3D Euclidean space.
struct Point3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    // This is witchcraft. Good luck if you decide to uncomment it.
    /* Fn inline float operator[](usize i) const { return reinterpret_cast<float const*>(this)[i]; } */
    /* Fn inline float& operator[](usize i) { return reinterpret_cast<float*>(this)[i]; } */

    // Utility methods.
    Fn static constexpr Point3 from_vec(Vec3 const& v) { return Point3 { v.x, v.y, v.z }; }

    Fn static constexpr bool all_close(Point3 const& lhs, Point3 const& rhs, float eps = Eps);

    // Smallest/largest component of a point.
    Fn static inline float min(Point3 const& p);
    Fn static inline float max(Point3 const& p);

    // Component-wise min/max of two points.
    Fn static inline Point3 min(Point3 const& lhs, Point3 const& rhs);
    Fn static inline Point3 max(Point3 const& lhs, Point3 const& rhs);
};

Fn constexpr Point3 operator+(Point3 const& p, Vec3 const& v) { return Point3 { p.x + v.x, p.y + v.y, p.z + v.z }; }
Fn constexpr Point3 operator-(Point3 const& p, Vec3 const& v) { return Point3 { p.x - v.x, p.y - v.y, p.z - v.z }; }

Fn constexpr Vec3 operator-(Point3 const& lhs, Point3 const& rhs) { return Vec3 { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z }; }

Fn constexpr bool Point3::all_close(Point3 const& lhs, Point3 const& rhs, float eps) {
    return approx_eq(lhs.x, rhs.x, eps) && approx_eq(lhs.y, rhs.y, eps) && approx_eq(lhs.z, rhs.z, eps);
}

Fn inline float Point3::min(Point3 const& p) { return fmin(p.x, fmin(p.y, p.z)); }
Fn inline float Point3::max(Point3 const& p) { return fmax(p.x, fmax(p.y, p.z)); }
Fn inline Point3 Point3::min(Point3 const& lhs, Point3 const& rhs) { return Point3 { fmin(lhs.x, rhs.x), fmin(lhs.y, rhs.y), fmin(lhs.z, rhs.z) }; }
Fn inline Point3 Point3::max(Point3 const& lhs, Point3 const& rhs) { return Point3 { fmax(lhs.x, rhs.x), fmax(lhs.y, rhs.y), fmax(lhs.z, rhs.z) }; }

// clang-format on

#undef Fn
} // namespace iris
