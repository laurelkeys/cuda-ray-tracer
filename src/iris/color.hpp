#pragma once

#include "prelude.hpp"

#include "linalg.hpp"
#include <iostream>

namespace iris {
#define Fn CudaHostFn CudaDeviceFn

class Rgb {
private:
    Vec3 color_;

    Fn constexpr Rgb(Vec3 const &inner_color)
        : color_(inner_color) {};

public:
    Fn constexpr Rgb()
        : color_ { 0, 0, 0 } {};
    Fn constexpr Rgb(float r, float g, float b)
        : color_ { r, g, b } {};

    Fn constexpr float r() const { return color_.x; };
    Fn constexpr float g() const { return color_.y; };
    Fn constexpr float b() const { return color_.z; };

    Fn constexpr Rgb &operator+=(Rgb const &other);
    Fn constexpr Rgb &operator*=(Rgb const &other);

    // @Note: using `friend` allows access to the private `color_`.
    Fn friend constexpr Rgb operator+(Rgb const &lhs, Rgb const &rhs);
    Fn friend constexpr Rgb operator-(Rgb const &lhs, Rgb const &rhs);
    Fn friend constexpr Rgb operator*(Rgb const &lhs, Rgb const &rhs);
    Fn friend constexpr Rgb operator*(float t, Rgb const &c);

    // Utility methods.
    Fn static constexpr Rgb black() { return Rgb { 0, 0, 0 }; }
    Fn static constexpr Rgb white() { return Rgb { 1, 1, 1 }; }

    Fn static constexpr Rgb from_vec(Vec3 const &v) { return Rgb(v.x, v.y, v.z); }

    Fn static inline Rgb gamma_2(Rgb const &color) {
        return Rgb(sqrt(color.r()), sqrt(color.g()), sqrt(color.b()));
    }
};

// clang-format off
Fn constexpr Rgb &Rgb::operator+=(Rgb const &other) { this->color_ += other.color_; return *this; }
Fn constexpr Rgb &Rgb::operator*=(Rgb const &other) { this->color_ *= other.color_; return *this; }

Fn constexpr Rgb operator+(Rgb const &lhs, Rgb const &rhs) { return Rgb(lhs.color_ + rhs.color_); }
Fn constexpr Rgb operator-(Rgb const &lhs, Rgb const &rhs) { return Rgb(lhs.color_ - rhs.color_); }
Fn constexpr Rgb operator*(Rgb const &lhs, Rgb const &rhs) { return Rgb(lhs.color_ * rhs.color_); }

Fn constexpr Rgb operator*(float t, Rgb const &c) { return Rgb(t * c.color_); }
Fn constexpr Rgb operator*(Rgb const &c, float t) { return t * c; }
Fn constexpr Rgb operator/(Rgb const &c, float t) { return (1.0f / t) * c; }
// clang-format on

namespace _ {

    class Rgb255 {
    private:
        constexpr u8 to255(float c) {
            return static_cast<u8>(256 * clamp(c, 0.0f, 0.9999f));
        }

    public:
        u8 r;
        u8 g;
        u8 b;

        Rgb255() = delete;
        constexpr Rgb255(Rgb const &from)
            : r(to255(from.r()))
            , g(to255(from.g()))
            , b(to255(from.b())) { }
    };

    std::ostream &operator<<(std::ostream &out, Rgb255 const &color) {
        // @Note: keep the unary `+`, otherwise we get ASCII values.
        out << +color.r << ' ' << +color.g << ' ' << +color.b;
        return out;
    }

} // namespace _

inline void write_color(std::ostream &out, Rgb const &linear_color) {
    // Convert the gamma 2.0 encoded pixel color to [0, 255].
    out << _::Rgb255(Rgb::gamma_2(linear_color)) << '\n';
}

#undef Fn
} // namespace iris
