#pragma once

#include "prelude.hpp"

#include <iostream>

namespace iris {

struct AspectRatio {
    int x;
    int y;

    // Note: `height * (x / y) = width`.
    constexpr float ratio() const { return x / static_cast<float>(y); }
    // Note: `width * (y / x) = height`.
    constexpr float ratio_rcp() const { return y / static_cast<float>(x); }
};

struct Resolution {
    int width;
    int height;

    // Computes the `height` that matches the given aspect ratio `x / y`.
    static constexpr Resolution with_width(int width, AspectRatio const &aspect) {
        return Resolution { width, static_cast<int>(width * aspect.ratio_rcp()) };
    }

    // Computes the `width` that matches the given aspect ratio `x / y`.
    static constexpr Resolution with_height(int height, AspectRatio const &aspect) {
        return Resolution { static_cast<int>(height * aspect.ratio()), height };
    }
};

} // namespace iris
