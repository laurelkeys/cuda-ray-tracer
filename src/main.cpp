#include "iris/prelude.hpp"

#include "iris/error.hpp"
#include "iris/image.hpp"
#include "iris/rng.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

#include "render.hpp"
#include "scenes.hpp"

using namespace iris;

int main(int argc, char *argv[]) {
    auto const program_start = std::chrono::system_clock::now();

    auto const die = [program = argv[0]](Error error = Error::None) {
        if (error != Error::None) {
            std::cerr << "Usage: " << program << " <test scene> [<output>]\n";
        }
        exit(static_cast<int>(error));
    };

#ifdef _PARALLEL
    die(Error::WrongIrisExecutable); // expected to be compiling "serial.exe"
#else
    // std::cerr << "=== Running SERIAL code. ===\n";
#endif

    //
    // Parse arguments
    //

    if (argc == 1) { die(Error::MissingArguments); }

    std::ofstream output_file;
    if (argc == 3) { output_file.open(argv[2]); }
    auto &out = argc == 3 ? output_file : std::cout;

    int const preset_scene = *argv[1] - '0';
    if (preset_scene < 0 || preset_scene > 6) { die(Error::InvalidArgumentValue); }

    //
    // Set render settings
    //

    int const samples_per_pixel = 16;
    int const max_ray_bounces = 8;

    auto const aspect = AspectRatio { 16, 9 };
    auto const resolution = Resolution::with_height(720, aspect);
    auto const width = resolution.width;
    auto const height = resolution.height;

    //
    // Setup scene
    //

#ifdef USE_PBRT_RNG
    Rng rng(RNG_SEED);
#else
    Rng rng = 0;
#endif

    HittableList **d_world = static_cast<HittableList **>(malloc(sizeof(HittableList *)));
    Camera **d_camera = static_cast<Camera **>(malloc(sizeof(Camera *)));

    // clang-format off
    {
        using namespace Preset;
        if      (preset_scene == 1) three_spheres               (d_world, d_camera, aspect.ratio(), &rng);
        else if (preset_scene == 2) grey_sphere                 (d_world, d_camera, aspect.ratio(), &rng);
        else if (preset_scene == 3) shiny_metal_spheres         (d_world, d_camera, aspect.ratio(), &rng);
        else if (preset_scene == 4) fuzzy_metal_spheres         (d_world, d_camera, aspect.ratio(), &rng);
        else if (preset_scene == 5) two_glass_one_metal_spheres (d_world, d_camera, aspect.ratio(), &rng);
        else if (preset_scene == 6) glass_blue_metal_spheres    (d_world, d_camera, aspect.ratio(), &rng);
        else                        rtiow_final_scene           (d_world, d_camera, aspect.ratio(), &rng);
    }
    // clang-format on

    // Allocate the framebuffer
    Rgb *fb = static_cast<Rgb *>(malloc(width * height * sizeof(Rgb)));

    //
    // Render scene
    //

    auto const render_start = std::chrono::system_clock::now();

    render(
        fb, width, height, samples_per_pixel, max_ray_bounces, d_camera, d_world, &rng);

    auto const render_end = std::chrono::system_clock::now();

    auto const render_duration =
        std::chrono::duration<float>(render_end - render_start).count();

    //
    // Output in PPM format
    //

    out << "P3\n" << width << ' ' << height << "\n255\n";
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) { write_color(out, fb[y * width + x]); }
    }

    //
    // Free allocated memory
    //

    free_scene(d_world, d_camera);

    free(fb);
    free(d_camera);
    free(d_world);

    auto const program_end = std::chrono::system_clock::now();
    auto const program_duration =
        std::chrono::duration<float>(program_end - program_start).count();

    std::cerr << "Test Scene " << preset_scene << " (Serial)"
              << "\nExecution Time (Total): " << program_duration << "s"
              << "\nExecution Time (Render): " << render_duration << "s\n";
}
