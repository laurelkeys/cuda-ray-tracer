#include "iris/prelude.hpp"

#include "iris/error.hpp"
#include "iris/image.hpp"

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
    // std::cerr << "=== Running PARALLEL code. ===\n";
#else
    die(Error::WrongIrisExecutable); // expected to be compiling "parallel.exe"
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

    dim3 const threads { 8, 8, 1 };
    dim3 const blocks { DivUp(width, threads.x), DivUp(height, threads.y), 1 };

    //
    // Setup scene
    //

    // Allocate random state for generating the scene
    curandState *d_rand_state_scene;
    CUDACHECK(cudaMalloc((void **) &d_rand_state_scene, sizeof(curandState)));

    init_scene_rng<<<1, 1>>>(d_rand_state_scene);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    // Allocate random state for rendering
    curandState *d_rand_state;
    CUDACHECK(cudaMalloc((void **) &d_rand_state, width * height * sizeof(curandState)));

    init_render_rng<<<blocks, threads>>>(width, height, d_rand_state);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    HittableList **d_world;
    Camera **d_camera;
    CUDACHECK(cudaMalloc((void **) &d_world, sizeof(HittableList *)));
    CUDACHECK(cudaMalloc((void **) &d_camera, sizeof(Camera *)));

    // clang-format off
    {
        using namespace Preset;
        if      (preset_scene == 1) three_spheres               <<<1, 1>>>(d_world, d_camera, aspect.ratio(), d_rand_state_scene);
        else if (preset_scene == 2) grey_sphere                 <<<1, 1>>>(d_world, d_camera, aspect.ratio(), d_rand_state_scene);
        else if (preset_scene == 3) shiny_metal_spheres         <<<1, 1>>>(d_world, d_camera, aspect.ratio(), d_rand_state_scene);
        else if (preset_scene == 4) fuzzy_metal_spheres         <<<1, 1>>>(d_world, d_camera, aspect.ratio(), d_rand_state_scene);
        else if (preset_scene == 5) two_glass_one_metal_spheres <<<1, 1>>>(d_world, d_camera, aspect.ratio(), d_rand_state_scene);
        else if (preset_scene == 6) glass_blue_metal_spheres    <<<1, 1>>>(d_world, d_camera, aspect.ratio(), d_rand_state_scene);
        else                        rtiow_final_scene           <<<1, 1>>>(d_world, d_camera, aspect.ratio(), d_rand_state_scene);
    }
    // clang-format on
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    // Allocate the framebuffer
    Rgb *fb;
    CUDACHECK(cudaMallocManaged((void **) &fb, width * height * sizeof(Rgb)));

    //
    // Render scene
    //

    auto const render_start = std::chrono::system_clock::now();

    render<<<blocks, threads>>>(
        fb,
        width,
        height,
        samples_per_pixel,
        max_ray_bounces,
        d_camera,
        d_world,
        d_rand_state);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

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

    free_scene<<<1, 1>>>(d_world, d_camera);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaFree(fb));
    CUDACHECK(cudaFree(d_camera));
    CUDACHECK(cudaFree(d_world));
    CUDACHECK(cudaFree(d_rand_state));
    CUDACHECK(cudaFree(d_rand_state_scene));

    CUDACHECK(cudaDeviceReset());

    auto const program_end = std::chrono::system_clock::now();
    auto const program_duration =
        std::chrono::duration<float>(program_end - program_start).count();

    std::cerr << "Test Scene " << preset_scene << " (Parallel)"
              << "\nExecution Time (Total): " << program_duration << "s"
              << "\nExecution Time (Render): " << render_duration << "s\n";
}
