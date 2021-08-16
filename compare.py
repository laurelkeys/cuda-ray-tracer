import os
import math
import argparse


# See: https://en.wikipedia.org/wiki/Luma_(video)#Use_of_relative_luminance
luma = lambda rgb: 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

# For expected values of MSE and PSNR, see:
# - https://www.inf.ufrgs.br/~oliveira/pubs_files/SC/Scene_Conversion_for_Physically-based_Renderers-SIBGRAPI_2018.pdf
# - https://www.inf.ufrgs.br/~oliveira/projects/FBKSD/Brito_Sen_Oliveira_pre-print.pdf
# - https://github.com/fbksd/fbksd/tree/master/src/iqa
PSNR_THRESHOLD = 25.0  # account for difference in random number generator


def read_ppm(file_path: str, post_process=None):
    with open(file_path, "r") as ppm:
        ppm_lines = [
            line for line in ppm.read().splitlines() if not line.startswith("#")
        ]

        assert ppm_lines[0] == "P3"
        width, height = [int(dim) for dim in ppm_lines[1].split(" ")]
        assert ppm_lines[2] == "255"

        img = []
        row = []
        for line in ppm_lines[3:]:
            rgb = [int(c) for c in line.rstrip("\n").split(" ")]
            row.append(rgb if post_process is None else post_process(rgb))
            if len(row) == width:
                img.append(row)
                row = []
        if row:
            assert len(row) == width
            img.append(row)

        assert len(img) == height
        return img


def mse(img, ref) -> float:
    assert len(img) == len(ref)
    assert len(img[0]) == len(ref[0])
    assert not isinstance(img[0][0], list)
    assert not isinstance(ref[0][0], list)

    height, width = len(img), len(img[0])
    squared_error = []
    for img_row, ref_row in zip(img, ref):
        squared_error.append(
            [(img_px - ref_px) ** 2 for img_px, ref_px in zip(img_row, ref_row)]
        )

    return sum([sum(row) for row in squared_error]) / float(width * height)


def psnr(mse_value: float, max_pixel_value: float = 255) -> float:
    if mse_value == 0:
        return math.inf  # avoid runtime warning
    else:
        return 10 * math.log10((max_pixel_value * max_pixel_value) / mse_value)


def main(scene: int, img_path: str, ref_path: str) -> None:
    img_luma = read_ppm(img_path, lambda rgb: luma(rgb))
    ref_luma = read_ppm(ref_path, lambda rgb: luma(rgb))

    mse_luma = mse(img_luma, ref_luma)
    psnr_luma = psnr(mse_luma)

    # NOTE for MSE, the smaller the better (with 0.0 for identical images),
    # but for PSNR, the greater the better (with inf for identical images).
    print(f"Test {scene}: MSE = {mse_luma:.2f}, PSNR = {psnr_luma:.2f}")

    exit(1 if psnr_luma < PSNR_THRESHOLD else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a rendered image to a reference."
    )

    parser.add_argument("scene", type=int, help="Number of the test scene")
    parser.add_argument("render", type=str, help="Path to the rendered image")
    parser.add_argument("reference", type=str, help="Path to the reference image")

    args = parser.parse_args()

    assert os.path.isfile(args.render)
    assert os.path.isfile(args.reference)
    main(args.scene, args.render, args.reference)
