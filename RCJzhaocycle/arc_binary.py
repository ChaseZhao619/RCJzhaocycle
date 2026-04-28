#!/usr/bin/env python3
"""Extract incomplete black circular arcs and write binary images.

The offline path only needs numpy, Pillow, and scipy. Camera input is optional
and uses OpenCV when it is installed.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class CircleCandidate:
    cx: float
    cy: float
    radius: float
    score: float
    coverage: int
    support: int


def otsu_threshold(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float32)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0

    hist, edges = np.histogram(data, bins=256)
    centers = (edges[:-1] + edges[1:]) * 0.5
    total = hist.sum()
    if total == 0:
        return float(data.mean())

    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1))[::-1]
    variance = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centers[int(np.argmax(variance))])


def disk_structure(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def image_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [input_path]


def to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def dark_candidate_mask(
    gray: np.ndarray,
    bg_sigma: float,
    min_contrast: float,
    min_area_ratio: float,
) -> np.ndarray:
    background = ndi.gaussian_filter(gray, sigma=bg_sigma)
    dark_response = background - gray
    positive = dark_response[dark_response > 0]
    threshold = max(min_contrast, otsu_threshold(positive) * 0.85 if positive.size else min_contrast)

    mask = dark_response > threshold
    mask &= gray < np.percentile(gray, 82)

    small = disk_structure(1)
    mask = ndi.binary_opening(mask, structure=small)
    mask = ndi.binary_closing(mask, structure=disk_structure(2))

    min_area = max(32, int(mask.size * min_area_ratio))
    return remove_small_objects(mask, min_area)


def downsample_mask(mask: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = mask.shape
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return mask.astype(bool), 1.0

    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    resized = img.resize(new_size, Image.Resampling.NEAREST)
    return np.asarray(resized) > 0, scale


def circle_from_points(points: np.ndarray) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = points.astype(np.float64)
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) * 0.5
    cd = (temp - x3 * x3 - y3 * y3) * 0.5
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-5:
        return None

    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    radius = math.hypot(cx - x1, cy - y1)
    if not all(np.isfinite([cx, cy, radius])):
        return None
    return cx, cy, radius


def sample_points(coords: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if coords.shape[0] <= max_points:
        return coords
    indexes = rng.choice(coords.shape[0], size=max_points, replace=False)
    return coords[indexes]


def score_circle(
    coords_xy: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    band: float,
    bins: int,
) -> tuple[float, int, int]:
    dx = coords_xy[:, 0] - cx
    dy = coords_xy[:, 1] - cy
    distances = np.hypot(dx, dy)
    close = np.abs(distances - radius) <= band
    support = int(close.sum())
    if support == 0:
        return 0.0, 0, 0

    angles = np.arctan2(dy[close], dx[close])
    angle_bins = ((angles + math.pi) / (2 * math.pi) * bins).astype(np.int32)
    coverage = int(np.unique(np.clip(angle_bins, 0, bins - 1)).size)
    interior = distances < max(0.0, radius - 2.5 * band)
    interior_support = int(interior.sum())
    interior_penalty = 1.0 + 1.15 * interior_support / max(support, 1)
    score = support * (1.0 + coverage / bins) / interior_penalty
    return float(score), coverage, support


def find_arc_circle(
    mask: np.ndarray,
    max_side: int,
    samples: int,
    iterations: int,
    min_radius_ratio: float,
    max_radius_ratio: float,
    random_seed: int,
) -> tuple[CircleCandidate | None, float]:
    small_mask, scale = downsample_mask(mask, max_side)
    edge = small_mask ^ ndi.binary_erosion(small_mask, structure=disk_structure(1))
    coords_yx = np.argwhere(edge)
    if coords_yx.shape[0] < 12:
        return None, scale

    rng = np.random.default_rng(random_seed)
    coords_yx = sample_points(coords_yx, samples, rng)
    coords_xy = coords_yx[:, ::-1].astype(np.float32)

    h, w = small_mask.shape
    side = max(h, w)
    min_radius = side * min_radius_ratio
    max_radius = side * max_radius_ratio
    band = max(3.0, side * 0.012)
    bins = 96

    best: CircleCandidate | None = None
    n = coords_xy.shape[0]
    for _ in range(iterations):
        picked = coords_xy[rng.choice(n, size=3, replace=False)]
        circle = circle_from_points(picked)
        if circle is None:
            continue
        cx, cy, radius = circle
        if radius < min_radius or radius > max_radius:
            continue
        if cx < -w or cx > 2 * w or cy < -h or cy > 2 * h:
            continue

        score, coverage, support = score_circle(coords_xy, cx, cy, radius, band, bins)
        if coverage < 6 or support < 20:
            continue
        if best is None or score > best.score:
            best = CircleCandidate(cx, cy, radius, score, coverage, support)

    return best, scale


def choose_radial_band(mask: np.ndarray, circle: CircleCandidate, scale: float) -> tuple[float, float]:
    cx = circle.cx / scale
    cy = circle.cy / scale
    base_radius = circle.radius / scale
    coords_yx = np.argwhere(mask)
    distances = np.hypot(coords_yx[:, 1] - cx, coords_yx[:, 0] - cy)
    near = distances[np.abs(distances - base_radius) <= max(30.0, base_radius * 0.12)]
    if near.size < 20:
        return base_radius - 8.0, base_radius + 8.0

    bin_width = max(2.0, min(mask.shape) / 450.0)
    low = max(0.0, float(near.min()) - bin_width)
    high = float(near.max()) + bin_width
    hist, edges = np.histogram(near, bins=max(12, int((high - low) / bin_width)), range=(low, high))
    hist = ndi.gaussian_filter1d(hist.astype(np.float32), sigma=1.2)
    peak = int(np.argmax(hist))
    peak_value = hist[peak]
    cutoff = max(peak_value * 0.28, np.percentile(hist, 65))

    left = peak
    while left > 0 and hist[left - 1] >= cutoff:
        left -= 1
    right = peak
    while right < hist.size - 1 and hist[right + 1] >= cutoff:
        right += 1

    r_inner = float(edges[left])
    r_outer = float(edges[right + 1])
    min_width = max(6.0, min(mask.shape) / 160.0)
    max_width = max(20.0, min(mask.shape) / 18.0)
    width = r_outer - r_inner
    if width < min_width:
        center = (r_inner + r_outer) * 0.5
        r_inner = center - min_width * 0.5
        r_outer = center + min_width * 0.5
    elif width > max_width:
        center = (r_inner + r_outer) * 0.5
        r_inner = center - max_width * 0.5
        r_outer = center + max_width * 0.5
    return max(0.0, r_inner), r_outer


def keep_supported_angles(
    mask: np.ndarray,
    circle: CircleCandidate,
    scale: float,
    radial_band: tuple[float, float],
    bins: int = 144,
) -> np.ndarray:
    cx = circle.cx / scale
    cy = circle.cy / scale
    yy, xx = np.indices(mask.shape)
    distances = np.hypot(xx - cx, yy - cy)
    ring = mask & (distances >= radial_band[0]) & (distances <= radial_band[1])

    coords_yx = np.argwhere(ring)
    if coords_yx.shape[0] == 0:
        return ring

    angles = np.arctan2(coords_yx[:, 0] - cy, coords_yx[:, 1] - cx)
    angle_bins = ((angles + math.pi) / (2 * math.pi) * bins).astype(np.int32)
    angle_bins = np.clip(angle_bins, 0, bins - 1)
    counts = np.bincount(angle_bins, minlength=bins)
    smoothed = ndi.gaussian_filter1d(counts.astype(np.float32), sigma=1.5, mode="wrap")
    threshold = max(3.0, min(np.percentile(smoothed[smoothed > 0], 35) if np.any(smoothed > 0) else 3.0, smoothed.max() * 0.18))
    good_bins = smoothed >= threshold
    good_bins = ndi.binary_dilation(good_bins, structure=np.ones(5, dtype=bool))

    all_angles = np.arctan2(yy - cy, xx - cx)
    all_bins = ((all_angles + math.pi) / (2 * math.pi) * bins).astype(np.int32)
    all_bins = np.clip(all_bins, 0, bins - 1)
    return ring & good_bins[all_bins]


def extract_arc_binary(
    image: Image.Image,
    bg_sigma: float | None,
    min_contrast: float,
    min_area_ratio: float,
    hough_side: int,
    samples: int,
    iterations: int,
    min_radius_ratio: float,
    max_radius_ratio: float,
    random_seed: int,
) -> np.ndarray:
    gray = to_gray(image)
    h, w = gray.shape
    sigma = bg_sigma if bg_sigma is not None else max(12.0, min(h, w) / 28.0)
    candidates = dark_candidate_mask(gray, sigma, min_contrast, min_area_ratio)
    circle, scale = find_arc_circle(
        candidates,
        max_side=hough_side,
        samples=samples,
        iterations=iterations,
        min_radius_ratio=min_radius_ratio,
        max_radius_ratio=max_radius_ratio,
        random_seed=random_seed,
    )

    if circle is None:
        return (candidates.astype(np.uint8) * 255)

    radial_band = choose_radial_band(candidates, circle, scale)
    result = keep_supported_angles(candidates, circle, scale, radial_band)
    result = ndi.binary_closing(result, structure=disk_structure(2))
    result = remove_small_objects(result, max(16, int(result.size * min_area_ratio * 0.5)))
    return (result.astype(np.uint8) * 255)


def save_binary(binary: np.ndarray, output_path: Path, invert: bool) -> None:
    if invert:
        binary = 255 - binary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode="L").save(output_path)


def process_files(paths: Iterable[Path], args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    for path in paths:
        image = Image.open(path).convert("RGB")
        binary = extract_arc_binary(
            image=image,
            bg_sigma=args.bg_sigma,
            min_contrast=args.min_contrast,
            min_area_ratio=args.min_area_ratio,
            hough_side=args.hough_side,
            samples=args.samples,
            iterations=args.iterations,
            min_radius_ratio=args.min_radius_ratio,
            max_radius_ratio=args.max_radius_ratio,
            random_seed=args.seed,
        )
        save_binary(binary, output_dir / f"{path.stem}_binary.png", args.invert)
        print(f"wrote {output_dir / f'{path.stem}_binary.png'}")


def run_camera(args: argparse.Namespace) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("camera mode requires opencv-python: pip install opencv-python") from exc

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"failed to open camera index {args.camera}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            binary = extract_arc_binary(
                image=image,
                bg_sigma=args.bg_sigma,
                min_contrast=args.min_contrast,
                min_area_ratio=args.min_area_ratio,
                hough_side=args.hough_side,
                samples=args.samples,
                iterations=args.iterations,
                min_radius_ratio=args.min_radius_ratio,
                max_radius_ratio=args.max_radius_ratio,
                random_seed=args.seed,
            )
            shown = 255 - binary if args.invert else binary
            cv2.imshow("arc binary", shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out = output_dir / f"camera_{frame_index:04d}_binary.png"
                save_binary(binary, out, args.invert)
                print(f"wrote {out}")
                frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract incomplete black circular arcs as binary images.")
    parser.add_argument("-i", "--input", default="pic", help="image file or directory for offline processing")
    parser.add_argument("-o", "--output", default="out", help="output directory")
    parser.add_argument("--camera", type=int, help="camera index; when set, process live camera frames")
    parser.add_argument("--invert", action="store_true", help="write black foreground on white background")
    parser.add_argument("--bg-sigma", type=float, help="background blur sigma; default is based on image size")
    parser.add_argument("--min-contrast", type=float, default=16.0, help="minimum local dark contrast")
    parser.add_argument("--min-area-ratio", type=float, default=0.00002, help="remove blobs smaller than this image-area ratio")
    parser.add_argument("--hough-side", type=int, default=720, help="max side length used by circle search")
    parser.add_argument("--samples", type=int, default=4500, help="max edge samples used by circle search")
    parser.add_argument("--iterations", type=int, default=9000, help="RANSAC circle iterations")
    parser.add_argument("--min-radius-ratio", type=float, default=0.08, help="min circle radius divided by max image side")
    parser.add_argument("--max-radius-ratio", type=float, default=1.6, help="max circle radius divided by max image side")
    parser.add_argument("--seed", type=int, default=7, help="random seed for repeatable circle search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.camera is not None:
        run_camera(args)
        return

    paths = image_paths(Path(args.input))
    if not paths:
        raise SystemExit(f"no images found in {args.input}")
    process_files(paths, args)


if __name__ == "__main__":
    main()
