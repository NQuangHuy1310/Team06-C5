"""Apply blur, solid masks, or pixelation to bounding boxes in an image.

Bounding boxes use pixel coordinates in ``(x1, y1, x2, y2)`` format, where
``x2`` and ``y2`` are exclusive (the usual NumPy slicing convention).

Requires: pip install opencv-python
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

BBox = Sequence[int]


def _crop_bounds(image: np.ndarray, bbox: BBox) -> tuple[int, int, int, int] | None:
    """Clip a bbox to image boundaries; return None if it has no area."""
    if image is None or image.ndim < 2:
        raise ValueError("image must be a valid image array")
    if len(bbox) != 4:
        raise ValueError("bbox must contain (x1, y1, x2, y2)")

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def blur_bbox(image: np.ndarray, bbox: BBox, strength: int = 31) -> np.ndarray:
    """Return a copy of image with the bbox blurred.

    ``strength`` controls Gaussian kernel size; it is rounded up to an odd
    number. Higher values generally produce stronger blur.
    """
    result = image.copy()
    bounds = _crop_bounds(result, bbox)
    if bounds is None:
        return result
    x1, y1, x2, y2 = bounds
    kernel = max(1, int(strength))
    if kernel % 2 == 0:
        kernel += 1
    result[y1:y2, x1:x2] = cv2.GaussianBlur(result[y1:y2, x1:x2], (kernel, kernel), 0)
    return result


def mask_bbox(
    image: np.ndarray,
    bbox: BBox,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Return a copy with the bbox completely covered by a solid color.

    For OpenCV color images, ``color`` is in BGR order. Use a grayscale value
    (for example ``0``) when masking a grayscale image.
    """
    result = image.copy()
    bounds = _crop_bounds(result, bbox)
    if bounds is None:
        return result
    x1, y1, x2, y2 = bounds
    result[y1:y2, x1:x2] = color
    return result


def pixelate_bbox(image: np.ndarray, bbox: BBox, pixel_size: int = 12) -> np.ndarray:
    """Return a copy with the bbox pixelated; larger ``pixel_size`` is coarser."""
    if pixel_size < 1:
        raise ValueError("pixel_size must be at least 1")
    result = image.copy()
    bounds = _crop_bounds(result, bbox)
    if bounds is None:
        return result
    x1, y1, x2, y2 = bounds
    region = result[y1:y2, x1:x2]
    small_width = max(1, (x2 - x1) // pixel_size)
    small_height = max(1, (y2 - y1) // pixel_size)
    tiny = cv2.resize(
        region, (small_width, small_height), interpolation=cv2.INTER_LINEAR
    )
    result[y1:y2, x1:x2] = cv2.resize(
        tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
    )
    return result


def apply_to_bboxes(
    image: np.ndarray,
    bboxes: Iterable[BBox],
    method: str = "blur",
    **options: object,
) -> np.ndarray:
    """Apply one effect to all bboxes and return the processed image.

    ``method`` is ``"blur"``, ``"mask"``, or ``"pixelation"``. Options are
    forwarded to the corresponding function (``strength``, ``color``, or
    ``pixel_size``).
    """
    effects = {
        "blur": blur_bbox,
        "mask": mask_bbox,
        "pixelation": pixelate_bbox,
    }
    try:
        effect = effects[method]
    except KeyError as error:
        raise ValueError(
            f"Unsupported method {method!r}; choose from {tuple(effects)}"
        ) from error

    result = image.copy()
    for bbox in bboxes:
        result = effect(result, bbox, **options)
    return result


def process_image_file(
    input_path: str,
    output_path: str,
    bboxes: Iterable[BBox],
    method: str = "blur",
    **options: object,
) -> None:
    """Read an image, apply an effect to bboxes, and save the result."""
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    result = apply_to_bboxes(image, bboxes, method, **options)
    if not cv2.imwrite(output_path, result):
        raise OSError(f"Could not write image: {output_path}")


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    bboxes_by_image: Mapping[str, Iterable[BBox]] | None = None,
) -> None:
    """Apply blur, solid mask, and pixelation to detected boxes.

    ``bboxes_by_image`` maps image filenames (or filename stems) to detected
    boxes. Images without a matching entry are copied unmodified. Existing
    output files with matching names are overwritten.
    """
    source = Path(input_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {source}")

    boxes_by_name = bboxes_by_image or {}
    methods = ("blur", "mask", "pixelation")
    output_dirs = {}
    for method in methods:
        method_dir = destination / method
        method_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[method] = method_dir

    # Read each source image once, then produce all three variants.
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Could not read image: {image_path}")
        bboxes = boxes_by_name.get(
            image_path.name, boxes_by_name.get(image_path.stem, ())
        )
        for method in methods:
            options = {"strength": 41} if method == "blur" else {}
            result = apply_to_bboxes(image, bboxes, method=method, **options)
            output_path = output_dirs[method] / image_path.name
            if not cv2.imwrite(str(output_path), result):
                raise OSError(f"Could not write image: {output_path}")


def load_detection_boxes(json_path: str | Path) -> dict[str, list[BBox]]:
    """Load YOLO detection boxes from the JSON emitted by ``detect.py``."""
    with Path(json_path).open(encoding="utf-8") as file:
        detections = json.load(file)

    boxes_by_image: dict[str, list[BBox]] = {}
    for item in detections:
        image_name = item["image"]
        boxes_by_image[image_name] = [
            detection["box_xyxy"] for detection in item.get("detections", [])
        ]
    return boxes_by_image


if __name__ == "__main__":
    detections_path = Path("detection_results.json")
    boxes_by_image = (
        load_detection_boxes(detections_path) if detections_path.exists() else {}
    )
    process_directory(
        input_dir="detect_images",
        output_dir="results",
        bboxes_by_image=boxes_by_image,
    )
    print("Done. Results saved in results/blur, results/mask, results/pixelation")
