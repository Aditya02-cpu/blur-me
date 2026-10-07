"""Pure image-blurring helpers shared by the Streamlit front-end.

Every function here works on plain RGB ``numpy`` arrays of shape
``(H, W, 3)`` and dtype ``uint8``, so the module can be exercised in a
unit test without launching the app.
"""

from __future__ import annotations

import io
import pathlib
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

#: Region modes accepted by :func:`apply_region`.
REGION_MODES = ("whole", "circle", "inverse")

BASE_DIR = pathlib.Path(__file__).resolve().parent
FACE_MODEL = BASE_DIR / "face_detection_yunet_2023mar.onnx"
DEFAULT_CONFIDENCE = 0.5

#: Box = (left, top, right, bottom) in pixels.
Box = Tuple[int, int, int, int]


def _odd(value: int) -> int:
    """Return the largest odd number not above ``value`` (minimum 1)."""
    size = max(1, int(value))
    return size if size % 2 else size - 1


def _kernel(strength: int, maximum: int = 61) -> int:
    """Map a 0-100 strength onto an odd kernel between 1 and ``maximum``."""
    if strength <= 0:
        return 1
    return _odd(1 + int(round(strength / 100 * (maximum - 1))))


def gaussian(img: np.ndarray, strength: int) -> np.ndarray:
    """Gaussian blur with a strength-derived kernel size."""
    kernel = _kernel(strength)
    if kernel < 3:
        return img.copy()
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def box(img: np.ndarray, strength: int) -> np.ndarray:
    """Fast normalised box blur."""
    kernel = _kernel(strength)
    if kernel < 2:
        return img.copy()
    return cv2.blur(img, (kernel, kernel))


def median(img: np.ndarray, strength: int) -> np.ndarray:
    """Median blur - great for hiding detail while keeping edges crisp."""
    return cv2.medianBlur(img, max(3, _kernel(strength)))


def pixelate(img: np.ndarray, strength: int) -> np.ndarray:
    """Mosaic/pixelation by down-scaling then up-scaling with nearest-neighbour."""
    block = 2 + int(round(strength / 100 * 48))
    if strength <= 0:
        return img.copy()

    height, width = img.shape[:2]
    small = cv2.resize(
        img,
        (max(1, width // block), max(1, height // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


#: Supported blur methods, keyed by their UI label.
METHODS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "Gaussian": gaussian,
    "Box": box,
    "Median": median,
    "Pixelate": pixelate,
}


def apply_blur(img: np.ndarray, method: str, strength: int) -> np.ndarray:
    """Blur ``img`` using ``method`` at ``strength`` (0-100)."""
    try:
        func = METHODS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown method {method!r}. Expected one of {sorted(METHODS)}."
        ) from exc
    return np.ascontiguousarray(func(img, int(strength)))


def region_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Boolean ``(H, W)`` mask of a circle.

    ``center`` is a fraction of width/height, ``radius`` a fraction of the
    shorter side.
    """
    height, width = shape
    cx, cy = center[0] * width, center[1] * height
    r = max(1.0, radius * min(height, width))

    yy, xx = np.ogrid[:height, :width]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def apply_region(
    original: np.ndarray,
    blurred: np.ndarray,
    mode: str,
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Composite ``blurred`` over ``original`` according to ``mode``.

    ``whole`` applies the blur everywhere, ``circle`` blurs only inside the
    circle and ``inverse`` blurs everything outside of it.
    """
    if mode not in REGION_MODES:
        raise ValueError(f"Unknown region mode {mode!r}. Expected one of {REGION_MODES}.")
    if mode == "whole":
        return blurred

    mask = region_mask(original.shape[:2], center, radius)[..., None]
    keep_original = mask if mode == "inverse" else ~mask
    return np.where(keep_original, original, blurred)


def load_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes into an RGB array (EXIF rotation applied)."""
    with Image.open(io.BytesIO(data)) as img:
        rotated = ImageOps.exif_transpose(img)
        return np.asarray(rotated.convert("RGB"))


def encode_image(img: np.ndarray, fmt: str = "PNG") -> bytes:
    """Encode an RGB array back into image bytes."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buffer, format=fmt)
    return buffer.getvalue()


def process(
    data: bytes,
    method: str = "Gaussian",
    strength: int = 50,
    mode: str = "whole",
    center: tuple[float, float] = (0.5, 0.5),
    radius: float = 0.35,
) -> np.ndarray:
    """Decode ``data``, blur it and composite the result - the whole pipeline."""
    original = load_image(data)
    blurred = apply_blur(original, method, strength)
    return apply_region(original, blurred, mode, center, radius)


@lru_cache(maxsize=1)
def load_face_detector() -> cv2.FaceDetectorYN:
    """Load the YuNet ONNX face detector, with a fixable hint when it is missing."""
    if not FACE_MODEL.is_file():
        raise FileNotFoundError(
            f"Missing '{FACE_MODEL.name}'. Copy it next to blur_utils.py to enable face blur."
        )

    try:
        return cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.5, 0.3, 5000)
    except cv2.error as exc:
        raise RuntimeError(f"Could not load '{FACE_MODEL.name}': {exc}") from exc


def detect_faces(
    img: np.ndarray,
    net: cv2.FaceDetectorYN | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> List[Box]:
    """Return pixel boxes for every face detected in the RGB image ``img``."""
    net = net if net is not None else load_face_detector()
    height, width = img.shape[:2]

    net.setScoreThreshold(float(confidence))
    net.setInputSize((width, height))
    # YuNet expects BGR like the rest of OpenCV.
    _, faces = net.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if faces is None:
        return []

    boxes: List[Box] = []
    for face in faces:
        # Row: x, y, w, h, then 10 landmark coords, then the score.
        x, y, face_w, face_h = face[:4]
        left, top = max(0, int(x)), max(0, int(y))
        right, bottom = min(width, left + int(face_w)), min(height, top + int(face_h))
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom))
    return boxes


def _expand(box: Box, padding: float, width: int, height: int) -> Box:
    """Grow a box by ``padding`` (fraction of its size), clipped to the image."""
    left, top, right, bottom = box
    pad_x = int((right - left) * padding)
    pad_y = int((bottom - top) * padding)
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def blur_faces(
    img: np.ndarray,
    method: str = "Gaussian",
    strength: int = 80,
    confidence: float = DEFAULT_CONFIDENCE,
    padding: float = 0.2,
    net: cv2.FaceDetectorYN | None = None,
) -> Tuple[np.ndarray, int]:
    """Blur every detected face. Returns the blurred image and the face count."""
    boxes = detect_faces(img, net=net, confidence=confidence)
    if not boxes:
        return img.copy(), 0

    blurred = apply_blur(img, method, strength)
    result = img.copy()
    height, width = result.shape[:2]
    for box in boxes:
        left, top, right, bottom = _expand(box, padding, width, height)
        if right > left and bottom > top:
            result[top:bottom, left:right] = blurred[top:bottom, left:right]
    return result, len(boxes)


def process_faces(
    data: bytes,
    method: str = "Gaussian",
    strength: int = 80,
    confidence: float = DEFAULT_CONFIDENCE,
    padding: float = 0.2,
) -> Tuple[np.ndarray, int]:
    """Decode ``data`` and blur all detected faces - returns image + face count."""
    return blur_faces(
        load_image(data),
        method=method,
        strength=strength,
        confidence=confidence,
        padding=padding,
    )
