from __future__ import annotations

from typing import Any, Union

import cv2
import numpy as np
from numpy.typing import NDarray
from pathlib import Path


def load_img(
    path: str, dsize: tuple[int, int] | None = None, grayscale: bool = False, channels_first: bool = False
) -> NDarray[np.uint8]:
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)

    if dsize is not None:
        img = cv2.resize(img, dsize=dsize)

    if channels_first:
        img = img.transpose(2, 0, 1)

    return img


def save_img(path: str, img: NDarray[Union[np.uint8, np.float32]], flag_rescale_to_255: bool = False) -> None:
    if flag_rescale_to_255:
        img = (img * 255).astype(np.uint8)

    if img.dtype == np.float32:
        img = img.astype(np.uint8)
    cv2.imwrite(path, img)


def draw_boxes_on_image(
    img: NDarray[np.uint8],
    boxes: NDarray[Union[np.float32, np.uint16]],
    colors: tuple[int, int, int] | list[tuple[int, int, int]] = (0, 255, 0),
    classes: list[int] | NDarray[Any] | None = None,
    thickness: int = 2,
) -> NDarray[np.uint8]:
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[idx] if isinstance(colors, list) else colors

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
        if classes is not None:
            img = cv2.putText(
                img,
                str(classes[0]),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return img


def draw_circles_on_image(
    img: NDarray[np.uint8],
    circles: NDarray[np.uint16],
    colors: tuple[int, int, int] | list[tuple[int, int, int]] = (0, 255, 0),
    classes: list[int] | NDarray[Any] | None = None,
) -> NDarray[np.uint8]:
    for idx, circle in enumerate(circles):
        x, y = circle
        color = colors[idx] if isinstance(colors, list) else colors
        img = cv2.circle(img, (x, y), 10, color, thickness=2)
        if classes is not None:
            img = cv2.putText(
                img,
                str(classes[0]),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return img


def draw_on_image_and_save(
    path: str,
    img: NDarray[np.uint8],
    boxes: NDarray[Union[np.float32, np.uint16]] | None = None,
    circles: NDarray[np.uint16] | None = None,
    colors: tuple[int, int, int] | list[tuple[int, int, int]] = (0, 255, 0),
    classes: list[int] | NDarray[Any] | None = None,
    thickness: int = 2,
) -> None:
    storing_path = Path(path)
    if not storing_path.parent.exists():
        storing_path.parent.mkdir(parents=True)

    if boxes is not None:
        img = draw_boxes_on_image(img=img, boxes=boxes, colors=colors, classes=classes, thickness=thickness)

    if circles is not None:
        img = draw_circles_on_image(img=img, circles=circles, colors=colors, classes=classes)

    save_img(path, img)
