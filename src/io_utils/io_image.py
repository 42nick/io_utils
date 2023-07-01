from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_img(
    path: str, dsize: tuple[int, int] | None = None, grayscale: bool = False, channels_first: bool = False
) -> np.ndarray:
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)

    if dsize is not None:
        img = cv2.resize(img, dsize=dsize)

    if channels_first:
        img = img.transpose(2, 0, 1)

    return img


def save_img(path: str, img: np.ndarray, flag_rescale_to_255: bool = False) -> None:
    if flag_rescale_to_255:
        img = (img * 255).astype(np.uint8)

    if img.dtype == np.float32:
        img = img.astype(np.uint8)
    cv2.imwrite(path, img)


def draw_boxes_on_image(
    img: np.ndarray,
    boxes: np.ndarray,
    colors: tuple[int, int, int] | list[tuple[int, int, int]] = (0, 255, 0),
    classes: list[int] | np.ndarray | None = None,
    thickness: int = 2,
) -> np.ndarray:
    for idx, box in enumerate(boxes):
        x_1, y_1, x_2, y_2 = box
        color = colors[idx] if isinstance(colors, list) else colors

        img = cv2.rectangle(img, (x_1, y_1), (x_2, y_2), color, thickness=thickness)
        if classes is not None:
            img = cv2.putText(
                img,
                str(classes[0]),
                (x_1, y_1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return img


def draw_circles_on_image(
    img: np.ndarray,
    circles: np.ndarray,
    colors: tuple[int, int, int] | list[tuple[int, int, int]] = (0, 255, 0),
    classes: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    for idx, circle in enumerate(circles):
        circle_x, circle_y = circle
        color = colors[idx] if isinstance(colors, list) else colors
        img = cv2.circle(img, (circle_x, circle_y), 10, color, thickness=2)
        if classes is not None:
            img = cv2.putText(
                img,
                str(classes[0]),
                (circle_x, circle_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return img


def draw_on_image_and_save(
    path: str,
    img: np.ndarray,
    boxes: np.ndarray | None = None,
    circles: np.ndarray | None = None,
    colors: tuple[int, int, int] | list[tuple[int, int, int]] = (0, 255, 0),
    classes: list[int] | np.ndarray | None = None,
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
