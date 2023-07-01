from typing import Any, List, Tuple, Union
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from io_utils.io_image import draw_boxes_on_image, draw_on_image_and_save, load_img, save_img


def create_sample_image(path, size, color=(255, 0, 0)):
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)


def test_load_img(tmp_path):
    """
    Test case for loading an image using the load_img function.

    Args:
        tmp_path: pytest fixture for creating a temporary directory.
    """
    # Create a sample image
    img_path = (tmp_path / "test_image.png").as_posix()
    create_sample_image(img_path, (100, 100))

    # Test loading the image without any transformations
    loaded_img = load_img(img_path)
    assert isinstance(loaded_img, np.ndarray)
    assert loaded_img.shape == (100, 100, 3)
    assert loaded_img.dtype == np.uint8

    # Test loading the image with grayscale=True
    grayscale_img = load_img(img_path, grayscale=True)
    assert isinstance(grayscale_img, np.ndarray)
    assert grayscale_img.shape == (100, 100)
    assert grayscale_img.dtype == np.uint8

    # Test loading the image with dsize=(50, 50)
    resized_img = load_img(img_path, dsize=(50, 50))
    assert isinstance(resized_img, np.ndarray)
    assert resized_img.shape == (50, 50, 3)
    assert resized_img.dtype == np.uint8

    # Test loading the image with channels_first=True and grayscale image
    with pytest.raises(ValueError):
        load_img(img_path, grayscale=True, channels_first=True)


def test_save_img(tmp_path):
    """
    Test case for saving an image using the save_img function.

    Args:
        tmp_path: pytest fixture for creating a temporary directory.
    """
    # Create a sample image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "test_image.png"

    # Save the image without rescaling
    save_img(image_path, image)
    loaded_image = cv2.imread(str(image_path))
    assert loaded_image is not None
    assert loaded_image.shape == image.shape
    assert np.array_equal(loaded_image, image)

    # Save the image with rescaling
    save_img(image_path, image, flag_rescale_to_255=True)
    loaded_rescaled_image = cv2.imread(str(image_path))
    assert loaded_rescaled_image is not None
    assert loaded_rescaled_image.shape == image.shape
    assert np.array_equal(loaded_rescaled_image, image * 255)

    # Test saving a floating-point image
    float_image = np.zeros((100, 100, 3), dtype=np.float32)
    save_img(image_path, float_image)
    loaded_float_image = cv2.imread(str(image_path))
    assert loaded_float_image is not None
    assert loaded_float_image.shape == image.shape
    assert np.array_equal(loaded_float_image, image.astype(np.uint8))

    # Save the floating-point image with rescaling
    save_img(image_path, float_image, flag_rescale_to_255=True)
    loaded_rescaled_float_image = cv2.imread(str(image_path))
    assert loaded_rescaled_float_image is not None
    assert loaded_rescaled_float_image.shape == image.shape
    assert np.array_equal(loaded_rescaled_float_image, (image * 255).astype(np.uint8))


def test_draw_boxes_on_image():
    """
    Test case for drawing bounding boxes on an image using the draw_boxes_on_image function.
    """
    # Create a sample image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Define bounding boxes
    boxes = np.array([[10, 10, 40, 40], [60, 60, 90, 90]], dtype=np.uint16)

    # Draw boxes on the image
    drawn_image = draw_boxes_on_image(image.copy(), boxes, thickness=-1)

    # Verify the drawn image
    assert drawn_image.shape == image.shape

    # Verify the drawn bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        color = np.array([0, 255, 0])
        region = drawn_image[y1:y2, x1:x2, :]
        assert np.all(region == color)

    # Test drawing boxes with different colors
    colors = [(255, 0, 0), (0, 0, 255)]
    drawn_image_colors = draw_boxes_on_image(image.copy(), boxes, colors=colors, thickness=-1)

    # Verify the drawn image with different colors
    assert drawn_image_colors.shape == image.shape

    # Verify the drawn bounding boxes with different colors
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = np.array(colors[idx])
        region = drawn_image_colors[y1:y2, x1:x2, :]
        assert np.all(region == color)

    # checking that the classes can be added to the image.
    classes = [1, 2]
    # Draw boxes on the image
    drawn_image = draw_boxes_on_image(image.copy(), boxes, classes=classes, thickness=-1)


@patch("io_utils.io_image.draw_boxes_on_image")
@patch("io_utils.io_image.draw_circles_on_image")
@patch("io_utils.io_image.save_img")
def test_draw_on_image_and_save(mock_save_img, mock_draw_circles_on_image, mock_draw_boxes_on_image, tmp_path):
    """
    Test case for drawing on an image and saving it using the draw_on_image_and_save function.
    """

    # Set up mock objects
    path = (tmp_path / "test/test.png").as_posix()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[10, 10, 40, 40], [60, 60, 90, 90]]
    circles = [[50, 50, 20], [80, 80, 30]]
    colors = [(0, 255, 0), (255, 0, 0)]
    classes = [1, 2]
    thickness = 2

    mock_draw_boxes_on_image.return_value = img
    mock_draw_circles_on_image.return_value = img

    # Run the function
    draw_on_image_and_save(path, img, boxes, circles, colors, classes, thickness)

    # Verify the mock function calls
    mock_draw_boxes_on_image.assert_called_once_with(
        img=img, boxes=boxes, colors=colors, classes=classes, thickness=thickness
    )
    mock_draw_circles_on_image.assert_called_once_with(img=img, circles=circles, colors=colors, classes=classes)
    mock_save_img.assert_called_once_with(path, img)
