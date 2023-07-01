import cv2
import numpy as np
import pytest

from io_utils.io_image import load_img


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
