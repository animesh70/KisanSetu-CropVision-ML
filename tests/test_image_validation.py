import io

import pytest
from PIL import Image

from app.services.image_validation import InvalidImage, validate_image


def image_bytes(format_name="PNG", size=(256, 256)):
    image = Image.new("RGB", size, (60, 140, 65))
    output = io.BytesIO()
    image.save(output, format=format_name)
    return output.getvalue()


@pytest.mark.parametrize(
    ("format_name", "mime"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_supported_real_image_bytes_are_converted_to_rgb(format_name, mime):
    result = validate_image(image_bytes(format_name), mime)
    assert result.rgb.mode == "RGB"
    assert result.rgb.size == (256, 256)


def test_extension_or_mime_cannot_override_actual_format():
    with pytest.raises(InvalidImage):
        validate_image(image_bytes("PNG"), "image/jpeg")


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n"])
def test_empty_or_corrupt_bytes_are_rejected(data):
    with pytest.raises(InvalidImage):
        validate_image(data, "image/png")


def test_unsupported_mime_is_rejected():
    with pytest.raises(InvalidImage):
        validate_image(image_bytes(), "application/octet-stream")


def test_large_dimensions_are_rejected():
    with pytest.raises(InvalidImage):
        validate_image(image_bytes(size=(8100, 100)), "image/png")
