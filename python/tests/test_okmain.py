from okmain import dominant_color
from okmain._core import dominant_color_from_rgb_bytes
from PIL import Image


class TestDominantColor:
    def test_rgb_image(self) -> None:
        img = Image.new("RGB", (2, 2), (255, 0, 0))
        assert dominant_color(img) == (42, 42, 42)

    def test_rgba_raises(self) -> None:
        img = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        try:
            dominant_color(img)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "RGBA" in str(e)

    def test_grayscale_raises(self) -> None:
        img = Image.new("L", (2, 2), 128)
        try:
            dominant_color(img)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "L" in str(e)

    def test_1x1_image(self) -> None:
        img = Image.new("RGB", (1, 1), (10, 20, 30))
        assert dominant_color(img) == (42, 42, 42)


class TestCoreBinding:
    def test_valid_bytes(self) -> None:
        assert dominant_color_from_rgb_bytes(bytes([255, 0, 0, 0, 255, 0])) == (42, 42, 42)

    def test_invalid_length(self) -> None:
        try:
            dominant_color_from_rgb_bytes(bytes([1, 2]))
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_empty_bytes(self) -> None:
        try:
            dominant_color_from_rgb_bytes(b"")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
