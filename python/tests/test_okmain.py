import pytest
from PIL import Image

from okmain import RGB, DebugInfo, ScoredCentroid, colors


def make_rgb_image(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def test_rgb_image() -> None:
    img = make_rgb_image((10, 10), (200, 100, 50))
    result = colors(img)
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], RGB)


def test_rgba_raises() -> None:
    img = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
    with pytest.raises(ValueError, match="RGBA"):
        colors(img)


def test_non_rgb_raises() -> None:
    img = Image.new("L", (2, 2), 128)
    with pytest.raises(ValueError, match="L"):
        colors(img)


def test_single_color_roundtrip() -> None:
    img = Image.new("RGB", (1, 1), (255, 0, 0))
    result = colors(img)
    assert len(result) >= 1
    assert result[0].r > 200


def test_with_debug_returns_tuple() -> None:
    img = make_rgb_image((10, 10), (200, 100, 50))
    result = colors(img, with_debug=True)
    assert isinstance(result, tuple)
    color_list, debug = result
    assert isinstance(color_list, list)
    assert isinstance(debug, DebugInfo)
    assert len(debug.scored_centroids) > 0
    sc = debug.scored_centroids[0]
    assert isinstance(sc, ScoredCentroid)
    assert isinstance(sc.rgb, RGB)
    assert isinstance(sc.mask_weighted_counts, float)
    assert isinstance(sc.chroma, float)
    assert isinstance(sc.final_score, float)


def test_kw_only_config() -> None:
    from okmain._core import (
        DEFAULT_CHROMA_WEIGHT,
        DEFAULT_MASK_SATURATED_THRESHOLD,
        DEFAULT_MASK_WEIGHT,
        DEFAULT_WEIGHTED_COUNTS_WEIGHT,
    )

    img = make_rgb_image((10, 10), (200, 100, 50))
    result_default = colors(img)
    result_explicit = colors(
        img,
        mask_saturated_threshold=DEFAULT_MASK_SATURATED_THRESHOLD,
        mask_weight=DEFAULT_MASK_WEIGHT,
        mask_weighted_counts_weight=DEFAULT_WEIGHTED_COUNTS_WEIGHT,
        chroma_weight=DEFAULT_CHROMA_WEIGHT,
    )
    assert result_default == result_explicit


def test_invalid_config_raises() -> None:
    img = make_rgb_image((10, 10), (200, 100, 50))
    with pytest.raises(ValueError):
        colors(img, chroma_weight=2.0)


def test_dominance_ordering() -> None:
    # 2x2 image: pixel (0,0) is gray, rest are saturated red
    img = Image.new("RGB", (2, 2), (255, 0, 0))
    img.putpixel((0, 0), (100, 100, 100))
    result = colors(img)
    assert len(result) == 2
    assert result[0].r > 200
    assert result[0].g < 50
    assert abs(result[1].r - result[1].g) < 30
