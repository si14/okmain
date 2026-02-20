"""`okmain` finds the main colors of an image and makes sure they look good.

Color operations in a state-of-the-art perceptually linear color space (Oklab),
with position- and visual prominence-based color prioritization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self, overload

from PIL import Image

from okmain._core import (
    DEFAULT_CHROMA_WEIGHT,
    DEFAULT_MASK_SATURATED_THRESHOLD,
    DEFAULT_MASK_WEIGHT,
    DEFAULT_WEIGHTED_COUNTS_WEIGHT,
    _colors_debug,
    _DebugInfo,
    _ScoredCentroid,
)

__all__ = [
    "colors",
    "RGB",
    "Oklab",
    "ScoredCentroid",
    "DebugInfo",
    "DEFAULT_MASK_SATURATED_THRESHOLD",
    "DEFAULT_MASK_WEIGHT",
    "DEFAULT_WEIGHTED_COUNTS_WEIGHT",
    "DEFAULT_CHROMA_WEIGHT",
]


@dataclass(frozen=True, slots=True)
class RGB:
    """An sRGB color with red, green, and blue components in the [0, 255] range."""

    r: int
    g: int
    b: int


@dataclass(frozen=True, slots=True)
class Oklab:
    """A color in the Oklab perceptually uniform color space."""

    l: float  # noqa: E741
    a: float
    b: float


@dataclass(frozen=True, slots=True)
class ScoredCentroid:
    """Debug details about a centroid in the Oklab color space and its score.

    Attributes:
        rgb: sRGB color of the centroid.
        oklab: Oklab color of the centroid.
        mask_weighted_counts: The fraction of pixels assigned to this centroid, with a mask
            reducing the impact of peripheral pixels applied.
        mask_weighted_counts_score: The score of the centroid based on mask-weighted pixel counts.
        chroma: Centroid's Oklab chroma (calculated from the Oklab value and normalized to [0, 1]).
        chroma_score: The score of the centroid based on chroma.
        final_score: The final score of the centroid, combining two scores based on provided
            weights.
    """

    rgb: RGB
    oklab: Oklab
    mask_weighted_counts: float
    mask_weighted_counts_score: float
    chroma: float
    chroma_score: float
    final_score: float

    @classmethod
    def _from_core(cls, sc: _ScoredCentroid) -> Self:
        rgb_r, rgb_g, rgb_b = sc.rgb
        lab_l, lab_a, lab_b = sc.oklab
        return cls(
            rgb=RGB(rgb_r, rgb_g, rgb_b),
            oklab=Oklab(lab_l, lab_a, lab_b),
            mask_weighted_counts=sc.mask_weighted_counts,
            mask_weighted_counts_score=sc.mask_weighted_counts_score,
            chroma=sc.chroma,
            chroma_score=sc.chroma_score,
            final_score=sc.final_score,
        )


@dataclass(frozen=True, slots=True)
class DebugInfo:
    """Debug info returned by ``colors()`` when called with ``with_debug=True``.

    Attributes:
        scored_centroids: The Okmain algorithm looks for k-means centroids in the Oklab color
            space. This field contains details about the centroids that were found in the image.
        kmeans_loop_iterations: The number of iterations the k-means algorithm took until the
            position of centroids stopped changing. A list, because Okmain can re-run k-means
            with a lower number of centroids if some of the discovered centroids are too close.
        kmeans_converged: Did k-means search converge? If not, it was cut off by the maximum
            number of iterations. A list for the same reason ``kmeans_loop_iterations`` is.
    """

    scored_centroids: list[ScoredCentroid]
    kmeans_loop_iterations: list[int]
    kmeans_converged: list[bool]

    @classmethod
    def _from_core(cls, debug: _DebugInfo) -> Self:
        return cls(
            scored_centroids=[ScoredCentroid._from_core(sc) for sc in debug.scored_centroids],
            kmeans_loop_iterations=list(debug.kmeans_loop_iterations),
            kmeans_converged=list(debug.kmeans_converged),
        )


@overload
def colors(
    image: Image.Image,
    *,
    mask_saturated_threshold: float = ...,
    mask_weight: float = ...,
    mask_weighted_counts_weight: float = ...,
    chroma_weight: float = ...,
    with_debug: Literal[True],
) -> tuple[list[RGB], DebugInfo]: ...


@overload
def colors(
    image: Image.Image,
    *,
    mask_saturated_threshold: float = ...,
    mask_weight: float = ...,
    mask_weighted_counts_weight: float = ...,
    chroma_weight: float = ...,
    with_debug: Literal[False] = ...,
) -> list[RGB]: ...


def colors(
    image: Image.Image,
    *,
    mask_saturated_threshold: float = DEFAULT_MASK_SATURATED_THRESHOLD,
    mask_weight: float = DEFAULT_MASK_WEIGHT,
    mask_weighted_counts_weight: float = DEFAULT_WEIGHTED_COUNTS_WEIGHT,
    chroma_weight: float = DEFAULT_CHROMA_WEIGHT,
    with_debug: bool = False,
) -> list[RGB] | tuple[list[RGB], DebugInfo]:
    """Extract dominant colors from a PIL image.

    The image must be in RGB mode; other modes (e.g. RGBA) raise ``ValueError``.

    Returns up to four dominant colors as :class:`RGB` values, sorted by dominance
    (the most dominant color first). If some colors are too close, fewer colors
    might be returned.

    Pass ``with_debug=True`` to also receive a :class:`DebugInfo` with internal
    algorithm details.

    Args:
        image: A PIL image in RGB mode.
        mask_saturated_threshold: The algorithm uses a mask to prioritize central pixels while
            considering the relative color dominance. The mask is a 1.0-weight rectangle starting
            at ``mask_saturated_threshold * 100%`` and finishing at
            ``(1.0 - mask_saturated_threshold) * 100%`` on both axes, with linear weight falloff
            from 1.0 at the border of the rectangle to 0.1 at the border of the image.
            Must be in the ``[0.0, 0.5)`` range.
        mask_weight: The weight of the mask, which can be used to reduce the impact of the mask
            on less-central pixels. By default it's set to 1.0, but by reducing this number you
            can increase the relative contribution of peripheral pixels.
            Must be in the ``[0.0, 1.0]`` range.
        mask_weighted_counts_weight: After the number of pixels belonging to every color is added
            up (with the mask reducing the contribution of peripheral pixels), the sums are
            normalized to add up to 1.0, and used as a part of the final score that decides the
            ordering of the colors. This parameter sets the relative weight of this component in
            the final score. Must be in the ``[0.0, 1.0]`` range and add up to 1.0 together with
            ``chroma_weight``.
        chroma_weight: For each color its saturation (Oklab chroma) is used to prioritize colors
            that are visually more prominent. This parameter controls the relative contribution
            of chroma into the final score. Must be in the ``[0.0, 1.0]`` range and add up to
            1.0 together with ``mask_weighted_counts_weight``.
        with_debug: If ``True``, return a ``(colors, debug_info)`` tuple instead of just the
            color list.

    Returns:
        A list of :class:`RGB` colors sorted by dominance, or a tuple of that list and a
        :class:`DebugInfo` if ``with_debug=True``.

    Raises:
        ValueError: If the image mode is not RGB, or if any config parameter is out of range.
    """
    if image.mode != "RGB":
        raise ValueError(f"expected RGB image, got {image.mode!r}")
    buf = image.tobytes()
    width, height = image.size
    raw_colors, raw_debug = _colors_debug(
        buf,
        width,
        height,
        mask_saturated_threshold,
        mask_weight,
        mask_weighted_counts_weight,
        chroma_weight,
    )
    color_list = [RGB(*c) for c in raw_colors]
    if with_debug:
        return color_list, DebugInfo._from_core(raw_debug)
    return color_list
