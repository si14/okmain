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
    r: int
    g: int
    b: int


@dataclass(frozen=True, slots=True)
class Oklab:
    l: float  # noqa: E741
    a: float
    b: float


@dataclass(frozen=True, slots=True)
class ScoredCentroid:
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
