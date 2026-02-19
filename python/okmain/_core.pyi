DEFAULT_MASK_SATURATED_THRESHOLD: float
DEFAULT_MASK_WEIGHT: float
DEFAULT_WEIGHTED_COUNTS_WEIGHT: float
DEFAULT_CHROMA_WEIGHT: float

class _ScoredCentroid:
    @property
    def rgb(self) -> tuple[int, int, int]: ...
    @property
    def oklab(self) -> tuple[float, float, float]: ...
    @property
    def mask_weighted_counts(self) -> float: ...
    @property
    def mask_weighted_counts_score(self) -> float: ...
    @property
    def chroma(self) -> float: ...
    @property
    def chroma_score(self) -> float: ...
    @property
    def final_score(self) -> float: ...

class _DebugInfo:
    @property
    def scored_centroids(self) -> list[_ScoredCentroid]: ...
    @property
    def kmeans_loop_iterations(self) -> list[int]: ...
    @property
    def kmeans_converged(self) -> list[bool]: ...

def _colors(
    buf: bytes,
    width: int,
    height: int,
    mask_saturated_threshold: float,
    mask_weight: float,
    mask_weighted_counts_weight: float,
    chroma_weight: float,
) -> list[tuple[int, int, int]]: ...
def _colors_debug(
    buf: bytes,
    width: int,
    height: int,
    mask_saturated_threshold: float,
    mask_weight: float,
    mask_weighted_counts_weight: float,
    chroma_weight: float,
) -> tuple[list[tuple[int, int, int]], _DebugInfo]: ...
