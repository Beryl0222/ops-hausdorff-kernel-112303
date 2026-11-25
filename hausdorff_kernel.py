"""Chunked, bidirectional point-set distances for calibration and training.

The operator is intentionally streaming: a chunk pair is the largest geometric
working set, and nearest-neighbour state is one record per unique query point.
It never constructs or retains an ``N_left * N_right`` distance matrix.

The exchange record is only one convenient adapter.  Generic callers can build
:class:`PointChunk` / :class:`PointFrame` objects from any scan source.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import sys

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from hausdorff_kernel_input import SAMPLE, load_record  # noqa: E402

# ---------------------------------------------------------------------------
# Errors and small value objects
# ---------------------------------------------------------------------------

EPS = 1e-12
BOUNDARY_EPS = 1e-10


class DistanceKernelError(Exception):
    """Base error for the point-distance kernel."""


class ChunkBudgetError(DistanceKernelError):
    """Raised when a source cannot respect the configured working budget."""


class RegionConflictError(DistanceKernelError):
    """Raised when strict mode asks for an immediate duplicate conflict error."""

    def __init__(self, conflicts: Sequence[Mapping[str, Any]]):
        self.conflicts = list(conflicts)
        ids = ", ".join(sorted({str(c.get("point_id")) for c in self.conflicts}))
        super().__init__(f"duplicate point content conflict in region(s): {ids}")


class UnstableSupportError(DistanceKernelError):
    """Raised when gradients are requested in strict mode on unstable supports."""

    def __init__(self, supports: Sequence[Mapping[str, Any]]):
        self.supports = list(supports)
        super().__init__(
            f"{len(self.supports)} distance support(s) do not have a stable "
            "local derivative; inspect their stability reasons before training"
        )


@dataclass(frozen=True)
class MemoryBudget:
    """Working budget.

    ``max_points`` corresponds to the exchange record's
    ``point_chunk_max``.  ``scratch_bytes`` bounds the temporary vector/tile
    used while comparing one query with candidates in a chunk.
    """

    max_points: int = 8192
    scratch_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if int(self.max_points) <= 0:
            raise ValueError("max_points must be positive")
        if int(self.scratch_bytes) <= 64:
            raise ValueError("scratch_bytes is too small for distance scratch")
        object.__setattr__(self, "max_points", int(self.max_points))
        object.__setattr__(self, "scratch_bytes", int(self.scratch_bytes))

    def candidate_block_size(self, query_rows: int = 1, bytes_per_candidate: int = 256) -> int:
        """Return a safe candidate block size for a temporary calculation."""
        query_rows = max(1, int(query_rows))
        # delta Nx3 float64, squared distance, normal dot and masks/temporaries.
        per_candidate = max(1, int(bytes_per_candidate)) * query_rows
        by_scratch = max(1, self.scratch_bytes // per_candidate)
        return int(max(1, min(self.max_points, by_scratch)))

    def query_block_size(self) -> int:
        return int(self.max_points)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def as_box(box: Any) -> np.ndarray:
    arr = np.asarray(box, dtype=float)
    if arr.shape != (2, 3):
        raise ValueError(f"box must have shape (2, 3), got {arr.shape}")
    if np.any(~np.isfinite(arr)):
        raise ValueError("box bounds must be finite")
    if np.any(arr[0] > arr[1]):
        raise ValueError("box minimum must not exceed maximum")
    return arr.copy()


def points_in_box(box: Any, xyz: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Closed AABB membership."""
    b = as_box(box)
    p = np.asarray(xyz, dtype=float).reshape(-1, 3)
    return np.all((p >= b[0] - eps) & (p <= b[1] + eps), axis=1)


def box_boundary_mask(box: Any, xyz: np.ndarray, tol: float = BOUNDARY_EPS) -> np.ndarray:
    """True for points on any face of a closed AABB."""
    b = as_box(box)
    p = np.asarray(xyz, dtype=float).reshape(-1, 3)
    low = np.any(np.abs(p - b[0]) <= tol, axis=1)
    high = np.any(np.abs(p - b[1]) <= tol, axis=1)
    return low | high


def squared_distance_to_box(
    xyz: np.ndarray, box: Any
) -> np.ndarray | float:
    """Squared Euclidean distance from point(s) to a closed AABB."""
    b = as_box(box)
    p = np.asarray(xyz, dtype=float)
    single = p.ndim == 1
    if single:
        p = p.reshape(1, 3)
    closest = np.minimum(np.maximum(p, b[0]), b[1])
    delta = p - closest
    d2 = np.einsum("ij,ij->i", delta, delta)
    if single:
        return float(d2[0])
    return d2


def bbox_lower_bound_sq(query_xyz: np.ndarray, bbox: Any) -> float:
    return float(squared_distance_to_box(np.asarray(query_xyz, dtype=float), bbox))


@dataclass(eq=False)
class Pose:
    """Rigid transform ``p_left = R @ p_right + translation``."""

    id: str = "pose"
    R: np.ndarray | None = None
    translation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0)
    angle_z_deg: float | None = None

    def __init__(
        self,
        id: str = "pose",
        R: np.ndarray | Sequence[Sequence[float]] | None = None,
        translation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        angle_z_deg: float | None = None,
    ) -> None:
        self.id = str(id)
        self.translation = np.asarray(translation, dtype=float).reshape(3).copy()
        if R is None:
            deg = 0.0 if angle_z_deg is None else float(angle_z_deg)
            rad = math.radians(deg)
            c, s = math.cos(rad), math.sin(rad)
            self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            self.angle_z_deg = deg
        else:
            self.R = np.asarray(R, dtype=float).reshape(3, 3).copy()
            if angle_z_deg is not None:
                self.angle_z_deg = float(angle_z_deg)
            else:
                # Recover the exchange-record Z angle when possible.
                self.angle_z_deg = math.degrees(math.atan2(self.R[1, 0], self.R[0, 0]))
        if not np.all(np.isfinite(self.R)) or not np.all(np.isfinite(self.translation)):
            raise ValueError("pose contains non-finite values")

    @classmethod
    def z(cls, angle_z_deg: float = 0.0, translation: Sequence[float] = (0, 0, 0),
          pose_id: str | None = None) -> "Pose":
        return cls(id=pose_id or f"z{angle_z_deg:g}", angle_z_deg=angle_z_deg,
                   translation=translation)

    @property
    def t(self) -> np.ndarray:
        return self.translation

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        p = np.asarray(xyz, dtype=float)
        return p @ self.R.T + self.translation

    def inverse_transform(self, xyz: np.ndarray) -> np.ndarray:
        p = np.asarray(xyz, dtype=float)
        return (p - self.translation) @ self.R

    def transform_direction(self, normals: np.ndarray) -> np.ndarray:
        n = np.asarray(normals, dtype=float)
        return n @ self.R.T

    def rotation_z_derivative_left(self, xyz_left: np.ndarray) -> np.ndarray:
        """Derivative of transformed right point wrt Z angle, in left frame.

        ``xyz_left`` is the transformed right point ``q = R p_right + t``.
        The translation is constant, so the rotated vector is ``q - t`` and
        the derivative for a Z rotation is ``e_z x (q - t)``.
        """
        q = np.asarray(xyz_left, dtype=float).reshape(-1, 3)
        rotated = q - self.translation
        out = np.empty_like(rotated)
        out[..., 0] = -rotated[..., 1]
        out[..., 1] = rotated[..., 0]
        out[..., 2] = 0.0
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "R": self.R.tolist(),
            "translation": self.translation.tolist(),
            "angle_z_deg": self.angle_z_deg,
        }


RigidPose = Pose


def common_visibility_mask(
    xyz: np.ndarray,
    side: str,
    left_view: Any,
    right_view: Any,
    pose: Pose | None = None,
    *,
    mode: str = "mutual",
    eps: float = EPS,
) -> np.ndarray:
    """Closed-view mask for points on one side after applying a pose."""
    pose = pose or Pose()
    p = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if side == "left":
        own = points_in_box(left_view, p, eps=eps)
        other_local = pose.inverse_transform(p)
        other = points_in_box(right_view, other_local, eps=eps)
    elif side == "right":
        own = points_in_box(right_view, p, eps=eps)
        other = points_in_box(left_view, pose.transform(p), eps=eps)
    else:
        raise ValueError("side must be 'left' or 'right'")
    return own & other if mode == "mutual" else own


def normal_compatibility_mask(
    query_normals_left: np.ndarray,
    candidate_normals_left: np.ndarray,
    min_dot: float,
    *,
    tol: float = EPS,
) -> np.ndarray:
    """Return a query-by-candidate boolean compatibility matrix."""
    q = np.asarray(query_normals_left, dtype=float).reshape(-1, 3)
    c = np.asarray(candidate_normals_left, dtype=float).reshape(-1, 3)
    dots = q @ c.T
    return dots >= float(min_dot) - tol


def make_pose(pose: Any = None, **kwargs: Any) -> Pose:
    """Convenience constructor accepting :class:`Pose`, dict, or angle args."""
    if isinstance(pose, Pose):
        return pose
    if isinstance(pose, Mapping):
        return Pose(
            id=pose.get("id", kwargs.pop("id", "pose")),
            R=pose.get("R"),
            translation=pose.get("translation", pose.get("t", (0, 0, 0))),
            angle_z_deg=pose.get("angle_z_deg"),
        )
    if pose is None:
        return Pose(**kwargs)
    raise TypeError(f"unsupported pose: {type(pose)!r}")


# ---------------------------------------------------------------------------
# Rules, chunks and frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionRule:
    min_normal_dot: float = 0.0
    min_coverage: float = 0.8
    max_distance_sq: float = 0.4
    quantile: float = 0.9

    @classmethod
    def from_sequence(cls, value: Sequence[float], region: str | None = None) -> "RegionRule":
        if not isinstance(value, Sequence) or len(value) != 4:
            raise ValueError(f"rule for {region!r} must contain four numbers")
        return cls(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    def to_dict(self) -> dict[str, float]:
        return {
            "min_normal_dot": self.min_normal_dot,
            "min_coverage": self.min_coverage,
            "max_distance_sq": self.max_distance_sq,
            "quantile": self.quantile,
        }


@dataclass(eq=False)
class PointChunk:
    chunk_id: str
    side: str
    ids: Sequence[str]
    regions: np.ndarray
    xyz: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    bbox: np.ndarray
    unit_normals: np.ndarray = field(repr=False)

    def __init__(
        self,
        chunk_id: str | None = None,
        side: str = "",
        ids: Sequence[str] | None = None,
        xyz: Any = None,
        normals: Any = None,
        weights: Any = None,
        bbox: Any = None,
        regions: Any = None,
        *,
        id: str | None = None,
        region: Any = None,
        point_ids: Sequence[str] | None = None,
    ) -> None:
        cid = chunk_id if chunk_id is not None else id
        if cid is None:
            raise ValueError("chunk_id/id is required")
        if ids is None:
            ids = point_ids
        if ids is None:
            raise ValueError("ids are required")
        if xyz is None or normals is None:
            raise ValueError("xyz and normals are required")
        reg = regions if regions is not None else region
        if reg is None:
            raise ValueError("regions/region is required")

        self.chunk_id = str(cid)
        self.side = str(side)
        self.ids = tuple(str(x) for x in ids)
        self.xyz = np.asarray(xyz, dtype=float)
        self.normals = np.asarray(normals, dtype=float)
        self.weights = (
            np.ones(len(self.ids), dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        self.regions = np.asarray(
            [str(x) for x in (reg if isinstance(reg, Sequence) and not isinstance(reg, str)
                              else [reg] * len(self.ids))],
            dtype=object,
        )
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"{self.chunk_id}: xyz must have shape (N, 3)")
        n = self.xyz.shape[0]
        if len(self.ids) != n or self.normals.shape != (n, 3):
            raise ValueError(f"{self.chunk_id}: ids/normals lengths do not match xyz")
        if self.weights.shape != (n,) or self.regions.shape != (n,):
            raise ValueError(f"{self.chunk_id}: weights/regions lengths do not match xyz")
        if not all(isinstance(x, str) for x in self.ids):
            # IDs are used in stable tie ordering; normalise only non-strings?
            # Keep raw comparable values when possible.  The record uses strings.
            self.ids = tuple(ids)  # type: ignore[arg-type]
        if not np.all(np.isfinite(self.xyz)) or not np.all(np.isfinite(self.normals)):
            raise ValueError(f"{self.chunk_id}: non-finite point data")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError(f"{self.chunk_id}: non-finite weights")

        norms = np.linalg.norm(self.normals, axis=1)
        if np.any(norms <= EPS):
            raise ValueError(f"{self.chunk_id}: zero-length normal")
        self.unit_normals = self.normals / norms[:, None]

        if bbox is None:
            self.bbox = np.array([self.xyz.min(axis=0), self.xyz.max(axis=0)], dtype=float) if n else \
                np.zeros((2, 3))
        else:
            self.bbox = as_box(bbox)

    @property
    def id(self) -> str:
        return self.chunk_id

    def __len__(self) -> int:
        return len(self.ids)

    def row_by_id(self, point_id: str) -> int:
        for i, x in enumerate(self.ids):
            if x == point_id:
                return i
        raise KeyError(point_id)

    def batches(self, budget: MemoryBudget | int | None = None) -> Iterator[tuple[int, int]]:
        limit = budget.max_points if isinstance(budget, MemoryBudget) else int(budget or 8192)
        for start in range(0, len(self), limit):
            yield start, min(len(self), start + limit)

    @classmethod
    def from_record(cls, side: str, chunk: Mapping[str, Any]) -> "PointChunk":
        rows = chunk["points"]
        return cls(
            chunk_id=chunk["id"],
            side=side,
            ids=[row[0] for row in rows],
            regions=[row[1] for row in rows],
            xyz=np.array([row[2] for row in rows], dtype=float),
            normals=np.array([row[3] for row in rows], dtype=float),
            weights=np.array([row[4] for row in rows], dtype=float),
            bbox=chunk.get("bbox"),
        )


PointCloudChunk = PointChunk


@dataclass(eq=False)
class ChunkHandle:
    """Lazy chunk metadata/loader for sources larger than working memory.

    The loader should return one physical chunk no larger than the frame's
    :class:`MemoryBudget.max_points`; temporary distance vectors are additionally
    tiled by ``scratch_bytes``.
    """

    chunk_id: str
    side: str
    bbox: Any
    loader: Callable[[], PointChunk]
    regions: Sequence[str] | None = None
    count: int | None = None

    @property
    def id(self) -> str:
        return self.chunk_id

    def load(self) -> PointChunk:
        chunk = self.loader()
        if not isinstance(chunk, PointChunk):
            raise TypeError("ChunkHandle loader must return a PointChunk")
        if not chunk.chunk_id:
            chunk.chunk_id = self.chunk_id
        if not chunk.side:
            chunk.side = self.side
        return chunk


LazyChunk = ChunkHandle

@dataclass(eq=False)
class PointFrame:
    """One camera/side: chunks plus its own closed scan-view AABB."""

    name: str
    view: np.ndarray
    chunks: dict[str, PointChunk]

    def __init__(self, name: str, view: Any, chunks: Sequence[PointChunk] | Mapping[str, PointChunk]):
        self.name = str(name)
        self.view = as_box(view)
        if isinstance(chunks, Mapping):
            self.chunks = {str(k): v for k, v in chunks.items()}
        else:
            self.chunks = {c.chunk_id: c for c in chunks}
        for cid, chunk in self.chunks.items():
            if chunk.side and chunk.side != self.name:
                # Side is useful provenance, but do not make construction brittle.
                chunk.side = self.name
            if chunk.chunk_id != cid:
                chunk.chunk_id = cid

    def chunk(self, chunk_id: str) -> PointChunk:
        return self.chunks[str(chunk_id)]

    def region_chunk_ids(self, region: str) -> list[str]:
        out = []
        for cid, chunk in self.chunks.items():
            regions = getattr(chunk, "regions", None)
            if regions is None:
                continue
            try:
                if np.any(np.asarray(regions, dtype=object) == region):
                    out.append(cid)
            except Exception:
                continue
        return sorted(out)


Frame = PointFrame
ScanFrame = PointFrame


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Match:
    query_id: Any
    neighbor_id: Any
    region: str
    direction: str
    distance_sq: float
    query_weight: float
    normal_dot: float
    query_side: str
    neighbor_side: str
    query_xyz_local: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(3))
    neighbor_xyz_local: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(3))
    query_chunk_ids: list[str] = field(default_factory=list)
    neighbor_chunk_ids: list[str] = field(default_factory=list)
    neighbor_weight: float = 1.0
    equidistant: bool = False
    normal_boundary: bool = False
    visibility_boundary: bool = False
    boundary_reasons: list[str] = field(default_factory=list)

    @property
    def distance(self) -> float:
        return math.sqrt(max(0.0, float(self.distance_sq)))

    @property
    def stable(self) -> bool:
        return not (self.equidistant or self.normal_boundary or self.visibility_boundary)

    @property
    def stability_reasons(self) -> list[str]:
        reasons = []
        if self.equidistant:
            reasons.append("equidistant_candidate")
        if self.normal_boundary:
            reasons.append("normal_angle_boundary")
        if self.visibility_boundary:
            reasons.append("visibility_boundary")
        reasons.extend(self.boundary_reasons)
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "neighbor_id": self.neighbor_id,
            "region": self.region,
            "direction": self.direction,
            "distance_sq": self.distance_sq,
            "distance": self.distance,
            "query_weight": self.query_weight,
            "normal_dot": self.normal_dot,
            "query_side": self.query_side,
            "neighbor_side": self.neighbor_side,
            "query_xyz_local": np.asarray(self.query_xyz_local).tolist(),
            "neighbor_xyz_local": np.asarray(self.neighbor_xyz_local).tolist(),
            "query_chunk_ids": list(self.query_chunk_ids),
            "neighbor_chunk_ids": list(self.neighbor_chunk_ids),
            "equidistant": self.equidistant,
            "normal_boundary": self.normal_boundary,
            "visibility_boundary": self.visibility_boundary,
            "stable": self.stable,
            "stability_reasons": self.stability_reasons,
            "boundary_reasons": list(self.boundary_reasons),
        }


NearestMatch = Match
BidirectionalMatch = Match


@dataclass(eq=False)
class Exclusion:
    query_id: Any
    direction: str
    region: str
    chunk_id: str
    pose_id: str
    lower_bound_sq: float
    incumbent_id: Any
    incumbent_distance_sq: float
    query_xyz_local: list[float]
    candidate_bbox: list[list[float]]
    criterion: str = "bbox_lower_bound_gt_incumbent"
    incumbent_xyz_local: list[float] | None = None

    def recheck_lower_bound_sq(self, pose: Pose) -> float:
        q = np.asarray(self.query_xyz_local, dtype=float)
        if self.direction == "left_to_right":
            q_candidate_frame = pose.inverse_transform(q)
        elif self.direction == "right_to_left":
            q_candidate_frame = pose.transform(q)
        else:
            raise ValueError(f"unknown direction {self.direction}")
        return bbox_lower_bound_sq(q_candidate_frame, self.candidate_bbox)

    def incumbent_distance_sq_under_pose(self, pose: Pose) -> float | None:
        """Recompute the retained incumbent distance under another pose."""
        if self.incumbent_xyz_local is None:
            return None
        q = np.asarray(self.query_xyz_local, dtype=float)
        n = np.asarray(self.incumbent_xyz_local, dtype=float)
        if self.direction == "left_to_right":
            delta = pose.transform(n) - q
        else:
            delta = pose.transform(q) - n
        return float(delta @ delta)

    def holds_for_pose(
        self,
        pose: Pose,
        tol: float = EPS,
        incumbent_distance_sq: float | None = None,
    ) -> bool:
        """Whether the retained bbox exclusion still proves no improvement.

        By default the incumbent distance is recomputed under ``pose``; an
        explicit current incumbent distance may be supplied from a new session.
        """
        lower = self.recheck_lower_bound_sq(pose)
        incumbent = incumbent_distance_sq
        if incumbent is None:
            incumbent = self.incumbent_distance_sq_under_pose(pose)
        if incumbent is None:
            # Backwards-compatible basis without incumbent coordinates.
            incumbent = self.incumbent_distance_sq
        return lower > float(incumbent) + tol

    def recheck(
        self,
        pose: Pose,
        tol: float = EPS,
        incumbent_distance_sq: float | None = None,
    ) -> "ExclusionRecheck":
        lower = self.recheck_lower_bound_sq(pose)
        incumbent = incumbent_distance_sq
        if incumbent is None:
            incumbent = self.incumbent_distance_sq_under_pose(pose)
        if incumbent is None:
            incumbent = self.incumbent_distance_sq
        still = lower > float(incumbent) + tol
        return ExclusionRecheck(
            exclusion=self,
            pose_id=pose.id,
            lower_bound_sq=lower,
            incumbent_distance_sq=float(incumbent),
            still_excluded=still,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "direction": self.direction,
            "region": self.region,
            "chunk_id": self.chunk_id,
            "pose_id": self.pose_id,
            "lower_bound_sq": self.lower_bound_sq,
            "incumbent_id": self.incumbent_id,
            "incumbent_distance_sq": self.incumbent_distance_sq,
            "criterion": self.criterion,
            "query_xyz_local": list(self.query_xyz_local),
            "incumbent_xyz_local": self.incumbent_xyz_local,
            "candidate_bbox": np.asarray(self.candidate_bbox).tolist(),
        }


@dataclass(frozen=True)
class ExclusionRecheck:
    exclusion: Exclusion
    pose_id: str
    lower_bound_sq: float
    still_excluded: bool
    incumbent_distance_sq: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_pose_id": self.exclusion.pose_id,
            "pose_id": self.pose_id,
            "chunk_id": self.exclusion.chunk_id,
            "query_id": self.exclusion.query_id,
            "lower_bound_sq": self.lower_bound_sq,
            "incumbent_distance_sq": (
                self.exclusion.incumbent_distance_sq
                if self.incumbent_distance_sq is None
                else self.incumbent_distance_sq
            ),
            "still_excluded": self.still_excluded,
        }


def should_skip_chunk(
    query_xyz: np.ndarray,
    candidate_bbox: Any,
    incumbent_distance_sq: float | None,
    *,
    tie_aware: bool = True,
    tol: float = EPS,
) -> tuple[bool, float, str]:
    """Decide whether a bbox-proven chunk cannot improve or tie.

    A tied candidate can break an ID tie, so equality is not skipped when
    ``tie_aware=True``.
    """
    lower = bbox_lower_bound_sq(query_xyz, candidate_bbox)
    if incumbent_distance_sq is None or not math.isfinite(float(incumbent_distance_sq)):
        return False, lower, "no_incumbent"
    incumbent = float(incumbent_distance_sq)
    threshold = incumbent + (tol if tie_aware else -tol)
    if lower > threshold:
        return True, lower, "bbox_lower_bound_gt_incumbent"
    return False, lower, "bbox_may_improve_or_tie"


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------


def weighted_quantile(
    distances_sq: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantile: float,
) -> float | None:
    """Smallest distance whose cumulative weight reaches ``quantile`` fraction."""
    d = np.asarray(distances_sq, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if d.size == 0:
        return None
    if d.shape != w.shape:
        raise ValueError("distances and weights must have equal lengths")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    valid = np.isfinite(d) & np.isfinite(w) & (w > 0)
    d, w = d[valid], w[valid]
    total = float(w.sum())
    if d.size == 0 or total <= 0:
        return None
    order = np.argsort(d, kind="stable")
    ds, ws = d[order], w[order]
    cumulative = np.cumsum(ws)
    target = quantile * total
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(idx, ds.size - 1)
    return float(ds[idx])


@dataclass(eq=False)
class DirectionReport:
    direction: str
    query_side: str
    candidate_side: str
    region: str
    status: str = "partial"
    total_weight: float = 0.0
    matched_weight: float = 0.0
    unmatched_weight: float = 0.0
    coverage: float = 0.0
    count_query: int = 0
    count_matched: int = 0
    weighted_quantile_sq: float | None = None
    weighted_mean_sq: float | None = None
    max_distance_sq: float | None = None
    min_normal_dot: float | None = None
    min_coverage_required: float | None = None
    max_distance_sq_allowed: float | None = None
    quantile_required: float | None = None
    coverage_ok: bool = False
    distance_ok: bool = False
    ok: bool = False
    matches: list[Match] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)

    @property
    def quantile_sq(self) -> float | None:
        return self.weighted_quantile_sq

    @property
    def coverage_ratio(self) -> float:
        return self.coverage

    @property
    def weighted_quantile(self) -> float | None:
        return self.weighted_quantile_sq

    @property
    def weighted_quantile_distance(self) -> float | None:
        return None if self.weighted_quantile_sq is None else math.sqrt(self.weighted_quantile_sq)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "query_side": self.query_side,
            "candidate_side": self.candidate_side,
            "region": self.region,
            "status": self.status,
            "total_weight": self.total_weight,
            "matched_weight": self.matched_weight,
            "unmatched_weight": self.unmatched_weight,
            "coverage": self.coverage,
            "count_query": self.count_query,
            "count_matched": self.count_matched,
            "weighted_quantile_sq": self.weighted_quantile_sq,
            "quantile_sq": self.weighted_quantile_sq,
            "weighted_quantile_distance": self.weighted_quantile_distance,
            "weighted_mean_sq": self.weighted_mean_sq,
            "max_distance_sq": self.max_distance_sq,
            "rule": {
                "min_normal_dot": self.min_normal_dot,
                "min_coverage": self.min_coverage_required,
                "max_distance_sq": self.max_distance_sq_allowed,
                "quantile": self.quantile_required,
            },
            "coverage_ok": self.coverage_ok,
            "distance_ok": self.distance_ok,
            "ok": self.ok,
            "matches": [m.to_dict() for m in sorted(self.matches, key=lambda m: str(m.query_id))],
            "exclusions": [e.to_dict() for e in self.exclusions],
        }


@dataclass(eq=False)
class RegionReport:
    region: str
    pose_id: str
    status: str
    valid: bool
    released: bool
    assessable: bool
    missing_evidence: bool
    temporary_no_neighbor: bool
    frames: dict[str, Any]
    directions: dict[str, DirectionReport] = field(default_factory=dict)
    rule: RegionRule | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def left_to_right(self) -> DirectionReport | None:
        return self.directions.get("left_to_right")

    @property
    def right_to_left(self) -> DirectionReport | None:
        return self.directions.get("right_to_left")

    @property
    def evidence_missing(self) -> bool:
        return self.missing_evidence

    @property
    def unproven(self) -> bool:
        return self.missing_evidence

    @property
    def insufficient_evidence(self) -> bool:
        return self.missing_evidence

    @property
    def pending(self) -> bool:
        return self.status == "partial"

    @property
    def is_final(self) -> bool:
        return self.status != "partial"

    @property
    def final(self) -> bool:
        return self.is_final

    @property
    def complete(self) -> bool:
        return self.is_final

    @property
    def completion_status(self) -> str:
        return "final" if self.is_final else "partial"

    @property
    def conclusion(self) -> str:
        if self.status == "partial":
            return "temporary"
        if not self.valid:
            return "rejected"
        if self.missing_evidence:
            return "missing_evidence"
        if self.status == "not_visible":
            return "not_visible"
        return "matched"

    @property
    def eligible(self) -> bool:
        return self.released

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "pose_id": self.pose_id,
            "status": self.status,
            "completion_status": self.completion_status,
            "final": self.final,
            "complete": self.complete,
            "conclusion": self.conclusion,
            "valid": self.valid,
            "released": self.released,
            "assessable": self.assessable,
            "missing_evidence": self.missing_evidence,
            "temporary_no_neighbor": self.temporary_no_neighbor,
            "frames": self.frames,
            "rule": None if self.rule is None else self.rule.to_dict(),
            "issues": list(self.issues),
            "directions": {k: v.to_dict() for k, v in self.directions.items()},
        }


@dataclass(eq=False)
class PoseEvaluation:
    pose: Pose
    regions: dict[str, RegionReport]
    status: str
    approved: bool
    processed_chunks: list[str]
    pending_chunks: list[str]
    expected_chunks: list[str]
    telemetry: dict[str, Any]
    issues: list[dict[str, Any]] = field(default_factory=list)

    def released_regions(self) -> list[str]:
        return sorted([r for r, rep in self.regions.items() if rep.released])

    @property
    def ok(self) -> bool:
        return self.approved

    @property
    def final(self) -> bool:
        return self.status == "final"

    @property
    def allowed(self) -> bool:
        return self.approved

    def get_region(self, region: str) -> RegionReport:
        return self.regions[region]

    def iter_matches(self) -> Iterator[Match]:
        for region in sorted(self.regions):
            for direction in ("left_to_right", "right_to_left"):
                report = self.regions[region].directions.get(direction)
                if report is None:
                    continue
                yield from sorted(report.matches, key=lambda m: str(m.query_id))

    def write_matches_jsonl(self, path: str | Path) -> int:
        """Write one final nearest-neighbour row per query as JSON Lines."""
        count = 0
        with Path(path).open("w", encoding="utf-8") as handle:
            for match in self.iter_matches():
                handle.write(json.dumps(match.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        return count

    def write_exclusions_jsonl(self, path: str | Path) -> int:
        count = 0
        with Path(path).open("w", encoding="utf-8") as handle:
            for region in self.regions.values():
                for direction in region.directions.values():
                    for exclusion in direction.exclusions:
                        handle.write(json.dumps(exclusion.to_dict(), ensure_ascii=False) + "\n")
                        count += 1
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose": self.pose.to_dict(),
            "pose_id": self.pose.id,
            "status": self.status,
            "final": self.final,
            "approved": self.approved,
            "released_regions": self.released_regions(),
            "processed_chunks": list(self.processed_chunks),
            "pending_chunks": list(self.pending_chunks),
            "expected_chunks": list(self.expected_chunks),
            "telemetry": dict(self.telemetry),
            "issues": list(self.issues),
            "regions": {k: v.to_dict() for k, v in sorted(self.regions.items())},
        }

    def to_json(self, path: str | Path | None = None, **json_kwargs: Any) -> str:
        json_kwargs.setdefault("ensure_ascii", False)
        text = json.dumps(self.to_dict(), **json_kwargs)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


RegistrationReport = PoseEvaluation


# ---------------------------------------------------------------------------
# Exchange-record adapter
# ---------------------------------------------------------------------------


def pose_from_record(entry: Mapping[str, Any]) -> Pose:
    if "R" in entry:
        return Pose(
            id=str(entry.get("id", "pose")),
            R=entry["R"],
            translation=entry.get("translation", entry.get("t", (0, 0, 0))),
            angle_z_deg=entry.get("angle_z_deg"),
        )
    return Pose.z(
        angle_z_deg=float(entry.get("angle_z_deg", 0.0)),
        translation=entry.get("translation", entry.get("t", (0.0, 0.0, 0.0))),
        pose_id=str(entry.get("id", "pose")),
    )


def record_poses(record: Mapping[str, Any]) -> dict[str, Pose]:
    entries = record.get("pose_right_to_left", [])
    if isinstance(entries, Mapping):
        entries = [entries]
    poses = [pose_from_record(x) for x in entries]
    out = {p.id: p for p in poses}
    for i, p in enumerate(poses):
        out.setdefault(str(i), p)
    return out


def resolve_record_pose(
    record: Mapping[str, Any],
    pose: Any = None,
    pose_id: str | int | None = None,
) -> Pose:
    poses = record_poses(record)
    if isinstance(pose, Pose):
        return pose
    if isinstance(pose, Mapping):
        return pose_from_record(pose)
    if pose_id is not None:
        key = str(pose_id)
        if key not in poses:
            raise KeyError(f"unknown pose id/index {pose_id!r}; have {sorted(set(poses))}")
        return poses[key]
    if pose is not None:
        if isinstance(pose, int):
            entries = record["pose_right_to_left"]
            return pose_from_record(entries[pose])
        if isinstance(pose, str):
            return poses[pose]
        raise TypeError(f"unsupported pose selector {pose!r}")
    entries = record.get("pose_right_to_left", [])
    if isinstance(entries, list) and entries:
        return pose_from_record(entries[0])
    return Pose.z(pose_id="identity")


def frames_from_record(record: Mapping[str, Any]) -> dict[str, PointFrame]:
    views = record["views"]
    sets = record["point_sets"]
    frames: dict[str, PointFrame] = {}
    for side in ("left", "right"):
        chunks = [PointChunk.from_record(side, c) for c in sets[side]]
        frames[side] = PointFrame(side, views[side], chunks)
    return frames


def rules_from_record(record: Mapping[str, Any]) -> dict[str, RegionRule]:
    return DistanceSession._coerce_rules(record.get("region_rules", {}))


def budget_from_record(record: Mapping[str, Any]) -> MemoryBudget:
    kwargs: dict[str, Any] = {}
    if "point_chunk_max" in record:
        kwargs["max_points"] = record["point_chunk_max"]
    scale = record.get("scale") or {}
    if isinstance(scale, Mapping) and "scratch_bytes" in scale:
        kwargs["scratch_bytes"] = scale["scratch_bytes"]
    return MemoryBudget(**kwargs)


def session_from_record(
    record: Mapping[str, Any] | str | Path | None = None,
    pose: Any = None,
    pose_id: str | int | None = None,
    *,
    budget: MemoryBudget | int | Mapping[str, Any] | None = None,
    expected_chunks: Iterable[str] | None = None,
    **kwargs: Any,
) -> DistanceSession:
    if record is None or isinstance(record, (str, Path)):
        record = load_record(record or SAMPLE)  # type: ignore[arg-type]
    assert isinstance(record, Mapping)
    frames = frames_from_record(record)
    selected_pose = resolve_record_pose(record, pose=pose, pose_id=pose_id)
    rules = rules_from_record(record)
    if budget is None:
        budget = budget_from_record(record)
    if expected_chunks is None:
        expected_chunks = record.get("arrival")
    return DistanceSession(
        frames["left"],
        frames["right"],
        rules=rules,
        pose=selected_pose,
        budget=budget,
        expected_chunks=expected_chunks,
        **kwargs,
    )


def evaluate_record(
    record: Mapping[str, Any] | str | Path | None = None,
    pose: Any = None,
    pose_id: str | int | None = "p0",
    *,
    through: int | None = None,
    arrival: Sequence[str] | None = None,
    finish: bool = True,
    **kwargs: Any,
) -> PoseEvaluation:
    """Evaluate the exchange record, optionally stopping after an arrival prefix.

    ``through`` is an arrival count, not a Python slice endpoint adjustment:
    ``through=2`` processes ``arrival[:2]`` and returns a partial report.
    """
    if record is None or isinstance(record, (str, Path)):
        record = load_record(record or SAMPLE)  # type: ignore[arg-type]
    assert isinstance(record, Mapping)
    session = session_from_record(record, pose=pose, pose_id=pose_id, **kwargs)
    order = list(arrival if arrival is not None else record.get("arrival", []))
    if through is not None:
        order = order[: max(0, int(through))]
    for cid in order:
        session.ingest(cid)
    if finish and through is None:
        return session.finish()
    return session.snapshot()


# ---------------------------------------------------------------------------
# Fixed graph and local derivatives
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class FixedNeighborGraph:
    """A frozen bidirectional nearest-neighbour graph for released regions."""

    pose: Pose
    edges: list[Match]
    regions: list[str]
    source_status: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_evaluation(
        cls,
        evaluation: PoseEvaluation,
        regions: Sequence[str] | None = None,
        *,
        only_released: bool = True,
        require_released: bool | None = None,
        directions: Sequence[str] = ("left_to_right", "right_to_left"),
    ) -> "FixedNeighborGraph":
        if require_released is not None:
            only_released = require_released
        if regions is None:
            region_names = evaluation.released_regions() if only_released else list(evaluation.regions)
        else:
            region_names = [str(r) for r in regions]
            if only_released:
                missing = [r for r in region_names if not evaluation.regions[r].released]
                if missing:
                    raise ValueError(f"regions are not released/final: {missing}")

        edges: list[Match] = []
        status: dict[str, str] = {}
        for region in region_names:
            report = evaluation.regions[region]
            status[region] = report.status
            if only_released and not report.released:
                continue
            for d in directions:
                dr = report.directions[d]
                edges.extend(dr.matches)
        return cls(pose=evaluation.pose, edges=edges, regions=list(region_names), source_status=status)

    def unstable_edges(self) -> list[Match]:
        return [e for e in self.edges if not e.stable]

    def __len__(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class GradientSource:
    region: str
    direction: str
    query_id: Any
    neighbor_id: Any
    query_side: str
    neighbor_side: str
    distance_sq: float
    edge_weight: float
    loss_derivative_wrt_distance_sq: float
    query_point_gradient: list[float]
    neighbor_point_gradient: list[float]
    translation_gradient: list[float]
    angle_z_rad_gradient: float
    stable: bool
    stability_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "direction": self.direction,
            "query_id": self.query_id,
            "neighbor_id": self.neighbor_id,
            "query_side": self.query_side,
            "neighbor_side": self.neighbor_side,
            "distance_sq": self.distance_sq,
            "edge_weight": self.edge_weight,
            "loss_derivative_wrt_distance_sq": self.loss_derivative_wrt_distance_sq,
            "query_point_gradient": list(self.query_point_gradient),
            "neighbor_point_gradient": list(self.neighbor_point_gradient),
            "translation_gradient": list(self.translation_gradient),
            "angle_z_rad_gradient": self.angle_z_rad_gradient,
            "stable": self.stable,
            "stability_reasons": list(self.stability_reasons),
        }


@dataclass(eq=False)
class GradientResult:
    loss: float
    reduction: str
    weight_mode: str
    edge_count: int
    point_gradients: dict[str, dict[Any, np.ndarray]]
    pose_gradient: dict[str, Any]
    sources: list[GradientSource]
    unstable_supports: list[dict[str, Any]]
    stable: bool
    trainable: bool
    graph: FixedNeighborGraph | None = None

    def raise_if_unstable(self) -> None:
        if self.unstable_supports:
            raise UnstableSupportError(self.unstable_supports)
        if self.edge_count == 0:
            raise UnstableSupportError([{"reason": "empty_fixed_graph"}])

    def gradients_for_ids(self, side: str, ids: Sequence[Any]) -> np.ndarray:
        """Return gradients in the caller's original ID/row order."""
        table = self.point_gradients[side]
        return np.stack([
            table.get(pid, np.zeros(3, dtype=float)) for pid in ids
        ], axis=0)

    def chunk_gradients(self, chunk: "PointChunk") -> np.ndarray:
        return self.gradients_for_ids(chunk.side, chunk.ids)

    def frame_gradients(self, frame: "PointFrame") -> dict[str, np.ndarray]:
        """Map each loaded chunk ID to row-order gradients for that frame."""
        out = {}
        for cid, chunk in frame.chunks.items():
            ids = getattr(chunk, "ids", None)
            if ids is not None:
                out[cid] = self.chunk_gradients(chunk)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "reduction": self.reduction,
            "weight_mode": self.weight_mode,
            "edge_count": self.edge_count,
            "stable": self.stable,
            "trainable": self.trainable,
            "unstable_supports": self.unstable_supports,
            "pose_gradient": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.pose_gradient.items()
            },
            "point_gradients": {
                side: {str(pid): grad.tolist() for pid, grad in vals.items()}
                for side, vals in self.point_gradients.items()
            },
            "sources": [s.to_dict() for s in self.sources],
            "regions": [] if self.graph is None else self.graph.regions,
        }


def _edge_weight(edge: Match, mode: str) -> float:
    if mode == "query":
        return max(0.0, float(edge.query_weight))
    if mode == "query_neighbor" or mode == "product":
        return max(0.0, float(edge.query_weight) * float(edge.neighbor_weight))
    if mode in {"none", "uniform"}:
        return 1.0
    raise ValueError("weight_mode must be 'query', 'query_neighbor', or 'none'")


def distance_loss_and_gradients(
    graph_or_evaluation: FixedNeighborGraph | PoseEvaluation,
    *,
    regions: Sequence[str] | None = None,
    only_released: bool = True,
    reduction: str = "weighted_mean",
    weight_mode: str = "query",
    require_stable: bool = False,
    strict: bool | None = None,
    allow_unstable: bool | None = None,
    pose: Pose | None = None,
) -> GradientResult:
    """Local derivatives of a weighted squared-distance loss.

    The graph is fixed: candidate changes caused by perturbing coordinates are
    not searched again.  Every edge contributes to its original left/right
    point and to the pose.  Contributions are aggregated by stable point ID.

    The default returns the selected-branch gradient with unstable supports
    explicitly flagged.  Pass ``require_stable=True`` (or
    ``allow_unstable=False``) to raise instead.
    """
    if allow_unstable is False:
        require_stable = True
    if strict is not None:
        require_stable = bool(strict)
    if isinstance(graph_or_evaluation, PoseEvaluation):
        graph = FixedNeighborGraph.from_evaluation(
            graph_or_evaluation, regions=regions, only_released=only_released
        )
    elif isinstance(graph_or_evaluation, FixedNeighborGraph):
        graph = graph_or_evaluation
    else:
        raise TypeError("expected PoseEvaluation or FixedNeighborGraph")
    if pose is not None:
        graph.pose = pose

    edges = list(graph.edges)
    raw_weights = np.array([_edge_weight(e, weight_mode) for e in edges], dtype=float)
    if reduction in {"weighted_mean", "weighted_average"}:
        denom = float(raw_weights.sum())
        if denom <= 0:
            coefficients = np.zeros(len(edges), dtype=float)
        else:
            coefficients = raw_weights / denom
        loss = float(np.sum(coefficients * np.array([e.distance_sq for e in edges])))
    elif reduction in {"sum", "weighted_sum"}:
        coefficients = raw_weights.copy()
        loss = float(np.sum(coefficients * np.array([e.distance_sq for e in edges])))
    elif reduction in {"mean", "average", "unweighted_mean"}:
        coefficients = np.full(len(edges), 1.0 / len(edges)) if edges else np.zeros(0)
        loss = float(np.mean([e.distance_sq for e in edges])) if edges else 0.0
    else:
        raise ValueError(f"unsupported reduction {reduction!r}")

    point_grads: dict[str, dict[Any, np.ndarray]] = {"left": {}, "right": {}}
    pose_t = np.zeros(3, dtype=float)
    pose_theta = 0.0
    pose_rotation_left = np.zeros(3, dtype=float)
    sources: list[GradientSource] = []
    unstable: list[dict[str, Any]] = []

    def add_point(side: str, pid: Any, value: np.ndarray) -> None:
        if pid not in point_grads[side]:
            point_grads[side][pid] = np.zeros(3, dtype=float)
        point_grads[side][pid] += value

    for edge, coef in zip(edges, coefficients):
        g2 = float(coef)  # d loss / d d2
        p_left = (
            edge.query_xyz_local if edge.query_side == "left" else edge.neighbor_xyz_local
        )
        p_right = (
            edge.query_xyz_local if edge.query_side == "right" else edge.neighbor_xyz_local
        )
        p_left = np.asarray(p_left, dtype=float).reshape(3)
        p_right = np.asarray(p_right, dtype=float).reshape(3)
        q = graph.pose.transform(p_right)
        delta = q - p_left
        # Recompute guards against a serialized/modified edge d2 mismatch.
        d2 = float(delta @ delta)
        if not math.isclose(d2, float(edge.distance_sq), rel_tol=1e-9, abs_tol=1e-9):
            warnings.warn(
                f"edge {edge.direction}:{edge.query_id}->{edge.neighbor_id} distance "
                "does not match stored local coordinates",
                RuntimeWarning,
                stacklevel=2,
            )

        grad_left_point = -2.0 * g2 * delta
        grad_q = 2.0 * g2 * delta
        grad_right_point = grad_q @ graph.pose.R
        grad_t = grad_q.copy()
        dq_dtheta = graph.pose.rotation_z_derivative_left(q)[0]
        grad_theta = float(grad_q @ dq_dtheta)
        # Spatial Z torque uses the rotated lever arm q - t, not q itself.
        rotated_q = q - graph.pose.translation
        grad_rotation_left = 2.0 * g2 * np.cross(rotated_q, delta)

        if edge.query_side == "left":
            query_grad, neighbor_grad = grad_left_point, grad_right_point
        else:
            query_grad, neighbor_grad = grad_right_point, grad_left_point
        add_point(edge.query_side, edge.query_id, query_grad)
        add_point(edge.neighbor_side, edge.neighbor_id, neighbor_grad)
        pose_t += grad_t
        pose_theta += grad_theta
        pose_rotation_left += grad_rotation_left

        reasons = edge.stability_reasons
        if reasons:
            unstable.append(edge.to_dict())
        sources.append(
            GradientSource(
                region=edge.region,
                direction=edge.direction,
                query_id=edge.query_id,
                neighbor_id=edge.neighbor_id,
                query_side=edge.query_side,
                neighbor_side=edge.neighbor_side,
                distance_sq=float(edge.distance_sq),
                edge_weight=float(_edge_weight(edge, weight_mode)),
                loss_derivative_wrt_distance_sq=g2,
                query_point_gradient=query_grad.tolist(),
                neighbor_point_gradient=neighbor_grad.tolist(),
                translation_gradient=grad_t.tolist(),
                angle_z_rad_gradient=grad_theta,
                stable=edge.stable,
                stability_reasons=reasons,
            )
        )

    theta_deg = pose_theta * math.pi / 180.0
    stable = not unstable and bool(edges)
    result = GradientResult(
        loss=float(loss),
        reduction=reduction,
        weight_mode=weight_mode,
        edge_count=len(edges),
        point_gradients=point_grads,
        pose_gradient={
            "translation": pose_t,
            "t": pose_t,
            "angle_z_rad": pose_theta,
            "angle_z": pose_theta,
            "angle_z_deg": theta_deg,
            "rotation_left_frame": pose_rotation_left,
        },
        sources=sources,
        unstable_supports=unstable,
        stable=stable,
        trainable=stable,
        graph=graph,
    )
    if require_stable:
        result.raise_if_unstable()
    return result


# Conventional training aliases.
distance_loss_backward = distance_loss_and_gradients
backward = distance_loss_and_gradients


def training_update_plan(
    evaluation: PoseEvaluation,
    *,
    allow_unstable: bool = False,
    **gradient_kwargs: Any,
) -> GradientResult:
    """Training entry point which refuses unstable supports unless acknowledged."""
    return distance_loss_and_gradients(
        evaluation,
        require_stable=not allow_unstable,
        **gradient_kwargs,
    )

__all__ = [
    "EPS", "BOUNDARY_EPS",
    "DistanceKernelError", "ChunkBudgetError", "RegionConflictError", "UnstableSupportError",
    "MemoryBudget",
    "as_box", "points_in_box", "box_boundary_mask", "squared_distance_to_box",
    "bbox_lower_bound_sq", "should_skip_chunk",
    "common_visibility_mask", "normal_compatibility_mask",
    "Pose", "RigidPose", "make_pose",
    "RegionRule", "PointChunk", "PointCloudChunk", "ChunkHandle", "LazyChunk",
    "PointFrame", "Frame", "ScanFrame",
    "Match", "NearestMatch", "BidirectionalMatch", "Exclusion", "ExclusionRecheck",
    "weighted_quantile",
    "DirectionReport", "RegionReport", "PoseEvaluation", "RegistrationReport",
    "DistanceSession", "DistanceEngine", "PointSetDistance",
    "BidirectionalDistanceOperator", "StreamingRegistrar", "RegistrationEngine",
    "RegistrationDistanceSession", "PoseDistanceEvaluator", "ChunkedDistanceOperator",
    "OutOfCoreDistanceOperator",
    "bidirectional_nearest_neighbors", "compute_distances", "point_set_distances",
    "pose_from_record", "record_poses", "resolve_record_pose",
    "frames_from_record", "rules_from_record", "budget_from_record",
    "session_from_record", "evaluate_record",
    "FixedNeighborGraph", "GradientSource", "GradientResult",
    "distance_loss_and_gradients", "distance_loss_backward", "backward",
    "training_update_plan",
]


# ---------------------------------------------------------------------------
# Streaming bidirectional operator
# ---------------------------------------------------------------------------


_DIRECTIONS = (
    ("left_to_right", "left", "right"),
    ("right_to_left", "right", "left"),
)


@dataclass
class _UniquePoint:
    point_id: Any
    side: str
    region: str
    xyz: np.ndarray
    normals: np.ndarray
    unit_normal: np.ndarray
    weight: float
    chunks: list[str] = field(default_factory=list)
    conflict: dict[str, Any] | None = None


class DistanceSession:
    """Pose-bound streaming nearest-neighbour session.

    Create a new session for every pose.  Visibility, candidate legality and
    release eligibility are never reused across poses.
    """

    def __init__(
        self,
        left: PointFrame | Sequence[PointChunk],
        right: PointFrame | Sequence[PointChunk],
        rules: Mapping[str, RegionRule | Sequence[float]] | None = None,
        pose: Pose | Mapping[str, Any] | None = None,
        *,
        left_view: Any = None,
        right_view: Any = None,
        budget: MemoryBudget | int | Mapping[str, Any] | None = None,
        expected_chunks: Iterable[str] | None = None,
        region_expected_chunks: Mapping[str, Iterable[str]] | None = None,
        tie_tol: float = 1e-12,
        normal_tol: float = 1e-12,
        boundary_tol: float = BOUNDARY_EPS,
        visibility_mode: str = "mutual",
        strict_conflicts: bool = False,
    ) -> None:
        self.pose = make_pose(pose)
        self.left = self._coerce_frame("left", left, left_view)
        self.right = self._coerce_frame("right", right, right_view)
        self.frames = {"left": self.left, "right": self.right}
        self.rules = self._coerce_rules(rules or {})
        self.budget = self._coerce_budget(budget)
        self.tie_tol = float(tie_tol)
        self.normal_tol = float(normal_tol)
        self.boundary_tol = float(boundary_tol)
        if visibility_mode not in {"mutual", "own"}:
            raise ValueError("visibility_mode must be 'mutual' or 'own'")
        self.visibility_mode = visibility_mode
        self.strict_conflicts = strict_conflicts

        self.points: dict[str, dict[Any, _UniquePoint]] = {"left": {}, "right": {}}
        self.states: dict[str, dict[Any, Match]] = {d: {} for d, _, _ in _DIRECTIONS}
        self.processed: dict[str, set[str]] = {"left": set(), "right": set()}
        self.processed_order: list[str] = []
        self.exclusions: list[Exclusion] = []
        self.issues: list[dict[str, Any]] = []
        self.conflicts: list[dict[str, Any]] = []
        self.conflicted_regions: set[str] = set()
        self.finalized = False
        self._visibility_cache: dict[str, dict[str, np.ndarray]] = {}
        self.telemetry: dict[str, Any] = {
            "candidate_distance_values": 0,
            "chunk_pairs": 0,
            "bbox_skips": 0,
            "largest_scratch_bytes": 0,
            "full_distance_matrices_materialized": 0,
        }

        all_expected = [c.chunk_id for c in self.left.chunks.values()] + [
            c.chunk_id for c in self.right.chunks.values()
        ]
        self.expected_chunks = list(expected_chunks) if expected_chunks is not None else all_expected
        self.region_expected_chunks = {
            str(r): list(cids) for r, cids in (region_expected_chunks or {}).items()
        }
        self.manually_completed_regions: set[str] = set()
        missing = set(self.expected_chunks).difference(all_expected)
        if missing:
            # A source may provide chunks later via ingest_chunk; keep names.
            self._unknown_expected = set(missing)
        else:
            self._unknown_expected = set()

    @staticmethod
    def _coerce_frame(name: str, value: Any, view: Any) -> PointFrame:
        if isinstance(value, PointFrame):
            if view is not None:
                value.view = as_box(view)
            return value
        if isinstance(value, np.ndarray):
            if view is None:
                raise ValueError(f"{name}_view is required for array input")
            n = len(value)
            chunk = PointChunk(
                chunk_id=f"{name}/0",
                side=name,
                ids=[f"{name}{i}" for i in range(n)],
                regions=["r"] * n,
                xyz=np.asarray(value, dtype=float),
                normals=np.tile([0.0, 0.0, 1.0], (n, 1)),
                weights=np.ones(n),
            )
            return PointFrame(name, view, [chunk])
        chunks = list(value)
        if view is None:
            # Try a view attribute for convenience.
            view = getattr(value, "view", None)
        if view is None:
            raise ValueError(f"{name}_view is required when chunks are not a PointFrame")
        return PointFrame(name, view, chunks)

    @staticmethod
    def _coerce_rules(rules: Mapping[str, Any]) -> dict[str, RegionRule]:
        out: dict[str, RegionRule] = {}
        for region, rule in rules.items():
            if isinstance(rule, RegionRule):
                out[str(region)] = rule
            elif isinstance(rule, Sequence):
                out[str(region)] = RegionRule.from_sequence(rule, str(region))
            elif isinstance(rule, Mapping):
                out[str(region)] = RegionRule(**rule)
            else:
                raise TypeError(f"unsupported rule for {region!r}: {rule!r}")
        return out

    @staticmethod
    def _coerce_budget(budget: Any) -> MemoryBudget:
        if budget is None:
            return MemoryBudget()
        if isinstance(budget, MemoryBudget):
            return budget
        if isinstance(budget, int):
            return MemoryBudget(max_points=budget)
        if isinstance(budget, Mapping):
            keys = {
                "point_chunk_max": "max_points",
                "max_points": "max_points",
                "scratch_bytes": "scratch_bytes",
            }
            kwargs = {}
            for k, v in budget.items():
                if k in keys:
                    kwargs[keys[k]] = v
            return MemoryBudget(**kwargs)
        raise TypeError("unsupported budget")

    # ----- ingestion and duplicate handling -------------------------------

    def ingest(self, chunk: str | PointChunk, side: str | None = None) -> "DistanceSession":
        if isinstance(chunk, str):
            cid = chunk
            if cid in self.left.chunks:
                obj = self.left.chunk(cid)
            elif cid in self.right.chunks:
                obj = self.right.chunk(cid)
            else:
                raise KeyError(f"unknown chunk {cid!r}; use ingest_chunk for a late source chunk")
            if isinstance(obj, ChunkHandle):
                pc = obj.load()
            else:
                pc = obj
            if side is not None and pc.side != side:
                raise ValueError(f"chunk {cid!r} is on side {pc.side!r}, not {side!r}")
            return self.ingest_chunk(pc)
        if isinstance(chunk, ChunkHandle):
            return self.ingest_chunk(chunk.load())
        if side is not None and not chunk.side:
            chunk.side = side
        return self.ingest_chunk(chunk)

    def ingest_chunk(self, chunk: PointChunk) -> "DistanceSession":
        side = chunk.side
        if side not in self.frames:
            raise ValueError(f"chunk side must be 'left' or 'right', got {side!r}")
        cid = chunk.chunk_id
        # A previously unknown expected chunk has now been supplied.
        self._unknown_expected.discard(cid)
        # Replace a lazy handle with its loaded chunk for provenance/statistics.
        self.frames[side].chunks[cid] = chunk
        if cid in self.processed[side]:
            # Delivery systems can redeliver an unchanged block.  Content
            # conflicts between redeliveries are handled by point IDs below.
            return self

        self._merge_points(chunk)
        self.processed[side].add(cid)
        if cid not in self.processed_order:
            self.processed_order.append(cid)

        opposite = "right" if side == "left" else "left"
        for old_id in sorted(self.processed[opposite]):
            old_chunk = self.frames[opposite].chunk(old_id)
            left_chunk = chunk if side == "left" else old_chunk
            right_chunk = old_chunk if side == "left" else chunk
            self._process_chunk_pair(left_chunk, right_chunk)
            self.telemetry["chunk_pairs"] += 1
        return self

    def _merge_points(self, chunk: PointChunk) -> None:
        table = self.points[chunk.side]
        for i, pid in enumerate(chunk.ids):
            xyz = chunk.xyz[i].copy()
            raw_n = chunk.normals[i].copy()
            unit_n = chunk.unit_normals[i].copy()
            weight = float(chunk.weights[i])
            region = str(chunk.regions[i])
            if pid not in table:
                table[pid] = _UniquePoint(
                    point_id=pid,
                    side=chunk.side,
                    region=region,
                    xyz=xyz,
                    normals=raw_n,
                    unit_normal=unit_n,
                    weight=weight,
                    chunks=[chunk.chunk_id],
                )
                continue

            old = table[pid]
            same = (
                old.region == region
                and np.array_equal(old.xyz, xyz)
                and np.array_equal(old.normals, raw_n)
                and old.weight == weight
            )
            if chunk.chunk_id not in old.chunks:
                old.chunks.append(chunk.chunk_id)
            if same:
                continue

            conflict = {
                "type": "duplicate_content_conflict",
                "side": chunk.side,
                "point_id": pid,
                "chunks": list(dict.fromkeys(old.chunks + [chunk.chunk_id])),
                "existing": {
                    "region": old.region,
                    "xyz": old.xyz.tolist(),
                    "normal": old.normals.tolist(),
                    "weight": old.weight,
                },
                "observed": {
                    "region": region,
                    "xyz": xyz.tolist(),
                    "normal": raw_n.tolist(),
                    "weight": weight,
                },
            }
            old.conflict = conflict
            self.conflicts.append(conflict)
            self.issues.append(conflict)
            self.conflicted_regions.add(old.region)
            self.conflicted_regions.add(region)
            if self.strict_conflicts:
                raise RegionConflictError([conflict])

    def finish(self) -> PoseEvaluation:
        self.finalized = True
        return self.snapshot()

    # Readable aliases for callers expecting a compute/report vocabulary.
    def compute(self, final: bool = True) -> PoseEvaluation:
        return self.finish() if final else self.snapshot()

    run = compute

    def report(self) -> PoseEvaluation:
        return self.snapshot()

    def get_report(self) -> PoseEvaluation:
        return self.snapshot()

    def mark_region_complete(self, region: str) -> "DistanceSession":
        """Declare that no further chunks for ``region`` will arrive."""
        self.manually_completed_regions.add(str(region))
        return self

    def _region_is_complete(self, region: str) -> bool:
        if self.finalized or region in self.manually_completed_regions:
            return True
        explicit = self.region_expected_chunks.get(region)
        if explicit is not None:
            return set(explicit).issubset(set(self.processed_order))

        known_chunks: set[str] = set()
        for frame in self.frames.values():
            known_chunks.update(frame.region_chunk_ids(region))
        if not known_chunks:
            # A region named only in configuration cannot be declared complete
            # from chunk metadata until the whole arrival is finished.
            return self._is_complete()
        if not known_chunks.issubset(set(self.processed_order)):
            return False
        # Unknown expected source blocks might still contain this region.
        return not self._unknown_expected

    # ----- visibility ------------------------------------------------------

    def _visibility(self, chunk: PointChunk) -> dict[str, np.ndarray]:
        if chunk.chunk_id in self._visibility_cache:
            return self._visibility_cache[chunk.chunk_id]

        side = chunk.side
        p = chunk.xyz
        if side == "left":
            own_box = self.left.view
            other_box = self.right.view
            own_local = p
            other_local = self.pose.inverse_transform(p)
            left_xyz = p
        elif side == "right":
            own_box = self.right.view
            other_box = self.left.view
            own_local = p
            other_local = self.pose.transform(p)  # For right side this is left frame.
            left_xyz = other_local
        else:
            raise ValueError(f"bad side {side!r}")

        in_own = points_in_box(own_box, own_local)
        in_other = points_in_box(other_box, other_local)
        common = in_own & in_other if self.visibility_mode == "mutual" else in_own.copy()
        valid_weight = chunk.weights >= 0
        eligible = common & valid_weight

        boundary_own = box_boundary_mask(own_box, own_local, self.boundary_tol)
        boundary_other = box_boundary_mask(other_box, other_local, self.boundary_tol)
        boundary = boundary_own | boundary_other

        cache = {
            "in_own": in_own,
            "in_other": in_other,
            "common": common,
            "eligible": eligible,
            "boundary": boundary,
            "boundary_own": boundary_own,
            "boundary_other": boundary_other,
            "left_xyz": left_xyz,
        }
        self._visibility_cache[chunk.chunk_id] = cache
        return cache

    def _boundary_reasons(self, chunk: PointChunk, row: int) -> list[str]:
        vis = self._visibility(chunk)
        reasons: list[str] = []
        p = chunk.xyz[row]
        if chunk.side == "left":
            own_name, own_box, own_p = "left_view", self.left.view, p
            other_name = "right_view_transformed"
            other_box = self.right.view
            other_p = self.pose.inverse_transform(p)
        else:
            own_name, own_box, own_p = "right_view", self.right.view, p
            other_name = "left_view"
            other_box = self.left.view
            other_p = self.pose.transform(p)
        for axis, value, lo, hi in zip("xyz", own_p, own_box[0], own_box[1]):
            if abs(value - lo) <= self.boundary_tol:
                reasons.append(f"{own_name}:{axis}=min")
            elif abs(value - hi) <= self.boundary_tol:
                reasons.append(f"{own_name}:{axis}=max")
        for axis, value, lo, hi in zip("xyz", other_p, other_box[0], other_box[1]):
            if abs(value - lo) <= self.boundary_tol:
                reasons.append(f"{other_name}:{axis}=min")
            elif abs(value - hi) <= self.boundary_tol:
                reasons.append(f"{other_name}:{axis}=max")
        return reasons or (["visibility_boundary"] if vis["boundary"][row] else [])

    # ----- pair evaluation -------------------------------------------------

    def _process_chunk_pair(self, left_chunk: PointChunk, right_chunk: PointChunk) -> None:
        if left_chunk.side != "left" or right_chunk.side != "right":
            raise ValueError("_process_chunk_pair expects left then right")
        lvis = self._visibility(left_chunk)
        rvis = self._visibility(right_chunk)
        # Transform right chunk once per bounded batch rather than materializing
        # an N_left x N_right matrix.
        for direction, q_side, c_side in _DIRECTIONS:
            if q_side == "left":
                q_chunk, c_chunk = left_chunk, right_chunk
                q_vis, c_vis = lvis, rvis
            else:
                q_chunk, c_chunk = right_chunk, left_chunk
                q_vis, c_vis = rvis, lvis
            self._process_direction(
                direction, q_chunk, c_chunk, q_vis, c_vis
            )

    def _query_left_coords(self, chunk: PointChunk, row: int, vis: Mapping[str, np.ndarray]) -> np.ndarray:
        if chunk.side == "left":
            return chunk.xyz[row]
        return vis["left_xyz"][row]

    def _candidate_left_block(
        self, chunk: PointChunk, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz = chunk.xyz[rows]
        normals = chunk.unit_normals[rows]
        if chunk.side == "right":
            return self.pose.transform(xyz), self.pose.transform_direction(normals)
        return xyz, normals

    def _query_normal_left(self, chunk: PointChunk, row: int) -> np.ndarray:
        n = chunk.unit_normals[row]
        if chunk.side == "right":
            return self.pose.transform_direction(n.reshape(1, 3))[0]
        return n

    def _incumbent_left_coords(self, point: _UniquePoint) -> np.ndarray:
        if point.side == "left":
            return point.xyz
        return self.pose.transform(point.xyz)

    def _process_direction(
        self,
        direction: str,
        q_chunk: PointChunk,
        c_chunk: PointChunk,
        q_vis: Mapping[str, np.ndarray],
        c_vis: Mapping[str, np.ndarray],
    ) -> None:
        q_side, c_side = q_chunk.side, c_chunk.side
        states = self.states[direction]
        rule_cache: dict[str, RegionRule] = {}

        # Region masks are processed within one chunk only; no whole-cloud copy.
        for region in sorted(set(q_chunk.regions.tolist()) & set(c_chunk.regions.tolist())):
            rule = self.rules.get(region)
            rule_cache[region] = rule or RegionRule(min_normal_dot=-2.0)
            q_rows = np.where((q_chunk.regions == region) & q_vis["eligible"])[0]
            c_rows_all = np.where((c_chunk.regions == region) & c_vis["eligible"])[0]
            if q_rows.size == 0:
                continue

            for q_row in q_rows:
                qid = q_chunk.ids[int(q_row)]
                q_point = self.points[q_side].get(qid)
                if q_point is None:
                    continue
                # Conflicted regions are reported as rejected; do not let a
                # later clean copy silently rehabilitate them.
                if region in self.conflicted_regions:
                    continue

                q_left = self._query_left_coords(q_chunk, int(q_row), q_vis)
                q_normal_left = self._query_normal_left(q_chunk, int(q_row))
                state = states.get(qid)

                # Bbox branch-and-bound before reading/scanning candidate rows.
                if c_rows_all.size:
                    if c_side == "right":
                        q_in_candidate_frame = self.pose.inverse_transform(q_left)
                    else:
                        q_in_candidate_frame = q_left
                    skip, lower, criterion = should_skip_chunk(
                        q_in_candidate_frame,
                        c_chunk.bbox,
                        None if state is None else state.distance_sq,
                        tol=self.tie_tol,
                    )
                    if skip and state is not None:
                        exclusion = Exclusion(
                            query_id=qid,
                            direction=direction,
                            region=region,
                            chunk_id=c_chunk.chunk_id,
                            pose_id=self.pose.id,
                            lower_bound_sq=float(lower),
                            incumbent_id=state.neighbor_id,
                            incumbent_distance_sq=state.distance_sq,
                            query_xyz_local=q_point.xyz.tolist(),
                            candidate_bbox=c_chunk.bbox.tolist(),
                            criterion=criterion,
                            incumbent_xyz_local=state.neighbor_xyz_local.tolist(),
                        )
                        self.exclusions.append(exclusion)
                        self.telemetry["bbox_skips"] += 1
                        continue

                if c_rows_all.size == 0:
                    continue

                block_size = self.budget.candidate_block_size(1, bytes_per_candidate=256)
                for start in range(0, c_rows_all.size, block_size):
                    rows = c_rows_all[start : start + block_size]
                    cand_left, cand_normals_left = self._candidate_left_block(c_chunk, rows)
                    scratch_bytes = int(rows.size * 256)
                    self.telemetry["candidate_distance_values"] += int(rows.size)
                    self.telemetry["largest_scratch_bytes"] = max(
                        int(self.telemetry["largest_scratch_bytes"]), scratch_bytes
                    )

                    delta = cand_left - q_left[None, :]
                    d2 = np.einsum("ij,ij->i", delta, delta)
                    d2 = np.maximum(d2, 0.0)
                    dots = cand_normals_left @ q_normal_left
                    min_dot = rule_cache[region].min_normal_dot
                    legal = dots >= (min_dot - self.normal_tol)
                    if not np.any(legal):
                        continue

                    legal_local_pos = np.where(legal)[0]
                    legal_d2 = d2[legal_local_pos]
                    min_d = float(np.min(legal_d2))
                    tied_positions = np.where(
                        np.abs(legal_d2 - min_d) <= self.tie_tol * max(1.0, min_d)
                    )[0]
                    winner_local = min(
                        (int(legal_local_pos[int(j)]) for j in tied_positions),
                        key=lambda pos: self._id_sort_key(c_chunk.ids[int(rows[pos])]),
                    )
                    winner_row = int(rows[winner_local])
                    winner_id = c_chunk.ids[winner_row]
                    winner_d2 = float(d2[winner_local])
                    winner_dot = float(dots[winner_local])
                    batch_tie = tied_positions.size > 1

                    cand_point = self.points[c_side].get(winner_id)
                    if cand_point is None:
                        continue

                    is_new = state is None
                    is_better = (not is_new) and winner_d2 < state.distance_sq - self.tie_tol * max(
                        1.0, winner_d2, state.distance_sq
                    )
                    is_tied = (not is_new) and abs(winner_d2 - state.distance_sq) <= self.tie_tol * max(
                        1.0, winner_d2, state.distance_sq
                    )

                    if is_new or is_better:
                        selected_id = winner_id
                        selected_d2 = winner_d2
                        selected_dot = winner_dot
                        equidistant = batch_tie
                    elif is_tied:
                        # Equal distances to different stable IDs are ambiguous.
                        # A repeated identical copy of the incumbent ID is not.
                        different_id = self._id_sort_key(winner_id) != self._id_sort_key(state.neighbor_id)
                        equidistant = bool(batch_tie or state.equidistant or different_id)
                        # Stable ID decides the independent nearest-neighbour tie.
                        if self._id_sort_key(winner_id) < self._id_sort_key(state.neighbor_id):
                            selected_id = winner_id
                            selected_d2 = winner_d2
                            selected_dot = winner_dot
                        else:
                            selected_id = state.neighbor_id
                            selected_d2 = state.distance_sq
                            selected_dot = state.normal_dot
                    else:
                        continue

                    selected_point = (
                        cand_point
                        if selected_id == winner_id
                        else self.points[c_side][selected_id]
                    )
                    q_boundary = bool(q_vis["boundary"][int(q_row)])
                    c_boundary = bool(c_vis["boundary"][winner_row])
                    if selected_id != winner_id:
                        # The retained incumbent may itself have a visibility
                        # boundary even though the tied block winner does not.
                        try:
                            incumbent_chunk = self._first_processed_chunk(selected_point)
                            if incumbent_chunk is not None:
                                incumbent_row = incumbent_chunk.row_by_id(selected_id)
                                incumbent_vis = self._visibility(incumbent_chunk)
                                c_boundary = c_boundary or bool(incumbent_vis["boundary"][incumbent_row])
                        except (KeyError, ValueError):
                            pass
                    normal_boundary = abs(selected_dot - min_dot) <= max(
                        self.boundary_tol, self.normal_tol * 10
                    )
                    reasons: list[str] = []
                    if q_boundary:
                        reasons.extend(self._boundary_reasons(q_chunk, int(q_row)))
                    if c_boundary and selected_id == winner_id:
                        reasons.extend(self._boundary_reasons(c_chunk, winner_row))
                    elif c_boundary:
                        incumbent_chunk = self._first_processed_chunk(selected_point)
                        if incumbent_chunk is not None:
                            incumbent_row = incumbent_chunk.row_by_id(selected_id)
                            reasons.extend(self._boundary_reasons(incumbent_chunk, incumbent_row))

                    new_match = Match(
                        query_id=qid,
                        neighbor_id=selected_id,
                        region=region,
                        direction=direction,
                        distance_sq=float(selected_d2),
                        query_weight=float(q_point.weight),
                        normal_dot=float(selected_dot),
                        query_side=q_side,
                        neighbor_side=c_side,
                        query_xyz_local=q_point.xyz.copy(),
                        neighbor_xyz_local=selected_point.xyz.copy(),
                        query_chunk_ids=list(q_point.chunks),
                        neighbor_chunk_ids=list(selected_point.chunks),
                        neighbor_weight=float(selected_point.weight),
                        equidistant=bool(equidistant),
                        normal_boundary=bool(normal_boundary),
                        visibility_boundary=bool(q_boundary or c_boundary),
                        boundary_reasons=sorted(set(reasons)),
                    )
                    states[qid] = new_match

    @staticmethod
    def _id_sort_key(point_id: Any) -> tuple[Any, ...]:
        # Stable, deterministic ordering.  Same-type IDs compare naturally;
        # mixed input gets a deterministic type prefix without TypeError.
        if isinstance(point_id, str):
            return ("str", point_id)
        if isinstance(point_id, (int, float, bool)):
            return (type(point_id).__name__, point_id)
        return (type(point_id).__name__, str(point_id))

    # ----- reporting -------------------------------------------------------

    def _is_complete(self) -> bool:
        if self.finalized:
            return True
        return set(self.expected_chunks).issubset(set(self.processed_order)) and not self._unknown_expected

    def snapshot(self) -> PoseEvaluation:
        complete = self._is_complete()
        overall_status = "final" if complete else "partial"
        all_regions = set(self.rules)
        for side in ("left", "right"):
            all_regions.update(p.region for p in self.points[side].values())
            for chunk in self.frames[side].chunks.values():
                regions = getattr(chunk, "regions", None)
                if regions is not None:
                    try:
                        all_regions.update(str(r) for r in np.asarray(regions, dtype=object).tolist())
                    except Exception:
                        pass

        region_reports: dict[str, RegionReport] = {}
        for region in sorted(all_regions):
            region_reports[region] = self._region_report(region)

        required = set(self.rules)
        required_final = all(self._region_is_complete(r) for r in required)
        approved = bool(required) and required_final and all(
            region_reports[r].released for r in required
        )
        if self.conflicted_regions & required:
            approved = False

        processed = list(self.processed_order)
        expected = list(self.expected_chunks)
        pending = [x for x in expected if x not in set(processed)]
        telemetry = dict(self.telemetry)
        telemetry["unique_points"] = {
            side: len(table) for side, table in self.points.items()
        }
        return PoseEvaluation(
            pose=self.pose,
            regions=region_reports,
            status=overall_status,
            approved=bool(approved),
            processed_chunks=processed,
            pending_chunks=pending,
            expected_chunks=expected,
            telemetry=telemetry,
            issues=list(self.issues),
        )

    def _region_report(self, region: str) -> RegionReport:
        complete = self._region_is_complete(region)
        rule = self.rules.get(region)
        issues = [x for x in self.issues if isinstance(x, Mapping) and (
            x.get("type") != "duplicate_content_conflict"
            or region in x.get("chunks", [])  # broad fallback
            or x.get("existing", {}).get("region") == region
            or x.get("observed", {}).get("region") == region
        )]
        valid = region not in self.conflicted_regions

        frames_info: dict[str, Any] = {}
        assessable = False
        for side, frame in self.frames.items():
            region_chunks = frame.region_chunk_ids(region)
            processed_here = sorted(self.processed[side])
            frames_info[side] = {
                "name": frame.name,
                "view": frame.view.tolist(),
                "region_chunks": region_chunks,
                "processed_region_chunks": [c for c in region_chunks if c in processed_here],
                "pending_region_chunks": [c for c in region_chunks if c not in processed_here],
                "all_chunks": list(frame.chunks),
            }

        directions: dict[str, DirectionReport] = {}
        region_assessable = False
        for direction, q_side, c_side in _DIRECTIONS:
            report = self._direction_report(direction, q_side, c_side, region, rule, complete)
            directions[direction] = report
            if report.total_weight > 0:
                region_assessable = True
            assessable = assessable or report.total_weight > 0

        any_match = any(d.count_matched > 0 for d in directions.values())
        total_query_weight = sum(d.total_weight for d in directions.values())
        missing_evidence = bool(complete and valid and region_assessable and not any_match)
        temporary_no_neighbor = bool((not complete) and region_assessable and not any_match)
        if not complete:
            status = "partial"
        elif not valid:
            status = "rejected"
        elif not region_assessable:
            status = "not_visible"
        elif missing_evidence:
            status = "missing_evidence"
        else:
            status = "final"

        released = bool(
            complete
            and valid
            and region_assessable
            and all(d.ok for d in directions.values())
        )
        return RegionReport(
            region=region,
            pose_id=self.pose.id,
            status=status,
            valid=valid,
            released=released,
            assessable=bool(region_assessable),
            missing_evidence=missing_evidence,
            temporary_no_neighbor=temporary_no_neighbor,
            frames=frames_info,
            directions=directions,
            rule=rule,
            issues=issues,
        )

    def _direction_report(
        self,
        direction: str,
        q_side: str,
        c_side: str,
        region: str,
        rule: RegionRule | None,
        complete: bool,
    ) -> DirectionReport:
        # Denominator: unique, common-visible/valid query points in this region.
        total_weight = 0.0
        count_query = 0
        for pid, point in self.points[q_side].items():
            if point.region != region:
                continue
            chunk = self._first_processed_chunk(point)
            if chunk is None:
                continue
            vis = self._visibility(chunk)
            row = chunk.row_by_id(pid)
            if bool(vis["eligible"][row]):
                total_weight += max(0.0, point.weight)
                count_query += 1

        matches = [m for m in self.states[direction].values() if m.region == region]
        # Defensive: states are only inserted for eligible points, but ensure
        # duplicate/conflict changes never count invalid rows.
        matches = [m for m in matches if m.query_id in self.points[q_side]]
        matched_weight = float(sum(max(0.0, m.query_weight) for m in matches))
        d2_values = np.array([m.distance_sq for m in matches], dtype=float)
        weights = np.array([m.query_weight for m in matches], dtype=float)
        q = rule.quantile if rule is not None else None
        quantile = weighted_quantile(d2_values, weights, q) if q is not None else None
        mean = None
        max_d2 = None
        if matches and weights.sum() > 0:
            mean = float(np.sum(d2_values * weights) / np.sum(weights))
            max_d2 = float(np.max(d2_values))

        coverage = matched_weight / total_weight if total_weight > 0 else 0.0
        min_cov = rule.min_coverage if rule is not None else None
        max_allowed = rule.max_distance_sq if rule is not None else None
        coverage_ok = bool(rule is not None and total_weight > 0 and coverage + EPS >= rule.min_coverage)
        distance_ok = bool(
            rule is not None
            and quantile is not None
            and quantile <= rule.max_distance_sq + EPS
        )
        ok = coverage_ok and distance_ok

        exclusions = [e for e in self.exclusions if e.direction == direction and e.region == region]
        return DirectionReport(
            direction=direction,
            query_side=q_side,
            candidate_side=c_side,
            region=region,
            status="final" if complete else "partial",
            total_weight=float(total_weight),
            matched_weight=matched_weight,
            unmatched_weight=float(max(0.0, total_weight - matched_weight)),
            coverage=float(coverage),
            count_query=count_query,
            count_matched=len(matches),
            weighted_quantile_sq=quantile,
            weighted_mean_sq=mean,
            max_distance_sq=max_d2,
            min_normal_dot=None if rule is None else rule.min_normal_dot,
            min_coverage_required=min_cov,
            max_distance_sq_allowed=max_allowed,
            quantile_required=q,
            coverage_ok=coverage_ok,
            distance_ok=distance_ok,
            ok=ok,
            matches=matches,
            exclusions=exclusions,
        )

    def _first_processed_chunk(self, point: _UniquePoint) -> PointChunk | None:
        for cid in point.chunks:
            if cid in self.processed[point.side]:
                return self.frames[point.side].chunk(cid)
        return None


# ---------------------------------------------------------------------------
# Functional convenience API
# ---------------------------------------------------------------------------


def bidirectional_nearest_neighbors(
    left: PointFrame | Sequence[PointChunk],
    right: PointFrame | Sequence[PointChunk],
    pose: Pose | Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
    *,
    left_view: Any = None,
    right_view: Any = None,
    budget: MemoryBudget | int | None = None,
    expected_chunks: Iterable[str] | None = None,
    **kwargs: Any,
) -> PoseEvaluation:
    """Run the complete streaming operator and return a final report."""
    session = DistanceSession(
        left=left,
        right=right,
        rules=rules,
        pose=pose,
        left_view=left_view,
        right_view=right_view,
        budget=budget,
        expected_chunks=expected_chunks,
        **kwargs,
    )
    for cid in session.expected_chunks:
        if cid in session.left.chunks:
            session.ingest(cid, "left")
        elif cid in session.right.chunks:
            session.ingest(cid, "right")
        else:
            # Generic callers may instead pass all chunks directly; expected
            # defaults then always exist.  Keep this guard for custom IDs.
            raise KeyError(f"expected chunk {cid!r} is not available")
    return session.finish()


# Concise aliases used by some callers/tests.
compute_distances = bidirectional_nearest_neighbors
point_set_distances = bidirectional_nearest_neighbors
DistanceEngine = DistanceSession
PointSetDistance = DistanceSession
BidirectionalDistanceOperator = DistanceSession
StreamingRegistrar = DistanceSession
RegistrationEngine = DistanceSession
RegistrationDistanceSession = DistanceSession
PoseDistanceEvaluator = DistanceSession
ChunkedDistanceOperator = DistanceSession
OutOfCoreDistanceOperator = DistanceSession
PointSetDistanceOperator = DistanceSession
DistanceOperator = DistanceSession


class PointDistanceOperator:
    """Reusable scan-pair operator; each pose gets a fresh distance session."""

    def __init__(
        self,
        left: Any = None,
        right: Any = None,
        rules: Mapping[str, Any] | None = None,
        *,
        left_view: Any = None,
        right_view: Any = None,
        views: Mapping[str, Any] | None = None,
        budget: MemoryBudget | int | Mapping[str, Any] | None = None,
        expected_chunks: Iterable[str] | None = None,
        poses: Mapping[str, Pose] | None = None,
        **session_kwargs: Any,
    ) -> None:
        # Convenience: PointDistanceOperator(record_dict_or_path).
        if right is None and isinstance(left, (Mapping, str, Path)) and (
            isinstance(left, (str, Path)) or "point_sets" in left
        ):
            record = load_record(left) if isinstance(left, (str, Path)) else left
            frames = frames_from_record(record)
            rules = rules if rules is not None else rules_from_record(record)
            if budget is None:
                budget = budget_from_record(record)
            if expected_chunks is None:
                expected_chunks = record.get("arrival")
            left = frames["left"]
            right = frames["right"]
        if views is not None:
            left_view = left_view or views.get("left")
            right_view = right_view or views.get("right")
        self.left_frame = self._as_frame("left", left, left_view)
        self.right_frame = self._as_frame("right", right, right_view)
        self.rules = DistanceSession._coerce_rules(rules or {})
        self.budget = DistanceSession._coerce_budget(budget)
        self.expected_chunks = list(expected_chunks) if expected_chunks is not None else None
        self.poses = dict(poses or {})
        self.session_kwargs = dict(session_kwargs)

    @staticmethod
    def _as_frame(side: str, value: Any, view: Any) -> PointFrame:
        if isinstance(value, PointFrame):
            return value
        if isinstance(value, Mapping) and "view" in value and ("chunks" in value or "point_sets" in value):
            chunks = value.get("chunks", value.get("point_sets"))
            return PointFrame(side, value["view"], chunks)
        if isinstance(value, tuple) and len(value) == 2 and view is None:
            value, view = value
        if view is None:
            raise ValueError(f"{side}_view is required")
        if isinstance(value, np.ndarray):
            chunk = PointChunk(
                chunk_id=f"{side}/0",
                side=side,
                ids=[f"{side}{i}" for i in range(len(value))],
                regions=["r"] * len(value),
                xyz=value,
                normals=np.tile([0.0, 0.0, 1.0], (len(value), 1)),
                weights=np.ones(len(value)),
            )
            value = [chunk]
        return PointFrame(side, view, value)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any] | str | Path | None = None,
        **kwargs: Any,
    ) -> "PointDistanceOperator":
        if record is None or isinstance(record, (str, Path)):
            record = load_record(record or SAMPLE)  # type: ignore[arg-type]
        frames = frames_from_record(record)
        return cls(
            frames["left"],
            frames["right"],
            rules=rules_from_record(record),
            budget=budget_from_record(record),
            expected_chunks=record.get("arrival"),
            poses=record_poses(record),
            **kwargs,
        )

    def session(
        self,
        pose: Any = None,
        pose_id: str | int | None = None,
        **pose_kwargs: Any,
    ) -> DistanceSession:
        if pose is None and pose_id is not None:
            pose = pose_id
        if isinstance(pose, (str, int)) and self.poses:
            key = str(pose)
            if key not in self.poses:
                raise KeyError(f"unknown pose id/index {pose!r}")
            selected_pose: Pose | None = self.poses[key]
        elif isinstance(pose, Pose):
            selected_pose = pose
        else:
            selected_pose = make_pose(pose, **pose_kwargs)
        return DistanceSession(
            self.left_frame,
            self.right_frame,
            rules=self.rules,
            pose=selected_pose,
            budget=self.budget,
            expected_chunks=self.expected_chunks,
            **self.session_kwargs,
        )

    def evaluate_pose(
        self,
        pose: Any = None,
        *,
        pose_id: str | int | None = None,
        arrival: Sequence[str] | None = None,
        through: int | None = None,
        finish: bool = True,
        **pose_kwargs: Any,
    ) -> PoseEvaluation:
        session = self.session(pose, pose_id=pose_id, **pose_kwargs)
        if arrival is not None:
            order = list(arrival)
        elif self.expected_chunks is not None:
            order = list(self.expected_chunks)
        else:
            order = list(session.expected_chunks)
        if through is not None:
            order = order[: max(0, int(through))]
        for cid in order:
            session.ingest(cid)
        if finish and through is None:
            return session.finish()
        return session.snapshot()

    requery_pose = evaluate_pose
    evaluate = evaluate_pose


HausdorffKernel = PointDistanceOperator
DistanceKernel = PointDistanceOperator
RegistrationEvaluator = PointDistanceOperator


class RegistrationGate:
    """Configure and inspect the two-direction coverage/distance gate."""

    def __init__(self, rules: Mapping[str, Any]):
        self.rules = DistanceSession._coerce_rules(rules)

    def direction_allowed(self, report: DirectionReport) -> bool:
        rule = self.rules[report.region]
        return bool(
            report.total_weight > 0
            and report.coverage + EPS >= rule.min_coverage
            and report.weighted_quantile_sq is not None
            and report.weighted_quantile_sq <= rule.max_distance_sq + EPS
        )

    def region_allowed(self, report: RegionReport) -> bool:
        return bool(
            report.status == "final"
            and report.valid
            and report.assessable
            and all(self.direction_allowed(d) for d in report.directions.values())
        )

    def allowed_regions(self, evaluation: PoseEvaluation) -> list[str]:
        return [r for r, rep in evaluation.regions.items() if self.region_allowed(rep)]

    def approve(self, evaluation: PoseEvaluation) -> bool:
        return bool(self.rules) and all(
            self.region_allowed(evaluation.regions[r]) for r in self.rules
        )


def release_decision(evaluation: PoseEvaluation) -> dict[str, Any]:
    """Return a small read-only-style decision summary."""
    regions = {}
    for name, report in evaluation.regions.items():
        regions[name] = {
            "status": report.status,
            "released": report.released,
            "missing_evidence": report.missing_evidence,
            "temporary_no_neighbor": report.temporary_no_neighbor,
            "directions": {
                d: {
                    "coverage": dr.coverage,
                    "weighted_quantile_sq": dr.weighted_quantile_sq,
                    "ok": dr.ok,
                }
                for d, dr in report.directions.items()
            },
        }
    return {
        "pose_id": evaluation.pose.id,
        "status": evaluation.status,
        "approved": evaluation.approved,
        "released_regions": evaluation.released_regions(),
        "regions": regions,
    }


def region_statistics(evaluation: PoseEvaluation) -> dict[str, Any]:
    return {
        region: {
            d: {
                "coverage": rep.coverage,
                "total_weight": rep.total_weight,
                "matched_weight": rep.matched_weight,
                "weighted_quantile_sq": rep.weighted_quantile_sq,
                "weighted_mean_sq": rep.weighted_mean_sq,
                "max_distance_sq": rep.max_distance_sq,
                "count_query": rep.count_query,
                "count_matched": rep.count_matched,
            }
            for d, rep in report.directions.items()
        }
        for region, report in evaluation.regions.items()
    }


def hausdorff_distances(evaluation: PoseEvaluation) -> dict[str, dict[str, float | None]]:
    """Maximum nearest-neighbour distance (squared and root) per direction."""
    out: dict[str, dict[str, float | None]] = {}
    for region, report in evaluation.regions.items():
        per_direction: dict[str, float | None] = {}
        max_sq_values = []
        for direction, dr in report.directions.items():
            per_direction[direction + "_sq"] = dr.max_distance_sq
            per_direction[direction] = (
                None if dr.max_distance_sq is None else math.sqrt(dr.max_distance_sq)
            )
            if dr.max_distance_sq is not None:
                max_sq_values.append(dr.max_distance_sq)
        max_sq = max(max_sq_values) if max_sq_values else None
        per_direction["bidirectional_sq"] = max_sq
        per_direction["bidirectional"] = None if max_sq is None else math.sqrt(max_sq)
        out[region] = per_direction
    return out


# Verb-name aliases for differing caller vocabularies.
bidirectional_nearest_neighbor = bidirectional_nearest_neighbors
compute_bidirectional_distances = bidirectional_nearest_neighbors
distance_report = evaluate_record
evaluate_registration = evaluate_record
process_record = evaluate_record
run_record = evaluate_record
solve = evaluate_record
analyze = evaluate_record
compute_hausdorff_distance = evaluate_record
create_session = session_from_record
engine_from_record = session_from_record
weighted_quantile_distance = weighted_quantile
bbox_can_skip = should_skip_chunk
hausdorff_distance = hausdorff_distances
region_distance_statistics = region_statistics
distance_statistics = region_statistics
RegistrationReportEvaluator = PointDistanceOperator


__all__.extend([
    "PointDistanceOperator", "DistanceKernel", "HausdorffKernel", "RegistrationEvaluator",
    "RegistrationGate", "release_decision", "region_statistics", "hausdorff_distances",
    "hausdorff_distance",
    "PointSetDistanceOperator", "DistanceOperator",
])


# ---------------------------------------------------------------------------
# Small command-line summary for the bundled exchange record
# ---------------------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compute sample point-distance reports")
    parser.add_argument("path", nargs="?", type=Path, default=None, help="record JSON path")
    parser.add_argument("--pose", default="p0", help="pose id or index")
    parser.add_argument("--through", type=int, default=None, help="arrival prefix length")
    parser.add_argument("--json", action="store_true", help="print full JSON report")
    args = parser.parse_args(argv)

    pose_selector: Any = args.pose
    if str(args.pose).isdigit():
        pose_selector = int(args.pose)
    evaluation = evaluate_record(
        args.path,
        pose_id=None if isinstance(pose_selector, int) else pose_selector,
        pose=pose_selector if isinstance(pose_selector, int) else None,
        through=args.through,
    )
    if args.json:
        print(evaluation.to_json(indent=2, ensure_ascii=False))
        return 0

    print(f"pose={evaluation.pose.id} status={evaluation.status} approved={evaluation.approved}")
    for region, report in evaluation.regions.items():
        print(
            f"[{region}] status={report.status} released={report.released} "
            f"missing_evidence={report.missing_evidence}"
        )
        for direction, dr in report.directions.items():
            q = None if dr.weighted_quantile_sq is None else round(dr.weighted_quantile_sq, 6)
            print(
                f"  {direction}: coverage={dr.coverage:.3f} "
                f"matched_weight={dr.matched_weight:.3f}/{dr.total_weight:.3f} "
                f"quantile_sq={q} matches={dr.count_matched} ok={dr.ok}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

main = _main
