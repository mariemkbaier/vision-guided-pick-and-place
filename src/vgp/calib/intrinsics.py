"""Intrinsic calibration: recovering K and distortion from checkerboard views.

WHY THIS STEP EXISTS AT ALL
---------------------------
Without intrinsics, an image is a picture. With them, it is a bundle of rays
with known directions, and only then can PnP turn pixels into metres. Every
downstream number inherits this one's error, so it is worth getting right and
worth being able to say how right it is.

THE FOUR THINGS PEOPLE GET WRONG, AND THE GATES BELOW THAT CATCH THEM
---------------------------------------------------------------------

1. TOO FEW VIEWS, ALL FRONTO-PARALLEL.  This is the big one and it is subtly
   fatal. Photograph a planar board face-on and there is a near-exact ambiguity
   between focal length and distance: a board twice as far with twice the focal
   length produces nearly the same image. The optimiser finds SOME (f, Z) pair
   on that valley, reports a beautiful sub-pixel RMSE, and hands you a focal
   length that can be 20% wrong. Tilting the board breaks the degeneracy,
   because perspective foreshortening depends on f and Z differently.
   -> `tilt_span_deg` gate. A calibration of face-on views is rejected even
      when its RMSE is excellent. This is the gate that would have caught the
      most real-world bad calibrations.

2. NO CORNERS NEAR THE IMAGE EDGES.  Radial distortion is ~zero at the centre
   and grows with radius, so centre-only data cannot constrain k1/k2 -- they
   fit noise and then extrapolate wildly at the corners where you need them.
   -> `coverage` gate (convex hull of all corners / image area) plus an
      explicit check that some corners land in the outer image margin.

3. TRUSTING RMSE ALONE.  RMSE measures fit to the data you collected. With
   five distortion parameters and a handful of views you can drive it near zero
   while the model is nonsense off-sample. It is a necessary, not sufficient,
   condition -- which is exactly why the gates above are about the DATA, not
   the residual.

4. UNITS AND THE INNER-CORNER CONVENTION.  A "9x6 checkerboard" means 9x6
   INNER corners -- a board of 10x7 squares. Off by one here scales the whole
   thing. `square_size_m` is in metres, and it is the ONLY thing that sets the
   absolute scale of everything downstream: get it wrong by 10% and every
   object pose is 10% too far away, forever, with no symptom in the residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from vgp.msgs import CameraInfo, Header

# Sub-pixel refinement: stop after 30 iterations or when the corner moves less
# than 0.001 px. Refinement matters more than it sounds -- raw integer corners
# cost you roughly an order of magnitude in final pose accuracy.
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


@dataclass(frozen=True)
class CheckerboardSpec:
    """cols x rows INNER corners (a 10x7-square board is 9x6 inner)."""

    cols: int = 9
    rows: int = 6
    square_size_m: float = 0.025

    def object_points(self) -> np.ndarray:
        """3D corner positions in the board frame: planar, so Z = 0.

        Z=0 is not a simplification -- it is what makes the board a *planar*
        calibration target and lets a homography initialise the solve.
        """
        p = np.zeros((self.rows * self.cols, 3), np.float32)
        p[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        return p * self.square_size_m


@dataclass
class CalibrationResult:
    camera_info: CameraInfo
    rms: float                        # overall reprojection RMSE, pixels
    per_view_rms: np.ndarray          # per-view, to spot one bad board detection
    n_views: int
    coverage: float                   # fraction of image area spanned by corners
    edge_coverage: float              # fraction of corners in the outer margin
    tilt_span_deg: float              # spread of board orientations across views
    board: CheckerboardSpec

    # GATE THRESHOLDS -- measured, not folklore.
    #
    # Each was set by sweeping synthetic calibrations against a KNOWN K and
    # finding where recovery degrades (see docs/DESIGN_NOTES.md section 5 for
    # the tables). Reported rather than silently enforced, so a caller can
    # override with their eyes open.
    #
    # MIN_TILT_SPAN_DEG = 25:  measured fx error vs. tilt span, 5 seeds each --
    #   span  0.0 deg -> 6.7% median fx error, 7.8% worst   (RMSE 0.043 px!)
    #   span  5.2 deg -> 0.3% median,          1.0% worst
    #   span 26.1 deg -> 0.02% median,         0.08% worst
    #   Median recovers by ~6 deg, but WORST case stays erratic until ~26 deg
    #   (one seed hit 24% error at a 10 deg span). We gate on the worst case:
    #   a calibration you cannot trust to be good is not good.
    #
    # MIN_COVERAGE / MIN_EDGE_COVERAGE: fractions of occupied GRID CELLS
    #   (see grid_occupancy for why cells, not a convex hull or a corner
    #   ratio). Both are monotone in effort, so the thresholds are set where a
    #   good-faith operator lands after ~20 aspect-matched views. A checkerboard
    #   must be FULLY visible to be detected at all, so the extreme border cells
    #   are only reachable with the board close and near the edge -- which is
    #   exactly the data that constrains distortion. See `recommend_board`:
    #   a board whose aspect does not match the sensor cannot reach these at
    #   all, which is a property of the TARGET, not of the operator.
    #
    #   Measured ceiling (aspect-matched 14x7 board, 20mm squares, 960x540):
    #       20 views -> 75% cells, 54% border
    #       50 views -> 83% cells, 67% border
    #      400 views -> 100%, 100%   (so the metric does saturate, eventually)
    #   Thresholds sit just under the ~20-view figure: demanding enough that a
    #   centre-only collection fails outright, loose enough that a careful
    #   operator is not asked for 50 views. Smaller squares raise the ceiling
    #   (12mm reaches 100% in 100 views) because a smaller board can come
    #   closer to the lens and still fit.
    MIN_VIEWS = 10
    MAX_RMS_PX = 1.0
    MIN_COVERAGE = 0.70
    MIN_EDGE_COVERAGE = 0.45
    MIN_TILT_SPAN_DEG = 25.0

    def problems(self) -> list[str]:
        out = []
        if self.n_views < self.MIN_VIEWS:
            out.append(f"only {self.n_views} views (want >= {self.MIN_VIEWS})")
        if self.rms > self.MAX_RMS_PX:
            out.append(f"reprojection RMSE {self.rms:.3f} px > {self.MAX_RMS_PX} px")
        if self.coverage < self.MIN_COVERAGE:
            out.append(
                f"corners occupy only {self.coverage:.0%} of image cells "
                f"(want >= {self.MIN_COVERAGE:.0%}) -- move the board around more"
            )
        if self.edge_coverage < self.MIN_EDGE_COVERAGE:
            out.append(
                f"only {self.edge_coverage:.0%} of BORDER cells reached "
                f"(want >= {self.MIN_EDGE_COVERAGE:.0%}) -- distortion coefficients are "
                "unconstrained at the image periphery, where distortion is largest"
            )
        if self.tilt_span_deg < self.MIN_TILT_SPAN_DEG:
            out.append(
                f"board tilt varies by only {self.tilt_span_deg:.0f} deg "
                f"(want >= {self.MIN_TILT_SPAN_DEG:.0f} deg) -- focal length is "
                "nearly unidentifiable from fronto-parallel views, so fx/fy may "
                "be badly wrong DESPITE a good RMSE"
            )
        return out

    @property
    def ok(self) -> bool:
        return not self.problems()

    def summary(self) -> str:
        K = self.camera_info.K
        lines = [
            f"views          : {self.n_views}",
            f"reproj RMSE    : {self.rms:.4f} px   (worst view {self.per_view_rms.max():.4f})",
            f"fx, fy         : {K[0,0]:.2f}, {K[1,1]:.2f}",
            f"cx, cy         : {K[0,2]:.2f}, {K[1,2]:.2f}",
            f"distortion     : {np.array2string(self.camera_info.dist.ravel(), precision=4)}",
            f"cell coverage  : {self.coverage:.0%}   border cells {self.edge_coverage:.0%}",
            f"tilt span      : {self.tilt_span_deg:.1f} deg",
        ]
        probs = self.problems()
        lines.append("status         : OK" if not probs else "status         : PROBLEMS")
        lines += [f"  ! {p}" for p in probs]
        return "\n".join(lines)


# --------------------------------------------------------------- coverage

GRID_COLS, GRID_ROWS = 8, 6


def grid_occupancy(points: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Fraction of image cells, and of BORDER cells, containing >= 1 corner.

    WHY A GRID AND NOT A CONVEX HULL
    --------------------------------
    The obvious coverage metric is the convex hull area of all corners over the
    image area. It has two flaws that a grid does not:

      - A hull OVERSTATES. Corners arranged in a ring score 100% while the
        middle of the image is empty. The hull measures extent, not occupancy,
        and for calibration we care about occupancy.

      - It says nothing about WHERE the gaps are.

    WHY EDGE COVERAGE IS A CELL FRACTION AND NOT A CORNER FRACTION
    --------------------------------------------------------------
    The first version of this counted what fraction of all CORNERS fell in the
    outer margin. Measured behaviour, going from 15 to 40 views:

        views    hull coverage    edge corner-fraction
          15          52%                4.0%
          30          57%                2.3%
          40          58%                2.6%

    Coverage improved while "edge coverage" got WORSE -- because each extra
    view adds mostly central corners and dilutes the ratio. A quality metric
    that punishes collecting more data is simply wrong, and had we gated on it
    we would have coached operators to stop early.

    Counting occupied BORDER CELLS is monotone in effort: a new view can only
    ever fill a cell, never empty one. That is the property a coverage metric
    must have.
    """
    if len(points) == 0:
        return 0.0, 0.0
    cx = np.clip((points[:, 0] / width * GRID_COLS).astype(int), 0, GRID_COLS - 1)
    cy = np.clip((points[:, 1] / height * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
    occupied = set(zip(cx.tolist(), cy.tolist()))
    total = GRID_COLS * GRID_ROWS

    border = {
        (i, j)
        for i in range(GRID_COLS)
        for j in range(GRID_ROWS)
        if i in (0, GRID_COLS - 1) or j in (0, GRID_ROWS - 1)
    }
    return len(occupied) / total, len(occupied & border) / len(border)


class IntrinsicCalibrator:
    def __init__(self, board: CheckerboardSpec) -> None:
        self.board = board
        self._img_pts: list[np.ndarray] = []
        self._obj_pts: list[np.ndarray] = []
        self._size: tuple[int, int] | None = None

    def add_view(self, image_rgb: np.ndarray) -> bool:
        """Detect and sub-pixel-refine the board. Returns False if not found."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        if self._size is None:
            self._size = (w, h)
        elif self._size != (w, h):
            # Mixing resolutions would silently average two different K's.
            raise ValueError(f"view is {w}x{h}, earlier views were {self._size}")

        # ADAPTIVE_THRESH + NORMALIZE_IMAGE make detection robust to the uneven
        # lighting you always get from a handheld board under a room light.
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(
            gray, (self.board.cols, self.board.rows), flags
        )
        if not found:
            return False
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
        self._img_pts.append(corners)
        self._obj_pts.append(self.board.object_points())
        return True

    @property
    def n_views(self) -> int:
        return len(self._img_pts)

    def calibrate(self) -> CalibrationResult:
        if self.n_views < 3:
            raise ValueError(f"need at least 3 views to calibrate, have {self.n_views}")
        w, h = self._size
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._obj_pts, self._img_pts, (w, h), None, None
        )

        # Per-view residuals: one badly-detected board (motion blur, a partly
        # occluded corner) can dominate the overall RMSE. Reporting per-view
        # turns "the calibration is bad" into "view 7 is bad, drop it".
        per_view = []
        for i in range(self.n_views):
            proj, _ = cv2.projectPoints(self._obj_pts[i], rvecs[i], tvecs[i], K, dist)
            d = proj.reshape(-1, 2) - self._img_pts[i].reshape(-1, 2)
            per_view.append(float(np.sqrt((d ** 2).sum(axis=1).mean())))

        all_pts = np.concatenate([p.reshape(-1, 2) for p in self._img_pts])
        coverage, edge_coverage = grid_occupancy(all_pts, w, h)

        # Tilt span: the angle between each board's normal and the camera axis.
        # A planar target's Z axis after rvec IS its normal, so the third column
        # of R tells us how face-on that view was. Spread across views is what
        # breaks the focal-length/distance degeneracy (see note 1 at the top).
        tilts = []
        for rv in rvecs:
            Rm, _ = cv2.Rodrigues(rv)
            tilts.append(np.degrees(np.arccos(np.clip(abs(Rm[2, 2]), 0, 1))))
        tilt_span = float(np.max(tilts) - np.min(tilts)) if tilts else 0.0

        return CalibrationResult(
            camera_info=CameraInfo(Header(0.0, "camera_optical"), K, dist.ravel(), w, h),
            rms=float(rms),
            per_view_rms=np.array(per_view),
            n_views=self.n_views,
            coverage=coverage,
            edge_coverage=edge_coverage,
            tilt_span_deg=tilt_span,
            board=self.board,
        )


# ---------------------------------------------------------------- persistence

def save_intrinsics(path: str | Path, result: CalibrationResult) -> None:
    """YAML, not pickle or npz.

    Calibration is a physical measurement of a physical device -- it should be
    readable, diffable in git, and hand-editable when you need to sanity-check
    one number. We store the QUALITY METRICS alongside K, so six months later
    you can tell whether to trust the file without recalibrating.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ci = result.camera_info
    p.write_text(yaml.safe_dump({
        "image_width": ci.width,
        "image_height": ci.height,
        "camera_matrix": ci.K.tolist(),
        "distortion_model": "plumb_bob",
        "distortion_coefficients": ci.dist.ravel().tolist(),
        "quality": {
            "reprojection_rms_px": result.rms,
            "worst_view_rms_px": float(result.per_view_rms.max()),
            "n_views": result.n_views,
            "image_coverage": result.coverage,
            "edge_coverage": result.edge_coverage,
            "tilt_span_deg": result.tilt_span_deg,
            "problems": result.problems(),
        },
        "board": {
            "cols": result.board.cols,
            "rows": result.board.rows,
            "square_size_m": result.board.square_size_m,
        },
    }, sort_keys=False))


def load_intrinsics(path: str | Path) -> CameraInfo:
    d = yaml.safe_load(Path(path).read_text())
    return CameraInfo(
        header=Header(0.0, "camera_optical"),
        K=np.array(d["camera_matrix"], dtype=np.float64),
        dist=np.array(d["distortion_coefficients"], dtype=np.float64),
        width=int(d["image_width"]),
        height=int(d["image_height"]),
    )


def recommend_board(width: int, height: int, square_size_m: float = 0.025) -> CheckerboardSpec:
    """Suggest board dimensions whose aspect ratio matches the sensor.

    WHY THIS IS NOT COSMETIC
    ------------------------
    findChessboardCorners is all-or-nothing: a partly-visible board is not
    detected, so every view must fit ENTIRELY in frame. That means the board's
    bounding box, quiet zone included, is inscribed in the image -- and an
    aspect mismatch wastes the difference on every single view.

    Measured on a 960x540 (16:9) sensor, sweeping board shapes and taking the
    convex hull of all corners over 30 visibility-checked views:

        board   bordered aspect   coverage   edge corners
        9x6          1.38            52%         2.3%
        10x6         1.50            55%         3.1%
        11x6         1.62            58%         3.9%
        12x7         1.56            60%         4.4%

    The classic 9x6 board is 3:2 -- fine on an old 4:3 camera, and a permanent
    handicap on a 16:9 one. You cannot fix that by moving the board around
    more; the field of view is simply unreachable. Edge corners are what
    constrain the distortion coefficients, so this costs real accuracy exactly
    where distortion is largest.
    """
    sensor_aspect = width / height
    best, best_err = None, float("inf")
    for rows in range(5, 9):
        for cols in range(rows + 1, rows + 9):
            # +2 squares for the quiet zone the detector needs on each side.
            aspect = (cols + 2) / (rows + 2)
            err = abs(aspect - sensor_aspect)
            if err < best_err:
                best, best_err = (cols, rows), err
    return CheckerboardSpec(best[0], best[1], square_size_m)
