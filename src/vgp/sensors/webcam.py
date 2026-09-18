"""Real camera via OpenCV VideoCapture.

Small file, four hard-won details. None are intellectually interesting; all
of them cost someone a debugging session the first time.

1. BGR -> RGB AT THE BOUNDARY. cv2 hands back BGR; MuJoCo renders RGB. We fixed
   RGB as the internal convention (msgs/sensor.py), so the conversion happens
   here, once, at the edge. Mixing them does not raise -- it just makes
   everything look wrongly tinted, which reads as "bad lighting" and sends you
   hunting in the wrong place. For a colour-threshold detector it silently
   swaps which objects it finds.

2. RESOLUTION IS A REQUEST, NOT A COMMAND. `cap.set(FRAME_WIDTH, ...)` may be
   ignored -- the driver picks the nearest supported mode. We read back what we
   actually got and build intrinsics from THAT. Trusting the request is how you
   end up with a calibration for 1280x720 applied to 640x480 frames, which is a
   2x focal-length error and a completely wrong pose.

3. WARMUP FRAMES. The first few frames from a macOS camera are dark or garbage
   while auto-exposure converges. Calibrating on those gives a bad result that
   looks like a bad checkerboard.

4. NO INTRINSICS ARE INVENTED. Without a calibration file we could guess a
   plausible K from a typical FOV. We refuse: a guessed K yields poses that are
   confidently wrong with low reprojection error, which is far worse than an
   error message. Sim gets exact intrinsics for free; real cameras must earn
   theirs. See scripts/calibrate_intrinsics.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from vgp.msgs import CameraInfo, Header, Image
from vgp.sensors.source import CameraSource
from vgp.utils.frames import CAMERA_OPTICAL


class WebcamSource(CameraSource):
    def __init__(
        self,
        device: int = 0,
        width: int = 1280,
        height: int = 720,
        calibration: str | Path | None = None,
        frame_id: str = CAMERA_OPTICAL,
        warmup_frames: int = 10,
    ) -> None:
        self.frame_id = frame_id
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open camera {device}. On macOS the first run needs "
                "Camera permission for your terminal/IDE: System Settings > "
                "Privacy & Security > Camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Read back the ACTUAL mode -- see note 2.
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for _ in range(warmup_frames):          # note 3
            self.cap.read()
            time.sleep(0.01)

        self._K: np.ndarray | None = None
        self._dist: np.ndarray | None = None
        if calibration is not None:
            self.load_calibration(calibration)

    def load_calibration(self, path: str | Path) -> None:
        from vgp.calib.intrinsics import load_intrinsics

        info = load_intrinsics(path)
        if (info.width, info.height) != (self.width, self.height):
            # Rescale rather than refuse: calibrating at one resolution and
            # capturing at another is legitimate and common. But do it
            # explicitly and say so, because a SILENT mismatch is note 2's bug.
            sx, sy = self.width / info.width, self.height / info.height
            print(
                f"[webcam] calibration is {info.width}x{info.height} but capture is "
                f"{self.width}x{self.height}; rescaling intrinsics by ({sx:.3f}, {sy:.3f})"
            )
            info = info.scaled(sx, sy)
        self._K, self._dist = info.K, info.dist

    def read(self) -> tuple[Image, CameraInfo]:
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed (unplugged, or another app grabbed it)")
        if self._K is None:
            raise RuntimeError(
                "no intrinsics for this camera. Run "
                "`python scripts/calibrate_intrinsics.py` first. "
                "(We refuse to guess K -- see webcam.py note 4.)"
            )
        stamp = time.monotonic()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # note 1
        hdr = Header(stamp, self.frame_id)
        return (
            Image(hdr, rgb, "rgb8"),
            CameraInfo(hdr, self._K, self._dist, self.width, self.height),
        )

    def read_raw(self) -> np.ndarray:
        """RGB frame with no intrinsics required -- used BY the calibration
        script, which is the thing that produces the intrinsics."""
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
