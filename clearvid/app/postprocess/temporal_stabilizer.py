"""Optical-flow guided temporal stabilizer for per-frame super-resolution output.

Reduces temporal flickering by blending each enhanced frame with its warped
predecessor.  Static areas receive stronger blending (flicker suppression)
while high-motion areas keep the current frame intact (preserving dynamics).

Uses OpenCV DIS (Dense Inverse Search) optical flow which is already part of
the opencv-python dependency – no additional packages required.
"""

from __future__ import annotations

import cv2
import numpy as np


class TemporalStabilizer:
    """Stateful per-frame temporal stabilizer.

    Parameters
    ----------
    strength:
        Maximum blend ratio towards the warped previous frame (0.0–1.0).
        Higher values = more aggressive flicker suppression.
    scene_threshold:
        Mean optical-flow magnitude above which a scene change is declared
        and the buffer is reset.  Tuned heuristically for typical 1080p/4K
        super-resolution output.
    flow_scale:
        Downscale factor for optical-flow computation.  Smaller = faster but
        less precise flow.  0.25 is a good balance for HD/4K frames.
    """

    def __init__(
        self,
        strength: float = 0.6,
        scene_threshold: float = 40.0,
        flow_scale: float = 0.25,
    ) -> None:
        self._strength = float(np.clip(strength, 0.0, 1.0))
        self._scene_threshold = scene_threshold
        self._flow_scale = flow_scale
        self._prev_frame: np.ndarray | None = None
        self._prev_gray_small: np.ndarray | None = None
        self._dis = cv2.DISOpticalFlow_create(  # type: ignore[attr-defined]
            cv2.DISOPTICAL_FLOW_PRESET_FAST,  # type: ignore[attr-defined]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """Accept a BGR uint8 frame and return the stabilized version."""
        gray_small = self._to_gray_small(frame)

        # First frame or after a reset – nothing to blend with.
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            self._prev_gray_small = gray_small
            return frame

        # Compute dense optical flow (prev → current) at reduced resolution.
        flow_small = self._dis.calc(
            self._prev_gray_small, gray_small, None,  # type: ignore[arg-type]
        )

        # Scene-change detection: large average motion → reset buffer.
        magnitude = np.sqrt(
            flow_small[..., 0] ** 2 + flow_small[..., 1] ** 2,
        )
        if float(np.mean(magnitude)) > self._scene_threshold:
            self._prev_frame = frame.copy()
            self._prev_gray_small = gray_small
            return frame

        # Up-scale flow to full resolution.
        h, w = frame.shape[:2]
        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        scale_x = w / flow_small.shape[1]
        scale_y = h / flow_small.shape[0]
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y

        # Warp previous frame to align with the current one.
        warped = self._warp_frame(self._prev_frame, flow)

        # Build per-pixel adaptive blend weight from flow magnitude.
        mag_full = cv2.resize(magnitude, (w, h), interpolation=cv2.INTER_LINEAR)
        blend_weight = self._build_blend_weight(mag_full)

        # Blend: static areas → warped previous; moving areas → current.
        stabilized = self._blend(frame, warped, blend_weight)

        self._prev_frame = frame.copy()
        self._prev_gray_small = gray_small
        return stabilized

    def reset(self) -> None:
        """Clear the internal buffer (e.g. at a scene boundary)."""
        self._prev_frame = None
        self._prev_gray_small = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_gray_small(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        small_w = max(1, int(w * self._flow_scale))
        small_h = max(1, int(h * self._flow_scale))
        return cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Backward-warp *frame* using the forward optical flow.

        Forward flow (prev→curr) gives displacement of each prev pixel.
        To find the source in prev for each output pixel we subtract the flow.
        """
        h, w = flow.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        remap_x = map_x - flow[..., 0]
        remap_y = map_y - flow[..., 1]
        return cv2.remap(
            frame,
            remap_x,
            remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _build_blend_weight(self, magnitude: np.ndarray) -> np.ndarray:
        """Map per-pixel flow magnitude to a blend weight in [0, strength].

        Small motion → high weight (use warped previous = suppress flicker).
        Large motion → low weight (keep current frame = preserve dynamics).

        The transition zone is a smooth sigmoid centred around *motion_mid*
        (pixels with ~5px displacement keep about 50% blend).
        """
        motion_mid = 5.0
        steepness = 0.6
        # Logistic function: 1 → 0 as magnitude increases.
        alpha = 1.0 / (1.0 + np.exp(steepness * (magnitude - motion_mid)))
        return (alpha * self._strength).astype(np.float32)

    @staticmethod
    def _blend(
        current: np.ndarray,
        warped: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend current and warped frames using per-pixel weight."""
        w = weight[..., np.newaxis]  # (H, W, 1) for broadcasting over BGR
        blended = (1.0 - w) * current.astype(np.float32) + w * warped.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)
