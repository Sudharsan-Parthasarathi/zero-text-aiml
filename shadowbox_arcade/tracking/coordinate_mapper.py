"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: COORDINATE MAPPER (coordinate_mapper.py)
Unified Coordinate System & Geometric Projection:
- Maps normalized MediaPipe coordinates [0, 1] to screen display pixels
- Supports FILL, CENTER_CROP (AR optimal), and LETTERBOX projection modes
- Guaranteed single-point horizontal mirroring (zero double-flip confusion)
- Bidirectional mapping (Camera <-> Screen <-> Normalized)
=============================================================================
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


class CoordinateMapper:
    """
    Coordinates transformations between camera frames, neural network inference space,
    and game screen displays.
    """

    def __init__(self, cam_w: int = 640, cam_h: int = 480,
                 screen_w: int = 1280, screen_h: int = 720,
                 mode: str = "CENTER_CROP", mirror_x: bool = True):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.mode = mode.upper()  # "CENTER_CROP", "FILL", "LETTERBOX"
        self.mirror_x = mirror_x
        self._recompute_viewport()

    def update_dimensions(self, cam_w: int, cam_h: int, screen_w: int, screen_h: int):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._recompute_viewport()

    def _recompute_viewport(self):
        cam_aspect = self.cam_w / max(1.0, float(self.cam_h))
        screen_aspect = self.screen_w / max(1.0, float(self.screen_h))

        if self.mode == "FILL":
            self.scale_x = self.screen_w / float(self.cam_w)
            self.scale_y = self.screen_h / float(self.cam_h)
            self.offset_x = 0.0
            self.offset_y = 0.0
            self.crop_x = 0
            self.crop_y = 0
            self.crop_w = self.cam_w
            self.crop_h = self.cam_h
        elif self.mode == "CENTER_CROP":
            # Crop camera frame to match screen aspect ratio without stretching
            if cam_aspect > screen_aspect:
                # Camera is wider than screen: crop camera width
                self.crop_h = self.cam_h
                self.crop_w = int(self.cam_h * screen_aspect)
                self.crop_x = (self.cam_w - self.crop_w) // 2
                self.crop_y = 0
            else:
                # Camera is taller than screen: crop camera height
                self.crop_w = self.cam_w
                self.crop_h = int(self.cam_w / screen_aspect)
                self.crop_x = 0
                self.crop_y = (self.cam_h - self.crop_h) // 2

            self.scale_x = self.screen_w / float(self.crop_w)
            self.scale_y = self.screen_h / float(self.crop_h)
            self.offset_x = 0.0
            self.offset_y = 0.0
        else:  # "LETTERBOX"
            if cam_aspect > screen_aspect:
                # Pillarbox / letterbox top-bottom
                render_w = self.screen_w
                render_h = int(self.screen_w / cam_aspect)
                self.offset_x = 0.0
                self.offset_y = (self.screen_h - render_h) / 2.0
                self.scale_x = render_w / float(self.cam_w)
                self.scale_y = render_h / float(self.cam_h)
            else:
                render_h = self.screen_h
                render_w = int(self.screen_h * cam_aspect)
                self.offset_x = (self.screen_w - render_w) / 2.0
                self.offset_y = 0.0
                self.scale_x = render_w / float(self.cam_w)
                self.scale_y = render_h / float(self.cam_h)

            self.crop_x = 0
            self.crop_y = 0
            self.crop_w = self.cam_w
            self.crop_h = self.cam_h

    def norm_to_screen(self, nx: float, ny: float) -> Tuple[float, float]:
        """
        Converts normalized MediaPipe landmark coordinate [0.0, 1.0] to game screen pixels.
        Handles mirroring if mirror_x is enabled.
        """
        # Mirroring in normalized space
        eff_nx = (1.0 - nx) if self.mirror_x else nx

        # Camera pixel coordinates
        cam_px = eff_nx * self.cam_w
        cam_py = ny * self.cam_h

        # Project to screen
        if self.mode == "CENTER_CROP":
            rel_x = cam_px - self.crop_x
            rel_y = cam_py - self.crop_y
            sx = rel_x * self.scale_x
            sy = rel_y * self.scale_y
        elif self.mode == "FILL":
            sx = cam_px * self.scale_x
            sy = cam_py * self.scale_y
        else:  # LETTERBOX
            sx = self.offset_x + cam_px * self.scale_x
            sy = self.offset_y + cam_py * self.scale_y

        return (sx, sy)

    def screen_to_norm(self, sx: float, sy: float) -> Tuple[float, float]:
        """Converts game screen pixel coordinate back to normalized space [0.0, 1.0]."""
        if self.mode == "CENTER_CROP":
            cam_px = (sx / max(1e-4, self.scale_x)) + self.crop_x
            cam_py = (sy / max(1e-4, self.scale_y)) + self.crop_y
        elif self.mode == "FILL":
            cam_px = sx / max(1e-4, self.scale_x)
            cam_py = sy / max(1e-4, self.scale_y)
        else:  # LETTERBOX
            cam_px = (sx - self.offset_x) / max(1e-4, self.scale_x)
            cam_py = (sy - self.offset_y) / max(1e-4, self.scale_y)

        nx = cam_px / max(1.0, float(self.cam_w))
        ny = cam_py / max(1.0, float(self.cam_h))
        eff_nx = (1.0 - nx) if self.mirror_x else nx
        return (eff_nx, ny)

    def prepare_camera_frame(self, raw_cam_bgr: np.ndarray) -> np.ndarray:
        """
        Mirrors and crops/resizes the raw camera frame to exactly match
        the screen resolution and projection geometry.
        """
        frame = raw_cam_bgr
        if self.mirror_x:
            frame = cv2.flip(frame, 1)

        if self.mode == "CENTER_CROP":
            # Crop to matching aspect ratio
            crop = frame[self.crop_y:self.crop_y + self.crop_h, self.crop_x:self.crop_x + self.crop_w]
            return cv2.resize(crop, (self.screen_w, self.screen_h), interpolation=cv2.INTER_LINEAR)
        elif self.mode == "FILL":
            return cv2.resize(frame, (self.screen_w, self.screen_h), interpolation=cv2.INTER_LINEAR)
        else:  # LETTERBOX
            out = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            resized = cv2.resize(frame, (int(self.cam_w * self.scale_x), int(self.cam_h * self.scale_y)))
            ox = int(self.offset_x)
            oy = int(self.offset_y)
            out[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
            return out

    def prepare_infer_frame(self, raw_cam_bgr: np.ndarray, infer_w: int = 384, infer_h: int = 288) -> np.ndarray:
        """
        Produces downscaled un-mirrored frame for MediaPipe detection.
        MediaPipe neural nets perform best on standard orientation, while coordinate_mapper
        handles mirror inversion mathematically.
        """
        return cv2.resize(raw_cam_bgr, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
