"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: POSE TRACKER & GLOVE ANCHOR (pose_tracker.py)
Biomechanical Arm Kinematics & Authoritative Glove Pose Estimation:
- MediaPipe PoseLandmarker Lite (CPU optimized, downsampled inference)
- Landmark 11/12 (shoulders), 13/14 (elbows), 15/16 (wrists), 0 (nose)
- One Euro Filter 2D spatial stabilization (eliminates resting jitter)
- Adaptive Angular Filtering on forearm vector (smooth 360 rotation)
- Proportional glove visual anchor & hit detection centering
- Complete TrackingResult with velocity, acceleration, and quality metrics
=============================================================================
"""

import os
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

from .filters import OneEuroFilter2D, AngularFilter, MotionEstimator, OutlierDetector
from .coordinate_mapper import CoordinateMapper
from .tracking_quality import TrackingQualityEvaluator, TrackingState
from .identity_manager import IdentityManager

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")


@dataclass
class ArmTrackingResult:
    """Complete kinematic and tracking data for a single arm."""
    side: str  # "Left" or "Right"
    wrist_raw: Tuple[float, float] = (0.0, 0.0)
    wrist_filtered: Tuple[float, float] = (0.0, 0.0)
    elbow_raw: Tuple[float, float] = (0.0, 0.0)
    elbow_filtered: Tuple[float, float] = (0.0, 0.0)
    shoulder_raw: Tuple[float, float] = (0.0, 0.0)
    shoulder_filtered: Tuple[float, float] = (0.0, 0.0)

    forearm_vector: Tuple[float, float] = (0.0, -1.0)
    forearm_angle: float = -math.pi / 2.0  # Radians
    forearm_length: float = 80.0

    glove_center: Tuple[float, float] = (0.0, 0.0)
    hitbox_radius: float = 45.0
    extension_ratio: float = 0.0  # 0.0 (tucked) to 1.0+ (fully extended punch)

    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0  # px/s
    speed_kmh: float = 0.0
    acceleration: Tuple[float, float] = (0.0, 0.0)

    confidence: float = 0.0
    visibility: float = 0.0
    tracking_state: TrackingState = TrackingState.LOST
    is_valid_for_attacks: bool = False
    dropout_alpha: float = 0.0
    is_cross_body: bool = False


@dataclass
class PoseTrackingResult:
    """Unified tracking frame containing both arms, head, and pipeline metrics."""
    timestamp: float = 0.0
    frame_id: int = 0
    frame_dimensions: Tuple[int, int] = (1280, 720)
    nose: Optional[Tuple[float, float]] = None
    left_arm: ArmTrackingResult = field(default_factory=lambda: ArmTrackingResult("Left"))
    right_arm: ArmTrackingResult = field(default_factory=lambda: ArmTrackingResult("Right"))
    has_pose: bool = False
    inference_time_ms: float = 0.0
    tracking_fps: float = 0.0
    telemetry: Dict[str, Any] = field(default_factory=dict)


class ArmTrackerState:
    """Internal filter instances and state history for one arm."""

    def __init__(self, side: str):
        self.side = side
        self.filter_wrist = OneEuroFilter2D(min_cutoff=1.1, beta=0.018, d_cutoff=1.0)
        self.filter_elbow = OneEuroFilter2D(min_cutoff=0.9, beta=0.012, d_cutoff=1.0)
        self.filter_shoulder = OneEuroFilter2D(min_cutoff=0.6, beta=0.008, d_cutoff=1.0)
        self.filter_angle = AngularFilter(min_rate=0.25, max_rate=0.88, vel_scale=0.0018)
        self.outlier = OutlierDetector(max_speed_px_per_sec=3600.0)
        self.motion = MotionEstimator(vel_smoothing=0.35, accel_smoothing=0.25)
        self.quality = TrackingQualityEvaluator()

    def reset(self):
        self.filter_wrist.reset()
        self.filter_elbow.reset()
        self.filter_shoulder.reset()
        self.filter_angle.reset()
        self.outlier.reset()
        self.motion.reset()
        self.quality.reset()


class PoseTracker:
    """
    High-Performance MediaPipe Pose Tracker.
    Produces stabilized, biologically accurate arm tracking for Shadow Boxing.
    """

    def __init__(self, model_path: Optional[str] = None,
                 cam_w: int = 640, cam_h: int = 480,
                 screen_w: int = 1280, screen_h: int = 720,
                 mirror_x: bool = True):
        self.model_path = model_path or MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Pose landmarker model not found at {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.coord_mapper = CoordinateMapper(cam_w=cam_w, cam_h=cam_h,
                                             screen_w=screen_w, screen_h=screen_h,
                                             mode="CENTER_CROP", mirror_x=mirror_x)

        self.left_state = ArmTrackerState("Left")
        self.right_state = ArmTrackerState("Right")
        self.identity_mgr = IdentityManager()

        self.filter_nose = OneEuroFilter2D(min_cutoff=0.8, beta=0.01, d_cutoff=1.0)
        self.prev_process_time = time.time()
        self.tracking_fps = 30.0

        # Downsample dimensions for CPU inference
        self.infer_w = 384
        self.infer_h = 288

    def process_frame(self, raw_cam_bgr: np.ndarray, dt: Optional[float] = None,
                      frame_id: int = 0) -> PoseTrackingResult:
        """
        Receives raw (unflipped) camera BGR frame.
        Performs downsampled MediaPipe inference and generates PoseTrackingResult.
        """
        now = time.time()
        if dt is None:
            dt = max(1e-4, min(0.066, now - self.prev_process_time))
        self.prev_process_time = now

        # Prepare un-mirrored downscaled image for MediaPipe
        infer_frame = self.coord_mapper.prepare_infer_frame(raw_cam_bgr, self.infer_w, self.infer_h)
        rgb_frame = np.ascontiguousarray(infer_frame[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        t_infer_start = time.time()
        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            print(f"[PoseTracker Warning] MediaPipe error: {e}")
            result = None
        t_infer_end = time.time()
        infer_duration_ms = (t_infer_end - t_infer_start) * 1000.0

        inst_fps = 1.0 / max(1e-4, dt)
        self.tracking_fps = 0.8 * self.tracking_fps + 0.2 * inst_fps

        out = PoseTrackingResult(
            timestamp=now,
            frame_id=frame_id,
            frame_dimensions=(self.coord_mapper.screen_w, self.coord_mapper.screen_h),
            inference_time_ms=round(infer_duration_ms, 2),
            tracking_fps=round(self.tracking_fps, 1),
        )

        if not result or not result.pose_landmarks or len(result.pose_landmarks) == 0:
            # Tracking loss
            out.has_pose = False
            self._update_arm_lost(out.left_arm, self.left_state, now, dt)
            self._update_arm_lost(out.right_arm, self.right_state, now, dt)
            return out

        out.has_pose = True
        lms = result.pose_landmarks[0]

        # Landmark extraction & screen mapping:
        # In MediaPipe:
        # 11: Left Shoulder, 12: Right Shoulder
        # 13: Left Elbow, 14: Right Elbow
        # 15: Left Wrist, 16: Right Wrist
        # 0: Nose
        def get_screen_pt(idx: int) -> Tuple[Tuple[float, float], float, float]:
            lm = lms[idx]
            pt = self.coord_mapper.norm_to_screen(lm.x, lm.y)
            conf = getattr(lm, "presence", 0.9)
            vis = getattr(lm, "visibility", 0.9)
            return pt, float(conf), float(vis)

        nose_pt, nose_c, _ = get_screen_pt(0)
        out.nose = self.filter_nose.filter(nose_pt, dt)

        sh_l, sh_l_c, sh_l_v = get_screen_pt(11)
        sh_r, sh_r_c, sh_r_v = get_screen_pt(12)
        el_l, el_l_c, el_l_v = get_screen_pt(13)
        el_r, el_r_c, el_r_v = get_screen_pt(14)
        wr_l, wr_l_c, wr_l_v = get_screen_pt(15)
        wr_r, wr_r_c, wr_r_v = get_screen_pt(16)

        # Resolve left/right identities against shoulder origins
        res_wl, res_wr, swapped = self.identity_mgr.resolve_arm_identities(
            raw_shoulder_l=sh_l, raw_shoulder_r=sh_r,
            raw_wrist_l=wr_l, raw_wrist_r=wr_r
        )

        # Update Left Arm
        self._update_arm(
            out.left_arm, self.left_state,
            wrist_pt=res_wl, elbow_pt=el_l, shoulder_pt=sh_l,
            conf=wr_l_c, vis=wr_l_v, now=now, dt=dt,
            other_shoulder=sh_r
        )

        # Update Right Arm
        self._update_arm(
            out.right_arm, self.right_state,
            wrist_pt=res_wr, elbow_pt=el_r, shoulder_pt=sh_r,
            conf=wr_r_c, vis=wr_r_v, now=now, dt=dt,
            other_shoulder=sh_l
        )

        return out

    def _update_arm(self, arm_res: ArmTrackingResult, state: ArmTrackerState,
                    wrist_pt: Optional[Tuple[float, float]],
                    elbow_pt: Tuple[float, float],
                    shoulder_pt: Tuple[float, float],
                    conf: float, vis: float, now: float, dt: float,
                    other_shoulder: Tuple[float, float]):
        """Processes arm coordinates through 1€ filters and computes glove pose."""
        if wrist_pt is None:
            self._update_arm_lost(arm_res, state, now, dt)
            return

        # 1. Outlier sanitization
        clean_wrist, is_outlier = state.outlier.sanitize(wrist_pt, dt)

        # 2. Update tracking quality state
        q_state = state.quality.update(is_detected=True, confidence=conf, visibility=vis,
                                       pos=clean_wrist, now=now)

        # 3. 1€ Filtering
        filt_wrist = state.filter_wrist.filter(clean_wrist, dt)
        filt_elbow = state.filter_elbow.filter(elbow_pt, dt)
        filt_shoulder = state.filter_shoulder.filter(shoulder_pt, dt)

        # 4. Forearm Vector & Angle
        dx = filt_wrist[0] - filt_elbow[0]
        dy = filt_wrist[1] - filt_elbow[1]
        forearm_len = max(20.0, math.hypot(dx, dy))
        ux = dx / forearm_len
        uy = dy / forearm_len

        raw_angle = math.atan2(dy, dx)
        # Filter angle
        smooth_angle = state.filter_angle.filter(raw_angle, dt=dt)
        f_ux = math.cos(smooth_angle)
        f_uy = math.sin(smooth_angle)

        # 5. Proportional Glove Visual Center (Wrist + forward palm offset along forearm)
        # Visual glove sits naturally over player's fist
        glove_offset = 0.35 * forearm_len
        glove_x = filt_wrist[0] + f_ux * glove_offset
        glove_y = filt_wrist[1] + f_uy * glove_offset

        # 6. Motion estimation (velocity, speed, acceleration)
        state.motion.update((glove_x, glove_y), dt)
        speed = state.motion.speed
        speed_kmh = speed * 0.04  # Screen px/s to intuitive km/h display

        # 7. Extension ratio: ||Wrist - Shoulder|| / (||Elbow - Shoulder|| + ||Wrist - Elbow||)
        reach_dist = math.hypot(filt_wrist[0] - filt_shoulder[0], filt_wrist[1] - filt_shoulder[1])
        upper_arm_len = math.hypot(filt_elbow[0] - filt_shoulder[0], filt_elbow[1] - filt_shoulder[1])
        max_possible_arm = max(40.0, upper_arm_len + forearm_len)
        extension_ratio = max(0.0, min(1.2, reach_dist / max_possible_arm))

        # 8. Cross-body check
        is_cross = IdentityManager.is_cross_body(filt_wrist, filt_shoulder, other_shoulder)

        # Populate result
        arm_res.wrist_raw = clean_wrist
        arm_res.wrist_filtered = filt_wrist
        arm_res.elbow_raw = elbow_pt
        arm_res.elbow_filtered = filt_elbow
        arm_res.shoulder_raw = shoulder_pt
        arm_res.shoulder_filtered = filt_shoulder

        arm_res.forearm_vector = (f_ux, f_uy)
        arm_res.forearm_angle = smooth_angle
        arm_res.forearm_length = forearm_len

        arm_res.glove_center = (glove_x, glove_y)
        arm_res.extension_ratio = extension_ratio
        arm_res.velocity = state.motion.velocity
        arm_res.speed = speed
        arm_res.speed_kmh = speed_kmh
        arm_res.acceleration = state.motion.acceleration

        arm_res.confidence = conf
        arm_res.visibility = vis
        arm_res.tracking_state = q_state
        arm_res.is_valid_for_attacks = state.quality.is_valid_for_attacks
        arm_res.dropout_alpha = state.quality.dropout_alpha
        arm_res.is_cross_body = is_cross

    def _update_arm_lost(self, arm_res: ArmTrackingResult, state: ArmTrackerState,
                         now: float, dt: float):
        """Handles lost arm tracking gracefully with smooth dropout."""
        q_state = state.quality.update(is_detected=False, confidence=0.0, visibility=0.0,
                                       pos=None, now=now)
        arm_res.tracking_state = q_state
        arm_res.is_valid_for_attacks = False
        arm_res.dropout_alpha = state.quality.dropout_alpha
        arm_res.speed = 0.0
        arm_res.speed_kmh = 0.0
        arm_res.velocity = (0.0, 0.0)

    def close(self):
        """Releases MediaPipe detector resources."""
        if hasattr(self, "detector") and self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None
