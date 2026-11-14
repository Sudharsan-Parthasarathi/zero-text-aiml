"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: PLAYER CALIBRATION (player_calibration.py)
Interactive Biomechanical Calibration & Normalization Engine:
- Steps: Stance Detection -> Neutral Guard Hold -> Test Jab -> Test Cross -> Ready
- Measures: Shoulder width, forearm length, maximum arm reach, neutral guard zone
- Normalizes punch thresholds dynamically across children, adults & camera distances
- Generates real-time player status messages (READY, MOVE INTO FRAME, HAND LOST)
=============================================================================
"""

import time
import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

from .pose_tracker import PoseTrackingResult, ArmTrackingResult
from .tracking_quality import TrackingState


@dataclass
class CalibrationData:
    """Normalized player proportions and calibrated play boundaries."""
    shoulder_width: float = 160.0
    arm_reach: float = 240.0
    forearm_len_l: float = 80.0
    forearm_len_r: float = 80.0
    guard_center_l: Tuple[float, float] = (0.0, 0.0)
    guard_center_r: Tuple[float, float] = (0.0, 0.0)
    guard_radius: float = 65.0
    scale_multiplier: float = 1.0
    is_calibrated: bool = False


class CalibrationManager:
    """
    Manages interactive 5-step player calibration and status feedback.
    """

    STEPS = ["WAIT_FOR_STANCE", "HOLD_GUARD", "TEST_JAB", "TEST_CROSS", "COMPLETE"]

    def __init__(self, screen_w: int = 1280, screen_h: int = 720):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.data = CalibrationData()

        self.step_idx: int = 0
        self.step_timer: float = 0.0
        self.guard_hold_start: float = 0.0
        self.guard_samples_l = []
        self.guard_samples_r = []
        self.max_jab_reach: float = 0.0
        self.max_cross_reach: float = 0.0
        self.status_message: str = "POSITION YOURSELF IN FRONT OF CAMERA"

    @property
    def current_step(self) -> str:
        return self.STEPS[self.step_idx]

    @property
    def is_complete(self) -> bool:
        return self.step_idx >= 4 and self.data.is_calibrated

    def start(self):
        """Begins calibration sequence."""
        self.step_idx = 0
        self.step_timer = time.time()
        self.guard_hold_start = 0.0
        self.guard_samples_l.clear()
        self.guard_samples_r.clear()
        self.max_jab_reach = 0.0
        self.max_cross_reach = 0.0
        self.data.is_calibrated = False

    def update(self, pose: PoseTrackingResult) -> bool:
        """
        Updates calibration state machine based on incoming pose.
        Returns True when calibration completes.
        """
        now = time.time()
        l_arm = pose.left_arm
        r_arm = pose.right_arm

        if self.step_idx == 0:  # WAIT_FOR_STANCE
            self.status_message = "STEP 1/4: STAND IN FRONT OF CAMERA WITH BOTH ARMS VISIBLE"
            if (pose.has_pose and l_arm.tracking_state == TrackingState.TRACKING_OK and
                    r_arm.tracking_state == TrackingState.TRACKING_OK):
                # Measure baseline shoulder width
                sw = math.hypot(l_arm.shoulder_filtered[0] - r_arm.shoulder_filtered[0],
                                l_arm.shoulder_filtered[1] - r_arm.shoulder_filtered[1])
                if sw > 60.0:
                    self.data.shoulder_width = sw
                    self.data.scale_multiplier = max(0.65, min(1.8, sw / 160.0))
                    self.step_idx = 1
                    self.guard_hold_start = now
                    self.guard_samples_l.clear()
                    self.guard_samples_r.clear()

        elif self.step_idx == 1:  # HOLD_GUARD
            self.status_message = "STEP 2/4: HOLD NEUTRAL BOXING GUARD NEAR CHEST"
            if (l_arm.tracking_state == TrackingState.TRACKING_OK and
                    r_arm.tracking_state == TrackingState.TRACKING_OK):
                self.guard_samples_l.append(l_arm.glove_center)
                self.guard_samples_r.append(r_arm.glove_center)

                hold_duration = now - self.guard_hold_start
                if hold_duration >= 1.2:
                    # Average samples
                    avg_lx = float(np.mean([p[0] for p in self.guard_samples_l]))
                    avg_ly = float(np.mean([p[1] for p in self.guard_samples_l]))
                    avg_rx = float(np.mean([p[0] for p in self.guard_samples_r]))
                    avg_ry = float(np.mean([p[1] for p in self.guard_samples_r]))

                    self.data.guard_center_l = (avg_lx, avg_ly)
                    self.data.guard_center_r = (avg_rx, avg_ry)
                    self.data.forearm_len_l = l_arm.forearm_length
                    self.data.forearm_len_r = r_arm.forearm_length
                    self.step_idx = 2
                    self.step_timer = now
            else:
                self.guard_hold_start = now

        elif self.step_idx == 2:  # TEST_JAB
            self.status_message = "STEP 3/4: THROW A TEST JAB (LEFT HAND)"
            reach_l = math.hypot(l_arm.wrist_filtered[0] - l_arm.shoulder_filtered[0],
                                 l_arm.wrist_filtered[1] - l_arm.shoulder_filtered[1])
            if reach_l > self.max_jab_reach:
                self.max_jab_reach = reach_l

            if l_arm.speed > 280.0 and self.max_jab_reach > self.data.shoulder_width * 1.1:
                self.step_idx = 3
                self.step_timer = now

        elif self.step_idx == 3:  # TEST_CROSS
            self.status_message = "STEP 4/4: THROW A TEST CROSS (RIGHT HAND)"
            reach_r = math.hypot(r_arm.wrist_filtered[0] - r_arm.shoulder_filtered[0],
                                 r_arm.wrist_filtered[1] - r_arm.shoulder_filtered[1])
            if reach_r > self.max_cross_reach:
                self.max_cross_reach = reach_r

            if r_arm.speed > 280.0 and self.max_cross_reach > self.data.shoulder_width * 1.1:
                self.data.arm_reach = max(self.max_jab_reach, self.max_cross_reach)
                self.data.is_calibrated = True
                self.step_idx = 4
                self.status_message = "CALIBRATION COMPLETE! GET READY TO FIGHT!"
                return True

        return self.is_complete

    def get_tracking_status(self, pose: PoseTrackingResult) -> Tuple[str, Tuple[int, int, int]]:
        """
        Returns (status_text, (B, G, R) color) for gameplay tracking HUD badge.
        """
        if not pose.has_pose:
            return "MOVE INTO FRAME", (0, 80, 255)  # Orange

        l_ok = (pose.left_arm.tracking_state == TrackingState.TRACKING_OK)
        r_ok = (pose.right_arm.tracking_state == TrackingState.TRACKING_OK)

        if l_ok and r_ok:
            return "TRACKING OPTIMAL", (0, 255, 100)  # Green
        elif not l_ok and not r_ok:
            return "BOTH HANDS LOST", (0, 0, 255)  # Red
        elif not l_ok:
            return "LEFT HAND LOST", (0, 180, 255)
        else:
            return "RIGHT HAND LOST", (0, 180, 255)

    def draw_calibration_overlay(self, frame_bgr: np.ndarray):
        """Renders step instructions and visual calibration guides onto frame."""
        h, w = frame_bgr.shape[:2]

        # Top banner
        banner_h = 75
        cv2.rectangle(frame_bgr, (0, 0), (w, banner_h), (18, 22, 30), -1)
        cv2.line(frame_bgr, (0, banner_h), (w, banner_h), (0, 220, 255), 2)

        cv2.putText(frame_bgr, "PLAYER CALIBRATION", (w // 2 - 130, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 2)
        cv2.putText(frame_bgr, self.status_message, (w // 2 - int(len(self.status_message) * 4.2), 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 230, 240), 1)

        # Step indicator
        for idx, name in enumerate(["STANCE", "GUARD", "JAB", "CROSS"]):
            col = (0, 255, 100) if idx < self.step_idx else ((0, 220, 255) if idx == self.step_idx else (90, 95, 110))
            bx = w // 2 - 180 + idx * 95
            cv2.rectangle(frame_bgr, (bx, banner_h + 15), (bx + 80, banner_h + 35), col, -1 if idx == self.step_idx else 1)
            cv2.putText(frame_bgr, name, (bx + 12, banner_h + 29),
                        cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 0, 0) if idx == self.step_idx else col, 1)

        # Guard target zones
        if self.step_idx == 1:
            gl_x, gl_y = int(w * 0.40), int(h * 0.55)
            gr_x, gr_y = int(w * 0.60), int(h * 0.55)
            pulse = int(5 * math.sin(time.time() * 8))
            cv2.circle(frame_bgr, (gl_x, gl_y), 45 + pulse, (255, 200, 0), 2)
            cv2.circle(frame_bgr, (gr_x, gr_y), 45 + pulse, (0, 100, 255), 2)
            cv2.putText(frame_bgr, "HOLD LEFT GLOVE", (gl_x - 55, gl_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 100), 1)
            cv2.putText(frame_bgr, "HOLD RIGHT GLOVE", (gr_x - 60, gr_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 255), 1)
