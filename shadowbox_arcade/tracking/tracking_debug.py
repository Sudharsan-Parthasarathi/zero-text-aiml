"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: DEBUGGING & TELEMETRY (tracking_debug.py)
Real-Time F3 Engineering HUD & Motion Telemetry Logger:
- Visualizes raw vs filtered wrists, elbows, shoulders, and forearm vectors
- Glove anchor points, orientation vectors, and active punch state badges
- Performance metrics: Camera FPS, Frame Age (ms), Inference Latency (ms), Render FPS
- Tracking Quality % and jitter displacement analytics
- Circular buffer for recording motion telemetry samples
=============================================================================
"""

import time
import math
import json
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

from .pose_tracker import PoseTrackingResult, ArmTrackingResult
from .punch_detector import PunchState, PunchDetector


class TrackingDebugVisualizer:
    """
    Renders the F3 engineering HUD overlay and logs motion telemetry.
    """

    def __init__(self, max_history: int = 150):
        self.enabled: bool = False
        self.history = deque(maxlen=max_history)
        self.render_fps: float = 60.0
        self.prev_render_time: float = time.time()

    def toggle(self):
        self.enabled = not self.enabled

    def update_render_fps(self):
        now = time.time()
        dt = now - self.prev_render_time
        self.prev_render_time = now
        inst_fps = 1.0 / max(1e-4, dt)
        self.render_fps = 0.85 * self.render_fps + 0.15 * inst_fps

    def log_sample(self, pose: PoseTrackingResult, punch_detector: Optional[PunchDetector] = None):
        """Records a timestamped telemetry sample for offline diagnosis."""
        if not self.enabled and len(self.history) == 0:
            return

        l_arm = pose.left_arm
        r_arm = pose.right_arm
        jitter_l = math.hypot(l_arm.wrist_filtered[0] - l_arm.wrist_raw[0],
                              l_arm.wrist_filtered[1] - l_arm.wrist_raw[1])
        jitter_r = math.hypot(r_arm.wrist_filtered[0] - r_arm.wrist_raw[0],
                              r_arm.wrist_filtered[1] - r_arm.wrist_raw[1])

        sample = {
            "timestamp": round(pose.timestamp, 4),
            "frame_id": pose.frame_id,
            "has_pose": pose.has_pose,
            "infer_ms": pose.inference_time_ms,
            "track_fps": pose.tracking_fps,
            "render_fps": round(self.render_fps, 1),
            "left": {
                "raw": (round(l_arm.wrist_raw[0], 1), round(l_arm.wrist_raw[1], 1)),
                "filt": (round(l_arm.wrist_filtered[0], 1), round(l_arm.wrist_filtered[1], 1)),
                "speed": round(l_arm.speed, 1),
                "jitter": round(jitter_l, 2),
                "state": l_arm.tracking_state.value,
                "punch_state": punch_detector.left_sm.state.value if punch_detector else "N/A"
            },
            "right": {
                "raw": (round(r_arm.wrist_raw[0], 1), round(r_arm.wrist_raw[1], 1)),
                "filt": (round(r_arm.wrist_filtered[0], 1), round(r_arm.wrist_filtered[1], 1)),
                "speed": round(r_arm.speed, 1),
                "jitter": round(jitter_r, 2),
                "state": r_arm.tracking_state.value,
                "punch_state": punch_detector.right_sm.state.value if punch_detector else "N/A"
            }
        }
        self.history.append(sample)

    def export_samples_json(self, filepath: str):
        """Dumps captured tracking history to JSON."""
        try:
            with open(filepath, "w") as f:
                json.dump(list(self.history), f, indent=2)
            print(f"[Telemetry] Exported {len(self.history)} samples to {filepath}")
        except Exception as e:
            print(f"[Telemetry Error] Could not export samples: {e}")

    def render_overlay(self, frame_bgr: np.ndarray, pose: PoseTrackingResult,
                       punch_detector: Optional[PunchDetector] = None,
                       frame_age_ms: float = 0.0,
                       cam_telemetry: Optional[Dict[str, Any]] = None):
        """
        Draws comprehensive F3 engineering diagnostics directly onto OpenCV frame.
        """
        if not self.enabled:
            return

        self.update_render_fps()
        h, w = frame_bgr.shape[:2]

        # Top diagnostic bar
        cv2.rectangle(frame_bgr, (w // 2 - 190, 8), (w // 2 + 190, 36), (15, 20, 28), -1)
        cv2.rectangle(frame_bgr, (w // 2 - 190, 8), (w // 2 + 190, 36), (0, 255, 180), 1)
        cv2.putText(frame_bgr, "F3 DIAGNOSTICS: TRACKING ENGINE 2.0", (w // 2 - 170, 27),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 180), 1)

        # Draw arm skeletons
        for arm, col, punch_sm in [
            (pose.left_arm, (255, 220, 0), punch_detector.left_sm if punch_detector else None),
            (pose.right_arm, (0, 140, 255), punch_detector.right_sm if punch_detector else None)
        ]:
            if not arm.is_valid_for_attacks and arm.dropout_alpha <= 0.05:
                continue

            wx, wy = int(arm.wrist_filtered[0]), int(arm.wrist_filtered[1])
            ex, ey = int(arm.elbow_filtered[0]), int(arm.elbow_filtered[1])
            sx, sy = int(arm.shoulder_filtered[0]), int(arm.shoulder_filtered[1])
            gx, gy = int(arm.glove_center[0]), int(arm.glove_center[1])
            rwx, rwy = int(arm.wrist_raw[0]), int(arm.wrist_raw[1])

            # Raw wrist (small magenta dot) vs Filtered wrist
            cv2.circle(frame_bgr, (rwx, rwy), 4, (255, 0, 255), -1)
            cv2.line(frame_bgr, (rwx, rwy), (wx, wy), (255, 0, 255), 1)

            # Skeletal bones
            cv2.line(frame_bgr, (sx, sy), (ex, ey), (140, 150, 160), 2)
            cv2.line(frame_bgr, (ex, ey), (wx, wy), col, 2)
            cv2.line(frame_bgr, (wx, wy), (gx, gy), (0, 255, 255), 2)

            # Joint circles
            cv2.circle(frame_bgr, (sx, sy), 6, (180, 190, 200), -1)
            cv2.circle(frame_bgr, (ex, ey), 5, (200, 210, 220), -1)
            cv2.circle(frame_bgr, (wx, wy), 7, col, 2)
            cv2.circle(frame_bgr, (gx, gy), 8, (0, 255, 100), 2)

            # Forearm forward vector arrow
            ux, uy = arm.forearm_vector
            arrow_x = int(gx + ux * 45)
            arrow_y = int(gy + uy * 45)
            cv2.arrowedLine(frame_bgr, (gx, gy), (arrow_x, arrow_y), (0, 255, 120), 2, tipLength=0.25)

            # Arm label & punch state badge
            state_text = punch_sm.state.value if punch_sm else arm.tracking_state.value
            lbl = f"{arm.side[0]}: {state_text} ({arm.speed_kmh:.1f} km/h)"
            cv2.putText(frame_bgr, lbl, (gx - 35, gy - 16),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1)

        # Right-side Telemetry Panel
        card_w, card_h = 285, 185
        card_x = w - card_w - 20
        card_y = 50

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 220, 255), 1)

        cv2.putText(frame_bgr, "TELEMETRY METRICS", (card_x + 12, card_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 220, 255), 1)

        cap_fps = cam_telemetry.get("capture_fps", 0.0) if cam_telemetry else 0.0
        read_lat = cam_telemetry.get("read_latency_ms", 0.0) if cam_telemetry else 0.0

        jit_l = math.hypot(pose.left_arm.wrist_filtered[0] - pose.left_arm.wrist_raw[0],
                           pose.left_arm.wrist_filtered[1] - pose.left_arm.wrist_raw[1])
        jit_r = math.hypot(pose.right_arm.wrist_filtered[0] - pose.right_arm.wrist_raw[0],
                           pose.right_arm.wrist_filtered[1] - pose.right_arm.wrist_raw[1])

        lines = [
            f"Render FPS: {self.render_fps:.1f} | Game Target: 60",
            f"Camera FPS: {cap_fps:.1f} | Frame Age: {frame_age_ms:.1f}ms",
            f"Inference: {pose.inference_time_ms:.1f}ms ({pose.tracking_fps:.1f} FPS)",
            f"Webcam Read Latency: {read_lat:.1f}ms",
            f"1€ Jitter Delta: L={jit_l:.1f}px, R={jit_r:.1f}px",
            f"Left Ext: {pose.left_arm.extension_ratio:.2f} | Right: {pose.right_arm.extension_ratio:.2f}",
            f"Left State: {pose.left_arm.tracking_state.value} | Right: {pose.right_arm.tracking_state.value}"
        ]

        for i, text in enumerate(lines):
            cv2.putText(frame_bgr, text, (card_x + 12, card_y + 44 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 225, 240), 1)
