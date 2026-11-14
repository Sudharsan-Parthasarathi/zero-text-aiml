"""
Fruit Ninja Hand Tracker & Blade Ribbon Engine
Utilizes Google MediaPipe HandLandmarker Tasks API with CPU acceleration.
Provides dual-hand fingertip/palm tracking, coordinate mirroring,
velocity estimation, and glowing blade trail generation.
"""

import os
import time
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from laser_sword import LaserSword

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

class BladeTrail:
    """Maintains a fading history of hand positions to render a ribbon blade swipe."""
    def __init__(self, max_points=12, max_age=0.22, color=(0, 240, 255)):
        self.points = []  # list of (x, y, time)
        self.max_points = max_points
        self.max_age = max_age
        self.color = color  # (B, G, R)
        self.velocity = 0.0

    def add_point(self, x, y, now=None):
        if now is None:
            now = time.time()
        if self.points:
            prev_x, prev_y, prev_t = self.points[-1]
            dt = max(1e-4, now - prev_t)
            dist = math.hypot(x - prev_x, y - prev_y)
            inst_v = dist / dt
            self.velocity = 0.7 * self.velocity + 0.3 * inst_v
        self.points.append((float(x), float(y), now))
        self.clean(now)

    def clean(self, now=None):
        if now is None:
            now = time.time()
        self.points = [(x, y, t) for (x, y, t) in self.points if (now - t) <= self.max_age]
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points:]

    def get_segments(self):
        """Returns list of consecutive line segments ((x1, y1), (x2, y2), alpha)."""
        now = time.time()
        self.clean(now)
        if len(self.points) < 2:
            return []
        segs = []
        n = len(self.points)
        for i in range(n - 1):
            x1, y1, t1 = self.points[i]
            x2, y2, t2 = self.points[i+1]
            age = now - t2
            alpha = max(0.1, min(1.0, 1.0 - (age / self.max_age)))
            segs.append(((x1, y1), (x2, y2), alpha))
        return segs

    def draw(self, surface_or_frame, is_cv2=True):
        """Renders glowing fading blade streak onto frame or surface."""
        segs = self.get_segments()
        if not segs:
            return
            
        b, g, r = self.color
        for (x1, y1), (x2, y2), alpha in segs:
            thick_outer = int(14 * alpha) + 2
            thick_inner = max(2, int(6 * alpha))
            
            if is_cv2:
                # Outer glow
                glow_color = (int(b * alpha * 0.5), int(g * alpha * 0.5), int(r * alpha * 0.5))
                cv2.line(surface_or_frame, (int(x1), int(y1)), (int(x2), int(y2)), glow_color, thick_outer)
                # Bright white-hot core
                core_color = (255, 255, 255)
                cv2.line(surface_or_frame, (int(x1), int(y1)), (int(x2), int(y2)), core_color, thick_inner)

class HandData:
    """Holds tracked landmarks, blade trail, and fist laser sword for one hand."""
    def __init__(self, hand_label="Right", color=(0, 220, 255)):
        self.label = hand_label  # "Left" or "Right"
        self.is_present = False
        self.confidence = 0.0
        self.index_tip = None       # (x, y) in screen coords
        self.middle_tip = None
        self.thumb_tip = None
        self.palm_center = None
        self.prev_index_tip = None
        self.trail = BladeTrail(color=color)
        self.last_seen = 0.0
        # Dedicated Fist Laser Sword instance
        self.sword = LaserSword(hand_label=hand_label)

    def update(self, landmarks, screen_w, screen_h, confidence, now, dt=0.033, auto_grip=False):
        self.is_present = True
        self.confidence = confidence
        self.last_seen = now
        
        # In MediaPipe, landmark 8 is index tip, 12 is middle tip, 4 is thumb tip, 0 is wrist, 9 is middle MCP
        idx_lm = landmarks[8]
        mid_lm = landmarks[12]
        thb_lm = landmarks[4]
        wrist_lm = landmarks[0]
        mcp_lm = landmarks[9]

        self.prev_index_tip = self.index_tip
        self.index_tip = (float(idx_lm.x * screen_w), float(idx_lm.y * screen_h))
        self.middle_tip = (float(mid_lm.x * screen_w), float(mid_lm.y * screen_h))
        self.thumb_tip = (float(thb_lm.x * screen_w), float(thb_lm.y * screen_h))
        
        # Palm center: midpoint between wrist and middle MCP
        self.palm_center = (
            0.5 * (wrist_lm.x + mcp_lm.x) * screen_w,
            0.5 * (wrist_lm.y + mcp_lm.y) * screen_h
        )
        
        # Update Fist-Controlled Laser Sword first
        self.sword.update_from_landmarks(landmarks, screen_w, screen_h, confidence, dt, auto_grip=auto_grip)
        # Blade trail follows active sword tip (invariant to index fingertip)
        if self.sword.is_active and self.sword.blade_tip:
            self.trail.add_point(self.sword.blade_tip[0], self.sword.blade_tip[1], now)
        elif self.sword.blade_tip:
            self.trail.add_point(self.sword.blade_tip[0], self.sword.blade_tip[1], now)
        else:
            self.trail.add_point(self.palm_center[0], self.palm_center[1], now)

    def mark_absent(self, now, dt=0.033):
        self.sword.mark_absent(dt)
        if (now - self.last_seen) > 0.15:
            self.is_present = False
            self.prev_index_tip = None
            self.index_tip = None
            self.trail.clean(now)

class HandTracker:
    """Manages MediaPipe Hand Landmarker and coordinates dual-hand laser swords."""
    def __init__(self, model_path=None, max_hands=2):
        self.model_path = model_path or MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Hand landmarker model not found at {self.model_path}")
            
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Left hand: Electric Cyan blade; Right hand: Flaming Amber/Ruby blade
        self.left_hand = HandData("Left", color=(255, 230, 20))
        self.right_hand = HandData("Right", color=(40, 130, 255))
        self.left_sword = self.left_hand.sword
        self.right_sword = self.right_hand.sword
        self.last_detection_time = time.time()
        self.prev_process_time = time.time()
        self.hands_detected_count = 0
        self.auto_grip = False
        self.hand_mode = "RIGHT_ONLY"  # "RIGHT_ONLY", "LEFT_ONLY", "SINGLE_AUTO", "BOTH"

    def set_sword_size(self, length_mult=1.0, width_mult=1.0):
        """Sets size modifiers on both laser swords."""
        self.left_sword.set_size_modifiers(length_mult, width_mult)
        self.right_sword.set_size_modifiers(length_mult, width_mult)

    def get_swords(self):
        """Returns list of both LaserSword instances."""
        return [self.left_sword, self.right_sword]

    def get_active_swords(self):
        """Returns currently active/ignited LaserSword instances."""
        return [s for s in [self.left_sword, self.right_sword] if s.is_active]

    def process_frame(self, frame_bgr, screen_w, screen_h, dt=None, auto_grip=None, hand_mode=None):
        """
        Receives flipped BGR frame from webcam.
        Runs fast CPU inference and updates tracked hands and laser swords.
        Supports single-hand enforcement ("RIGHT_ONLY", "LEFT_ONLY", "SINGLE_AUTO")
        and dual-wield ("BOTH").
        """
        now = time.time()
        if dt is None:
            dt = max(1e-4, min(0.066, now - self.prev_process_time))
        self.prev_process_time = now
        
        if auto_grip is not None:
            self.auto_grip = auto_grip
        if hand_mode is not None:
            self.hand_mode = hand_mode

        # Snapshot current sword poses before inference for continuous swept collision
        self.left_sword.set_previous_pose()
        self.right_sword.set_previous_pose()
        
        # Resize to fast inference resolution (e.g. 384x288) to eliminate latency on CPU
        infer_frame = cv2.resize(frame_bgr, (384, 288), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            # Graceful error handling
            print(f"[HandTracker] MediaPipe inference warning: {e}")
            result = None

        left_found = False
        right_found = False
        self.hands_detected_count = 0

        is_single_hand = self.hand_mode in ["RIGHT_ONLY", "LEFT_ONLY", "SINGLE_AUTO", "AUTO"]

        if result and result.hand_landmarks:
            self.hands_detected_count = len(result.hand_landmarks)
            self.last_detection_time = now
            
            for i, landmarks in enumerate(result.hand_landmarks):
                # Handedness category: 'Left' or 'Right'
                label = "Right"
                score = 0.9
                if result.handedness and i < len(result.handedness) and result.handedness[i]:
                    cat = result.handedness[i][0]
                    # Because camera frame was mirrored, invert label for intuitive first-person mapping
                    label = "Left" if cat.category_name == "Right" else "Right"
                    score = cat.score

                if self.hand_mode == "RIGHT_ONLY":
                    # Strictly single-hand right: always bind primary hand to right sword and break
                    self.right_hand.update(landmarks, screen_w, screen_h, score, now, dt=dt, auto_grip=self.auto_grip)
                    right_found = True
                    break
                elif self.hand_mode == "LEFT_ONLY":
                    # Strictly single-hand left: always bind primary hand to left sword and break
                    self.left_hand.update(landmarks, screen_w, screen_h, score, now, dt=dt, auto_grip=self.auto_grip)
                    left_found = True
                    break
                elif is_single_hand:
                    # Single-hand auto: allow only one hand at a time
                    if label == "Left":
                        self.left_hand.update(landmarks, screen_w, screen_h, score, now, dt=dt, auto_grip=self.auto_grip)
                        left_found = True
                    else:
                        self.right_hand.update(landmarks, screen_w, screen_h, score, now, dt=dt, auto_grip=self.auto_grip)
                        right_found = True
                    break
                else:
                    # BOTH (Dual-wield mode)
                    if label == "Left" and not left_found:
                        self.left_hand.update(landmarks, screen_w, screen_h, score, now, dt=dt, auto_grip=self.auto_grip)
                        left_found = True
                    elif label == "Right" and not right_found:
                        self.right_hand.update(landmarks, screen_w, screen_h, score, now, dt=dt, auto_grip=self.auto_grip)
                        right_found = True
                    elif not left_found:
                        self.left_hand.update(landmarks, screen_w, screen_h, score, now, dt=dt, auto_grip=self.auto_grip)
                        left_found = True
                    elif not right_found:
                        self.right_hand.update(landmarks, screen_w, screen_h, score, now, dt=dt, auto_grip=self.auto_grip)
                        right_found = True

        if not left_found:
            self.left_hand.mark_absent(now, dt=dt)
        if not right_found:
            self.right_hand.mark_absent(now, dt=dt)

        return self.left_hand, self.right_hand

    @property
    def any_hand_present(self):
        return self.left_hand.is_present or self.right_hand.is_present

    @property
    def seconds_since_last_hand(self):
        return time.time() - self.last_detection_time

    def get_active_slices(self, min_velocity=120.0):
        """
        Returns active blade slice segments ((p1, p2), hand_label).
        Strictly sourced from active Laser Sword blade trajectory.
        Index-fingertip fallback is completely removed.
        """
        slices = []
        for hand in [self.left_hand, self.right_hand]:
            sword = hand.sword
            if sword.is_active and sword.tracking_validity and sword.velocity >= min_velocity and sword.prev_tip:
                slices.append(((sword.prev_tip, sword.blade_tip), hand.label))
        return slices

    def draw_blade_effects(self, frame_bgr):
        """Renders laser blade trails and sword geometry for detected hands (no fingertip reticles)."""
        for hand in [self.left_hand, self.right_hand]:
            if hand.trail.points:
                hand.trail.draw(frame_bgr, is_cv2=True)
            if hand.sword.is_active or hand.sword.is_present:
                hand.sword.render_cv2(frame_bgr)

    def close(self):
        """Cleanly releases MediaPipe HandLandmarker detector resources."""
        if hasattr(self, "detector") and self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None
