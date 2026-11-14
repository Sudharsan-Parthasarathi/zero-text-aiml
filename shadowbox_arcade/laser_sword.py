"""
=============================================================================
SHADOWBOX ARCADE: LASER SWORD SYSTEM (laser_sword.py)
Biomechanical Palm-Grip Coordinate Frame, One Euro Filter Stabilization,
Shortest-Path Angular Interpolation, Multi-Layer Plasma Rendering, and
Continuous Swept-Blade Geometry (Zero Finger-Pointer Artifacts).
=============================================================================
"""

import math
import time
import numpy as np
import pygame
import cv2

# Color Profiles (BGR for OpenCV, RGB for Pygame)
SWORD_PALETTES = {
    "Left": {
        "name": "Electric Cyan",
        "core_rgb": (255, 255, 255),
        "aura_rgb": (0, 235, 255),
        "outer_rgb": (0, 140, 255),
        "trail_rgb": (0, 220, 255),
        "hilt_glow_rgb": (0, 255, 240),
        "core_bgr": (255, 255, 255),
        "aura_bgr": (255, 235, 0),
        "outer_bgr": (255, 140, 0),
    },
    "Right": {
        "name": "Neon Amber",
        "core_rgb": (255, 255, 255),
        "aura_rgb": (255, 140, 25),
        "outer_rgb": (255, 45, 15),
        "trail_rgb": (255, 120, 20),
        "hilt_glow_rgb": (255, 100, 20),
        "core_bgr": (255, 255, 255),
        "aura_bgr": (25, 140, 255),
        "outer_bgr": (15, 45, 255),
    }
}

# -----------------------------------------------------------------------------
# 1. 1€ FILTER (ONE EURO FILTER) & ADAPTIVE STABILIZATION
# -----------------------------------------------------------------------------

class OneEuroFilter:
    """
    Casiez, Roussel, Vogel (CHI 2012) 1€ Filter.
    Adapts cutoff frequency dynamically based on signal speed:
    - At low speed: filters high-frequency landmark noise and eliminates jitter.
    - At high speed: reduces lag to near-zero for crisp, responsive swings.
    """
    def __init__(self, min_cutoff=1.2, beta=0.012, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    def reset(self, x=None):
        self.x_prev = float(x) if x is not None else None
        self.dx_prev = 0.0

    def filter(self, x, dt=None):
        x = float(x)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = max(1e-4, min(0.1, dt if dt is not None else 0.016))

        # 1. Filter derivative of signal
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # 2. Dynamic cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)

        # 3. Filter signal
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

class OneEuroFilter2D:
    """2D One Euro Filter for spatial coordinate stabilization."""
    def __init__(self, min_cutoff=1.2, beta=0.012, d_cutoff=1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self, pt=None):
        x = pt[0] if pt is not None else None
        y = pt[1] if pt is not None else None
        self.fx.reset(x)
        self.fy.reset(y)

    def filter(self, pt, dt=None):
        return (self.fx.filter(pt[0], dt), self.fy.filter(pt[1], dt))

class AngularFilter:
    """
    Adaptive angle filter with shortest-path interpolation across -pi/pi boundary.
    Damps resting angle jitter while tracking fast wrist rotations instantly.
    """
    def __init__(self, min_rate=0.28, max_rate=0.85, vel_scale=0.0015):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.vel_scale = vel_scale
        self.angle = None

    def reset(self, angle=None):
        self.angle = float(angle) if angle is not None else None

    def filter(self, target_angle, angular_vel_hint=0.0, dt=0.016):
        if self.angle is None:
            self.angle = target_angle
            return target_angle

        # Shortest angular difference
        diff = (target_angle - self.angle + math.pi) % (2.0 * math.pi) - math.pi
        rate = min(self.max_rate, max(self.min_rate, self.min_rate + abs(angular_vel_hint) * self.vel_scale))
        self.angle = (self.angle + rate * diff + math.pi) % (2.0 * math.pi) - math.pi
        return self.angle

# -----------------------------------------------------------------------------
# 2. GEOMETRIC COLLISION UTILITIES
# -----------------------------------------------------------------------------

def dist_point_to_line_segment(px, py, x1, y1, x2, y2):
    """
    Computes minimum Euclidean distance from point (px, py) to segment (x1, y1)-(x2, y2).
    Returns (distance, (proj_x, proj_y)).
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return math.hypot(px - x1, py - y1), (x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y), (proj_x, proj_y)

def dist_between_segments(p1, p2, p3, p4):
    """Computes minimum distance between two 2D line segments."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    d1, _ = dist_point_to_line_segment(x1, y1, x3, y3, x4, y4)
    d2, _ = dist_point_to_line_segment(x2, y2, x3, y3, x4, y4)
    d3, _ = dist_point_to_line_segment(x3, y3, x1, y1, x2, y2)
    d4, _ = dist_point_to_line_segment(x4, y4, x1, y1, x2, y2)
    return min(d1, d2, d3, d4)

class LaserSwordTrail:
    """Manages ribbon trail geometry of the swept sword blade."""
    def __init__(self, max_snapshots=8, max_age=0.18):
        self.snapshots = []  # [(root, tip, timestamp), ...]
        self.max_snapshots = max_snapshots
        self.max_age = max_age

    def add_pose(self, root, tip, now):
        self.snapshots.append((root, tip, now))
        self.clean(now)

    def clean(self, now=None):
        if now is None:
            now = time.time()
        self.snapshots = [s for s in self.snapshots if (now - s[2]) <= self.max_age]
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

# -----------------------------------------------------------------------------
# 3. LASER SWORD ENGINE (PALM-GRIP COORDINATE FRAME)
# -----------------------------------------------------------------------------

class LaserSword:
    """
    Fist-controlled sci-fi energy sword anchored inside the palm/knuckle region.
    - Grip Center: Anchored strictly between Wrist and Knuckles (MCP joints 5, 9, 13, 17).
    - Invariant to Landmark 12: Moving middle fingertip has ZERO influence on position/orientation.
    - One Euro Filter: Eliminates high-frequency landmark jitter while retaining instant swing response.
    - Symmetrical Pose: Slicing hitbox and visual render use the EXACT same filtered geometry.
    """

    def __init__(self, hand_label="Right", base_blade_length=235.0, blade_width=20.0):
        self.hand_label = hand_label  # "Left" or "Right"
        self.palette = SWORD_PALETTES.get(hand_label, SWORD_PALETTES["Right"])
        self.base_blade_length = float(base_blade_length)
        self.blade_length = float(base_blade_length)
        self.blade_width = float(blade_width)
        self.length_multiplier = 1.0
        self.width_multiplier = 1.0
        
        # Tracking State & Presence
        self.is_present = False
        self.tracking_confidence = 0.0
        self.tracking_validity = False
        
        # Spatial Anchors (in screen pixels)
        self.wrist = (0.0, 0.0)
        self.knuckle_center = (0.0, 0.0)
        self.palm_center = (0.0, 0.0)
        self.grip_anchor = (0.0, 0.0)     # Center of grip inside the fist
        self.pommel = (0.0, 0.0)          # Handle bottom endpoint towards wrist
        self.guard_center = (0.0, 0.0)     # Emitter guard at knuckle boundary
        self.blade_root = (0.0, 0.0)       # Blade base extending from guard
        self.blade_tip = (0.0, 0.0)        # Blade tip

        # Raw Diagnostic Coordinates (prior to filtering)
        self.raw_grip = (0.0, 0.0)
        self.raw_angle = 0.0
        self.mcp_joints = []
        self.fingertips = []
        
        # Palm Coordinate Frame Vectors
        self.primary_axis = (0.0, -1.0)   # Unit vector pointing outward through knuckles
        self.lateral_axis = (1.0, 0.0)    # Unit vector perpendicular across palm
        self.palm_scale = 50.0            # Dynamic distance from wrist to knuckles
        
        # Orientation (radians)
        self.angle = 0.0
        self.smooth_angle = 0.0
        
        # Dynamics & Speed
        self.velocity = 0.0
        self.prev_tip = None
        self.prev_root = None
        self.prev_pose_time = None
        
        # 1€ Filter & Angular Filter Instances
        self.filter_grip = OneEuroFilter2D(min_cutoff=1.2, beta=0.012, d_cutoff=1.0)
        self.filter_angle = AngularFilter(min_rate=0.28, max_rate=0.85, vel_scale=0.0015)
        
        # Fist Grip & Ignition State
        self.fist_closure_score = 0.0
        self.smooth_fist_score = 0.0
        self.is_gripped = False
        self.is_active = False
        self.fist_release_timer = 0.0      # Debounce hold to prevent release flicker
        self.extension_pct = 0.0           # 0.0 = retracted, 1.0 = fully ignited
        self.ignition_speed = 7.5          # Full ignition in ~0.13s
        self.retract_speed = 8.5           # Full retract in ~0.11s
        
        # Settings & Modifiers
        self.auto_grip = False
        self.min_swipe_velocity = 120.0    # pixels/second (stationary rejection gate)
        self.collision_margin = 14.0
        
        # Ribbon Trail
        self.trail = LaserSwordTrail(max_snapshots=7, max_age=0.16)
        
        # Spark Particle System
        self.sparks = []                   # [(x, y, vx, vy, color, life, max_life), ...]
        
        # Audio Trigger Flags
        self.just_ignited = False
        self.just_retracted = False

    def set_size_modifiers(self, length_mult=1.0, width_mult=1.0):
        """Adjusts the physical length and thickness of the plasma blade."""
        self.length_multiplier = max(0.5, min(2.5, float(length_mult)))
        self.width_multiplier = max(0.5, min(2.5, float(width_mult)))

    @property
    def velocity_vector(self):
        """Returns the 2D velocity vector (vx, vy) of the blade tip."""
        if self.prev_tip is not None and self.blade_tip is not None:
            return (self.blade_tip[0] - self.prev_tip[0], self.blade_tip[1] - self.prev_tip[1])
        return (math.cos(self.smooth_angle) * self.velocity, math.sin(self.smooth_angle) * self.velocity)

    def update_from_landmarks(self, landmarks, screen_w, screen_h, confidence, dt=0.016, auto_grip=False):
        """
        Updates sword pose and fist state strictly using the biomechanical palm frame.
        - Knuckle Center: Centroid of MCP landmarks 5, 9, 13, 17.
        - Grip Center: Weighted point between Wrist (0) and Knuckles (0.42 Wrist + 0.58 Knuckles).
        - Orientation: Primary forward vector (Knuckles - Wrist) with shortest-path angle filtering.
        - Landmark 12 (middle fingertip) is NEVER used in the sword anchor calculation.
        """
        now = time.time()
        was_absent = not self.is_present
        self.is_present = True
        self.tracking_confidence = confidence
        self.auto_grip = auto_grip
        self.just_ignited = False
        self.just_retracted = False
        dt = max(1e-4, min(0.066, dt if dt is not None else 0.016))

        # 1. Screen Coordinate Conversion
        def get_pt(idx):
            lm = landmarks[idx]
            return float(lm.x * screen_w), float(lm.y * screen_h)

        wrist_pt = get_pt(0)
        mcp_idx = get_pt(5)
        mcp_mid = get_pt(9)
        mcp_rng = get_pt(13)
        mcp_pnk = get_pt(17)
        self.mcp_joints = [mcp_idx, mcp_mid, mcp_rng, mcp_pnk]

        tip_thb = get_pt(4)
        tip_idx = get_pt(8)
        tip_mid = get_pt(12)
        tip_rng = get_pt(16)
        tip_pnk = get_pt(20)
        self.fingertips = [tip_thb, tip_idx, tip_mid, tip_rng, tip_pnk]

        # 2. Stable Knuckle Center (Centroid of all 4 MCP base joints)
        knuckle_x = 0.25 * (mcp_idx[0] + mcp_mid[0] + mcp_rng[0] + mcp_pnk[0])
        knuckle_y = 0.25 * (mcp_idx[1] + mcp_mid[1] + mcp_rng[1] + mcp_pnk[1])
        self.knuckle_center = (knuckle_x, knuckle_y)

        # Palm Reference Scale (Wrist to Knuckle distance)
        dist_w_k = math.hypot(knuckle_x - wrist_pt[0], knuckle_y - wrist_pt[1])
        self.palm_scale = max(16.0, dist_w_k)
        self.wrist = wrist_pt

        # Palm Center (Centroid of wrist and knuckle span)
        self.palm_center = (
            0.5 * (wrist_pt[0] + knuckle_x),
            0.5 * (wrist_pt[1] + knuckle_y)
        )

        # 3. Raw Grip Center (Inside the clenched fist, just proximal to MCP arch)
        # Strictly derived from Wrist and MCP Knuckles (Landmark 12 has ZERO influence)
        raw_gx = wrist_pt[0] + 0.58 * (knuckle_x - wrist_pt[0])
        raw_gy = wrist_pt[1] + 0.58 * (knuckle_y - wrist_pt[1])
        self.raw_grip = (raw_gx, raw_gy)

        # 4. One Euro Filter Stabilization on Grip Center
        if was_absent or self.filter_grip.fx.x_prev is None:
            self.filter_grip.reset(self.raw_grip)
            filtered_grip = self.raw_grip
            # Reset velocity on reacquisition to prevent velocity spikes
            self.velocity = 0.0
            self.prev_tip = None
            self.prev_root = None
        else:
            filtered_grip = self.filter_grip.filter(self.raw_grip, dt)
        self.grip_anchor = filtered_grip

        # 5. Natural Palm Orientation (Primary Forward Axis: Wrist -> Knuckles)
        dir_x = knuckle_x - wrist_pt[0]
        dir_y = knuckle_y - wrist_pt[1]
        
        if dist_w_k > 8.0:
            target_angle = math.atan2(dir_y, dir_x)
        else:
            target_angle = self.smooth_angle

        self.raw_angle = target_angle

        if was_absent or self.filter_angle.angle is None:
            self.filter_angle.reset(target_angle)
            self.smooth_angle = target_angle
        else:
            self.smooth_angle = self.filter_angle.filter(target_angle, angular_vel_hint=self.velocity, dt=dt)
        self.angle = self.smooth_angle

        # Unit Direction Vectors
        ux = math.cos(self.smooth_angle)
        uy = math.sin(self.smooth_angle)
        self.primary_axis = (ux, uy)
        self.lateral_axis = (-uy, ux)

        # 6. Proportional Sword Geometry (Rigid Weapon Held in Fist)
        # All offsets scale dynamically with palm_scale so sword looks natural at any camera distance
        hilt_length = 1.15 * self.palm_scale
        
        # Pommel (handle bottom extending behind palm towards wrist)
        self.pommel = (
            self.grip_anchor[0] - ux * (0.42 * hilt_length),
            self.grip_anchor[1] - uy * (0.42 * hilt_length)
        )
        
        # Guard Center (sits right at knuckle boundary of fist)
        self.guard_center = (
            self.grip_anchor[0] + ux * (0.58 * hilt_length),
            self.grip_anchor[1] + uy * (0.58 * hilt_length)
        )
        
        # Blade Base (emits from guard center)
        self.blade_root = (
            self.guard_center[0] + ux * (0.04 * self.palm_scale),
            self.guard_center[1] + uy * (0.04 * self.palm_scale)
        )
        
        # Blade Tip (extends outward from fist)
        # Dynamic distance scaling gives balanced blade length across camera depths
        scale_factor = (self.palm_scale / 65.0) ** 0.35
        scaled_blade_len = self.base_blade_length * scale_factor * self.length_multiplier
        current_len = scaled_blade_len * self.extension_pct
        self.blade_tip = (
            self.blade_root[0] + ux * current_len,
            self.blade_root[1] + uy * current_len
        )

        # 7. Fist Closure Recognition (Fingertips curled towards palm)
        # Measure Euclidean distance of each fingertip normalized by palm_scale
        d_thb = math.hypot(tip_thb[0] - self.palm_center[0], tip_thb[1] - self.palm_center[1]) / self.palm_scale
        d_idx = math.hypot(tip_idx[0] - self.palm_center[0], tip_idx[1] - self.palm_center[1]) / self.palm_scale
        d_mid = math.hypot(tip_mid[0] - self.palm_center[0], tip_mid[1] - self.palm_center[1]) / self.palm_scale
        d_rng = math.hypot(tip_rng[0] - self.palm_center[0], tip_rng[1] - self.palm_center[1]) / self.palm_scale
        d_pnk = math.hypot(tip_pnk[0] - self.palm_center[0], tip_pnk[1] - self.palm_center[1]) / self.palm_scale

        # Average extension ratio (closed fist < 0.65, open hand > 1.20)
        avg_finger_ext = (d_idx + d_mid + d_rng + d_pnk) / 4.0
        raw_score = max(0.0, min(1.0, (1.25 - avg_finger_ext) / (1.25 - 0.60)))
        self.fist_closure_score = raw_score

        # Temporal smoothing on closure score
        self.smooth_fist_score = 0.65 * self.smooth_fist_score + 0.35 * raw_score

        # Fist Grip Hysteresis (Ignite at >= 0.52, Release at < 0.36 with debounce)
        prev_gripped = self.is_gripped
        if self.auto_grip:
            self.is_gripped = True
            self.fist_release_timer = 0.0
        else:
            if not self.is_gripped and self.smooth_fist_score >= 0.52:
                self.is_gripped = True
                self.fist_release_timer = 0.0
            elif self.is_gripped:
                if self.smooth_fist_score < 0.36:
                    self.fist_release_timer += dt
                    if self.fist_release_timer >= 0.08:  # 80ms release debounce prevents swing flicker
                        self.is_gripped = False
                else:
                    self.fist_release_timer = 0.0

        # 8. Ignition Animation
        if self.is_gripped:
            if not prev_gripped:
                self.just_ignited = True
            self.extension_pct = min(1.0, self.extension_pct + dt * self.ignition_speed)
        else:
            if prev_gripped:
                self.just_retracted = True
            self.extension_pct = max(0.0, self.extension_pct - dt * self.retract_speed)

        self.is_active = (self.extension_pct > 0.05)
        self.tracking_validity = (self.is_present and confidence >= 0.40)

        # 9. Velocity & Movement Estimation
        if self.prev_tip is not None and dt > 1e-4 and not was_absent:
            tip_dist = math.hypot(self.blade_tip[0] - self.prev_tip[0], self.blade_tip[1] - self.prev_tip[1])
            root_dist = math.hypot(self.blade_root[0] - self.prev_root[0], self.blade_root[1] - self.prev_root[1])
            inst_vel = (0.75 * tip_dist + 0.25 * root_dist) / dt
            self.velocity = 0.65 * self.velocity + 0.35 * inst_vel
        else:
            self.velocity = 0.0

        # 10. Motion Trail
        if self.is_active and self.extension_pct > 0.30 and self.tracking_validity:
            self.trail.add_pose(self.blade_root, self.blade_tip, now)
        else:
            self.trail.clean(now)

        # 11. Sparks on High-Speed Swipes
        if self.is_active and self.velocity > 280.0 and len(self.sparks) < 25:
            frac = np.random.uniform(0.3, 1.0)
            sx = self.blade_root[0] + ux * current_len * frac
            sy = self.blade_root[1] + uy * current_len * frac
            spray_ang = self.smooth_angle + math.pi/2 + np.random.uniform(-0.5, 0.5)
            sp_speed = np.random.uniform(60, 220)
            color = self.palette["aura_rgb"]
            self.sparks.append([sx, sy, sp_speed * math.cos(spray_ang), sp_speed * math.sin(spray_ang), color, 0.0, np.random.uniform(0.15, 0.35)])

        self._update_sparks(dt)

    def _update_sparks(self, dt):
        surviving = []
        for s in self.sparks:
            s[0] += s[2] * dt
            s[1] += s[3] * dt
            s[5] += dt
            if s[5] < s[6]:
                surviving.append(s)
        self.sparks = surviving

    def mark_absent(self, dt=0.033):
        """Called when this hand is not detected in current frame."""
        self.is_present = False
        self.tracking_validity = False
        self.velocity = 0.0
        self.prev_tip = None
        self.prev_root = None
        self.extension_pct = max(0.0, self.extension_pct - dt * self.retract_speed * 1.5)
        self.is_active = (self.extension_pct > 0.05)
        self.trail.clean()
        self._update_sparks(dt)

    def set_previous_pose(self):
        """Caches current pose for continuous swept collision evaluation."""
        self.prev_root = self.blade_root
        self.prev_tip = self.blade_tip
        self.prev_pose_time = time.time()

    def check_swept_slice(self, fruit_pos, fruit_radius, min_swipe_velocity=None):
        """
        Continuous Swept Blade Collision.
        Uses the EXACT same blade pose that is visually rendered.
        Guaranteed zero slicing when tracking is invalid or blade is retracted.
        """
        if not self.is_active or not self.tracking_validity or self.extension_pct < 0.18:
            return False, None

        min_v = self.min_swipe_velocity if min_swipe_velocity is None else float(min_swipe_velocity)
        # Intentional Swipe Velocity Gate (prevents stationary sword camping)
        if self.velocity < min_v:
            return False, None

        fx, fy = fruit_pos
        effective_radius = fruit_radius + (self.blade_width * self.width_multiplier * 0.5) + self.collision_margin

        if self.prev_root is None or self.prev_tip is None:
            # Fallback to current blade segment
            p1 = self.blade_root
            p2 = self.blade_tip
            dist, proj = dist_point_to_line_segment(fx, fy, p1[0], p1[1], p2[0], p2[1])
            if dist <= effective_radius:
                return True, proj
            return False, None

        # 1. Test current blade segment
        dist_curr, pt_curr = dist_point_to_line_segment(fx, fy, self.blade_root[0], self.blade_root[1],
                                                        self.blade_tip[0], self.blade_tip[1])
        if dist_curr <= effective_radius:
            return True, pt_curr

        # 2. Test previous blade segment
        dist_prev, pt_prev = dist_point_to_line_segment(fx, fy, self.prev_root[0], self.prev_root[1],
                                                        self.prev_tip[0], self.prev_tip[1])
        if dist_prev <= effective_radius:
            return True, pt_prev

        # 3. Test swept quadrilateral via multi-point trajectory segments
        for s in (0.0, 0.25, 0.50, 0.75, 1.0):
            p_prev_x = self.prev_root[0] + s * (self.prev_tip[0] - self.prev_root[0])
            p_prev_y = self.prev_root[1] + s * (self.prev_tip[1] - self.prev_root[1])
            p_curr_x = self.blade_root[0] + s * (self.blade_tip[0] - self.blade_root[0])
            p_curr_y = self.blade_root[1] + s * (self.blade_tip[1] - self.blade_root[1])

            d, contact = dist_point_to_line_segment(fx, fy, p_prev_x, p_prev_y, p_curr_x, p_curr_y)
            if d <= effective_radius:
                return True, contact

        return False, None

    # -------------------------------------------------------------------------
    # RENDERING METHODS (Pygame Surface & OpenCV Frame)
    # -------------------------------------------------------------------------

    def render_pygame(self, surface):
        """
        Renders the sword rigidly held by the closed fist:
        - Handle: Extends from pommel to guard, centered at grip_anchor inside palm.
        - Guard Emitter: Positioned at knuckle boundary of the fist.
        - Plasma Energy Blade: Extends outward from blade_root through blade_tip.
        """
        if not self.is_present and self.extension_pct <= 0.01:
            return

        ux, uy = self.primary_axis
        nx, ny = self.lateral_axis

        # Proportional dimensions
        hw = max(7.0, 0.16 * self.palm_scale)

        px1, py1 = self.pommel
        px2, py2 = self.guard_center
        gx, gy = self.grip_anchor
        rx, ry = self.blade_root
        tx, ty = self.blade_tip

        # 1. Motion Trail Ribbon
        snaps = self.trail.snapshots
        if len(snaps) >= 2 and self.is_active:
            now = time.time()
            trail_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            for i in range(len(snaps) - 1):
                r1, t1, tm1 = snaps[i]
                r2, t2, tm2 = snaps[i+1]
                age = now - tm2
                alpha = max(0.0, min(1.0, 1.0 - (age / self.trail.max_age)))
                if alpha <= 0.05:
                    continue
                poly = [(int(r1[0]), int(r1[1])), (int(t1[0]), int(t1[1])),
                        (int(t2[0]), int(t2[1])), (int(r2[0]), int(r2[1]))]
                col = (*self.palette["trail_rgb"], int(140 * alpha))
                pygame.draw.polygon(trail_surf, col, poly)
            surface.blit(trail_surf, (0, 0))

        # 2. Render Hilt (Sits inside the clenched fist)
        hilt_poly = [
            (int(px1 + nx * hw), int(py1 + ny * hw)),
            (int(px2 + nx * hw), int(py2 + ny * hw)),
            (int(px2 - nx * hw), int(py2 - ny * hw)),
            (int(px1 - nx * hw), int(py1 - ny * hw))
        ]
        # Dark metallic handle body with cyber highlight border
        pygame.draw.polygon(surface, (36, 40, 48), hilt_poly)
        pygame.draw.polygon(surface, (150, 160, 180), hilt_poly, 2)

        # Handle Grip Ribs (Tactile ridges showing it is being held)
        for t_step in (0.35, 0.50, 0.65):
            rx_step = px1 + t_step * (px2 - px1)
            ry_step = py1 + t_step * (py2 - py1)
            pygame.draw.line(surface, (70, 75, 90),
                             (int(rx_step - nx * (hw - 1)), int(ry_step - ny * (hw - 1))),
                             (int(rx_step + nx * (hw - 1)), int(ry_step + ny * (hw - 1))), 2)

        # Pommel Cap (Handle bottom endpoint)
        pommel_color = (80, 90, 110)
        pygame.draw.circle(surface, pommel_color, (int(px1), int(py1)), int(hw + 1))

        # Emitter Guard Ring (Knuckle boundary where blade emerges)
        guard_color = self.palette["hilt_glow_rgb"] if self.is_gripped else (120, 120, 120)
        pygame.draw.circle(surface, guard_color, (int(px2), int(py2)), int(hw + 4))
        pygame.draw.circle(surface, (255, 255, 255), (int(px2), int(py2)), int(hw - 2))

        # Grip Indicator LED in Handle Center
        led_color = (0, 255, 120) if self.is_gripped else (255, 160, 0)
        pygame.draw.circle(surface, led_color, (int(gx), int(gy)), max(3, int(hw * 0.4)))

        # 3. Render Laser Energy Blade (Multi-Layer Plasma Glow)
        if self.is_active and self.extension_pct > 0.02:
            irx, iry = int(rx), int(ry)
            itx, ity = int(tx), int(ty)

            # Dynamic velocity glow boost
            vel_boost = min(1.35, 1.0 + (self.velocity / 800.0) * 0.35)
            b_width = max(14, int(self.blade_width * self.width_multiplier * (self.palm_scale / 65.0) ** 0.30))

            # Layer A: Soft Translucent Outer Aura
            aura_width = int(b_width * 1.8 * vel_boost)
            aura_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            aura_col = (*self.palette["outer_rgb"], 60)
            pygame.draw.line(aura_surf, aura_col, (irx, iry), (itx, ity), aura_width)
            pygame.draw.circle(aura_surf, aura_col, (itx, ity), aura_width // 2)

            # Layer B: Saturated Energy Beam Body
            body_width = int(b_width * 0.95 * vel_boost)
            body_col = (*self.palette["aura_rgb"], 180)
            pygame.draw.line(aura_surf, body_col, (irx, iry), (itx, ity), body_width)
            pygame.draw.circle(aura_surf, body_col, (itx, ity), body_width // 2)
            surface.blit(aura_surf, (0, 0))

            # Layer C: Blazing White-Hot Energy Core
            core_width = max(3, int(b_width * 0.35))
            pygame.draw.line(surface, self.palette["core_rgb"], (irx, iry), (itx, ity), core_width)
            pygame.draw.circle(surface, self.palette["core_rgb"], (itx, ity), core_width + 1)
            pygame.draw.circle(surface, self.palette["core_rgb"], (irx, iry), core_width + 2)

        # 4. Render Sparks
        for s in self.sparks:
            alpha = max(0.0, 1.0 - (s[5] / s[6]))
            r = max(2, int(4 * alpha))
            pygame.draw.circle(surface, s[4], (int(s[0]), int(s[1])), r)

    def render_cv2(self, frame_bgr):
        """Renders laser sword directly onto OpenCV BGR image."""
        if not self.is_present and self.extension_pct <= 0.01:
            return

        px1, py1 = int(self.pommel[0]), int(self.pommel[1])
        px2, py2 = int(self.guard_center[0]), int(self.guard_center[1])
        rx, ry = int(self.blade_root[0]), int(self.blade_root[1])
        tx, ty = int(self.blade_tip[0]), int(self.blade_tip[1])
        hw = max(5, int(0.16 * self.palm_scale))

        # Handle
        cv2.line(frame_bgr, (px1, py1), (px2, py2), (50, 50, 50), int(hw * 2))
        cv2.circle(frame_bgr, (px2, py2), hw + 2, self.palette["aura_bgr"], -1)

        # Blade
        if self.is_active and self.extension_pct > 0.02:
            b_width = max(12, int(self.blade_width * (self.palm_scale / 65.0) ** 0.30))
            cv2.line(frame_bgr, (rx, ry), (tx, ty), self.palette["outer_bgr"], int(b_width * 1.8))
            cv2.line(frame_bgr, (rx, ry), (tx, ty), self.palette["aura_bgr"], int(b_width * 0.95))
            cv2.line(frame_bgr, (rx, ry), (tx, ty), (255, 255, 255), max(3, int(b_width * 0.35)))
            cv2.circle(frame_bgr, (tx, ty), max(3, int(b_width * 0.4)), (255, 255, 255), -1)
