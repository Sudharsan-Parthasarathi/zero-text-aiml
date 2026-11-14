"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: FILTERS & ADAPTIVE STABILIZATION (filters.py)
Research-backed adaptive signal filtering:
- Casiez, Roussel, Vogel (CHI 2012) 1€ Filter (One Euro Filter)
- Shortest-path angular interpolation across -pi/pi boundary
- Anatomical outlier & velocity spike rejection
- Causal first and second-order motion estimation (velocity, acceleration)
=============================================================================
"""

import math
import time
from typing import Tuple, Optional


class OneEuroFilter:
    """
    Casiez, Roussel, Vogel (CHI 2012) 1€ Filter.
    Adapts cutoff frequency dynamically based on signal speed:
    - At low speed: filters high-frequency landmark noise and eliminates jitter.
    - At high speed: reduces lag to near-zero for crisp, responsive motion.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.015, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0

    def reset(self, x: Optional[float] = None):
        self.x_prev = float(x) if x is not None else None
        self.dx_prev = 0.0

    def filter(self, x: float, dt: Optional[float] = None) -> float:
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
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class OneEuroFilter2D:
    """2D One Euro Filter for spatial coordinate stabilization."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.015, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self, pt: Optional[Tuple[float, float]] = None):
        x = pt[0] if pt is not None else None
        y = pt[1] if pt is not None else None
        self.fx.reset(x)
        self.fy.reset(y)

    def filter(self, pt: Tuple[float, float], dt: Optional[float] = None) -> Tuple[float, float]:
        return (self.fx.filter(pt[0], dt), self.fy.filter(pt[1], dt))


class OneEuroFilter3D:
    """3D One Euro Filter for spatial coordinate stabilization (x, y, z)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.015, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fz = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self, pt: Optional[Tuple[float, float, float]] = None):
        x = pt[0] if pt is not None else None
        y = pt[1] if pt is not None else None
        z = pt[2] if pt is not None else None
        self.fx.reset(x)
        self.fy.reset(y)
        self.fz.reset(z)

    def filter(self, pt: Tuple[float, float, float], dt: Optional[float] = None) -> Tuple[float, float, float]:
        return (self.fx.filter(pt[0], dt), self.fy.filter(pt[1], dt), self.fz.filter(pt[2], dt))


class AngularFilter:
    """
    Adaptive angle filter with shortest-path interpolation across -pi/pi boundary.
    Damps resting angle jitter while tracking fast rotations instantly.
    """

    def __init__(self, min_rate: float = 0.28, max_rate: float = 0.85, vel_scale: float = 0.0015):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.vel_scale = vel_scale
        self.angle: Optional[float] = None

    def reset(self, angle: Optional[float] = None):
        self.angle = float(angle) if angle is not None else None

    def filter(self, target_angle: float, angular_vel_hint: float = 0.0, dt: float = 0.016) -> float:
        if self.angle is None:
            self.angle = target_angle
            return target_angle

        # Shortest angular difference across circular boundary
        diff = (target_angle - self.angle + math.pi) % (2.0 * math.pi) - math.pi
        rate = min(self.max_rate, max(self.min_rate, self.min_rate + abs(angular_vel_hint) * self.vel_scale))
        self.angle = (self.angle + rate * diff + math.pi) % (2.0 * math.pi) - math.pi
        return self.angle


class OutlierDetector:
    """
    Rejects physically implausible single-frame landmark jumps (e.g. tracking glitch).
    Preserves fast biological punches while clamping anomalous teleports.
    """

    def __init__(self, max_speed_px_per_sec: float = 3800.0, max_accel_px_per_sec2: float = 45000.0):
        self.max_speed = max_speed_px_per_sec
        self.max_accel = max_accel_px_per_sec2
        self.prev_pos: Optional[Tuple[float, float]] = None
        self.prev_vel: Tuple[float, float] = (0.0, 0.0)

    def reset(self, pos: Optional[Tuple[float, float]] = None):
        self.prev_pos = pos
        self.prev_vel = (0.0, 0.0)

    def sanitize(self, raw_pos: Tuple[float, float], dt: float) -> Tuple[Tuple[float, float], bool]:
        """
        Returns (sanitized_pos, is_outlier).
        If raw_pos is physically impossible, clamps displacement to max plausible limit.
        """
        if self.prev_pos is None or dt <= 1e-4:
            self.prev_pos = raw_pos
            self.prev_vel = (0.0, 0.0)
            return raw_pos, False

        dx = raw_pos[0] - self.prev_pos[0]
        dy = raw_pos[1] - self.prev_pos[1]
        dist = math.hypot(dx, dy)
        speed = dist / dt

        is_outlier = False
        if speed > self.max_speed:
            is_outlier = True
            scale = (self.max_speed * dt) / max(1e-4, dist)
            dx *= scale
            dy *= scale

        sanitized_x = self.prev_pos[0] + dx
        sanitized_y = self.prev_pos[1] + dy

        # Update velocity & position
        vx = dx / dt
        vy = dy / dt
        self.prev_pos = (sanitized_x, sanitized_y)
        self.prev_vel = (vx, vy)
        return (sanitized_x, sanitized_y), is_outlier


class MotionEstimator:
    """
    Computes smooth first-order (velocity) and second-order (acceleration) vectors.
    """

    def __init__(self, vel_smoothing: float = 0.35, accel_smoothing: float = 0.25):
        self.vel_smoothing = vel_smoothing
        self.accel_smoothing = accel_smoothing
        self.prev_pos: Optional[Tuple[float, float]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.speed: float = 0.0
        self.acceleration: Tuple[float, float] = (0.0, 0.0)
        self.accel_magnitude: float = 0.0

    def reset(self, pos: Optional[Tuple[float, float]] = None):
        self.prev_pos = pos
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.acceleration = (0.0, 0.0)
        self.accel_magnitude = 0.0

    def update(self, pos: Tuple[float, float], dt: float):
        if self.prev_pos is None or dt <= 1e-4:
            self.prev_pos = pos
            return

        dt = max(1e-4, min(0.1, dt))
        inst_vx = (pos[0] - self.prev_pos[0]) / dt
        inst_vy = (pos[1] - self.prev_pos[1]) / dt

        # Smooth velocity
        vx = (1.0 - self.vel_smoothing) * self.velocity[0] + self.vel_smoothing * inst_vx
        vy = (1.0 - self.vel_smoothing) * self.velocity[1] + self.vel_smoothing * inst_vy
        new_speed = math.hypot(vx, vy)

        # Smooth acceleration
        inst_ax = (vx - self.velocity[0]) / dt
        inst_ay = (vy - self.velocity[1]) / dt
        ax = (1.0 - self.accel_smoothing) * self.acceleration[0] + self.accel_smoothing * inst_ax
        ay = (1.0 - self.accel_smoothing) * self.acceleration[1] + self.accel_smoothing * inst_ay
        new_accel = math.hypot(ax, ay)

        self.prev_pos = pos
        self.velocity = (vx, vy)
        self.speed = new_speed
        self.acceleration = (ax, ay)
        self.accel_magnitude = new_accel
