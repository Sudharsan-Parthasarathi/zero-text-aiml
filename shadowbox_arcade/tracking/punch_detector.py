"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: PUNCH STATE MACHINE & CLASSIFIER (punch_detector.py)
Biomechanical Punch Recognition & Single-Event Impact Gating:
- States: GUARD -> ACCELERATION -> EXTENSION -> IMPACT -> RECOVERY -> GUARD
- Classifies: JAB, CROSS, HOOK, UPPERCUT
- Evaluates: Wrist velocity, elbow extension ratio, trajectory angle, cross-body reach
- Prevents stationary glove camping: exactly 1 event emitted per physical punch stroke
=============================================================================
"""

import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

from .pose_tracker import ArmTrackingResult


class PunchState(Enum):
    GUARD = "GUARD"
    ACCELERATION = "ACCEL"
    EXTENSION = "EXTEND"
    IMPACT = "IMPACT"
    RECOVERY = "RECOVER"


class PunchType(Enum):
    JAB = "JAB"
    CROSS = "CROSS"
    HOOK = "HOOK"
    UPPERCUT = "UPPERCUT"


@dataclass
class PunchEvent:
    """Emitted precisely once per physical punch stroke."""
    arm_side: str  # "Left" or "Right"
    punch_type: PunchType
    glove_pos: Tuple[float, float]
    speed: float  # px/s
    speed_kmh: float
    timestamp: float
    extension_ratio: float
    is_critical: bool = False


class ArmPunchStateMachine:
    """Tracks punch progression for a single arm."""

    def __init__(self, arm_side: str,
                 min_punch_speed: float = 240.0,
                 min_extension: float = 0.58,
                 cooldown_sec: float = 0.12):
        self.arm_side = arm_side
        self.min_punch_speed = min_punch_speed
        self.min_extension = min_extension
        self.cooldown_sec = cooldown_sec

        self.state = PunchState.GUARD
        self.last_punch_time: float = 0.0
        self.peak_speed: float = 0.0
        self.stroke_start_pos: Tuple[float, float] = (0.0, 0.0)
        self.stroke_max_ext: float = 0.0
        self.prev_reach: float = 0.0

    def reset(self):
        self.state = PunchState.GUARD
        self.last_punch_time = 0.0
        self.peak_speed = 0.0
        self.stroke_max_ext = 0.0
        self.prev_reach = 0.0

    def update(self, arm: ArmTrackingResult, now: Optional[float] = None) -> Optional[PunchEvent]:
        if now is None:
            now = time.time()

        if not arm.is_valid_for_attacks:
            self.state = PunchState.GUARD
            return None

        # Reach from shoulder to wrist
        reach = math.hypot(arm.wrist_filtered[0] - arm.shoulder_filtered[0],
                           arm.wrist_filtered[1] - arm.shoulder_filtered[1])
        reach_delta = reach - self.prev_reach
        self.prev_reach = reach

        speed = arm.speed
        ext = arm.extension_ratio
        event: Optional[PunchEvent] = None

        if self.state == PunchState.GUARD:
            # Transition to ACCELERATION if moving outward with momentum
            if speed >= self.min_punch_speed * 0.65 and reach_delta > 0:
                if (now - self.last_punch_time) >= self.cooldown_sec:
                    self.state = PunchState.ACCELERATION
                    self.peak_speed = speed
                    self.stroke_start_pos = arm.glove_center
                    self.stroke_max_ext = ext

        elif self.state == PunchState.ACCELERATION:
            if speed > self.peak_speed:
                self.peak_speed = speed
            if ext > self.stroke_max_ext:
                self.stroke_max_ext = ext

            # If arm is extending past minimum punch threshold, advance to EXTENSION
            if ext >= self.min_extension or speed >= self.min_punch_speed:
                self.state = PunchState.EXTENSION
            elif speed < self.min_punch_speed * 0.40 and reach_delta <= 0:
                # Abort false start
                self.state = PunchState.GUARD

        elif self.state == PunchState.EXTENSION:
            if speed > self.peak_speed:
                self.peak_speed = speed
            if ext > self.stroke_max_ext:
                self.stroke_max_ext = ext

            # Trigger IMPACT when velocity decelerates or extension peaks
            is_peak = (reach_delta <= 0.0) or (speed < self.peak_speed * 0.85) or (ext >= 0.88)
            if is_peak and self.peak_speed >= self.min_punch_speed and self.stroke_max_ext >= self.min_extension:
                self.state = PunchState.IMPACT
                self.last_punch_time = now

                # Classify punch type
                ptype = self._classify_punch(arm)
                event = PunchEvent(
                    arm_side=self.arm_side,
                    punch_type=ptype,
                    glove_pos=arm.glove_center,
                    speed=self.peak_speed,
                    speed_kmh=self.peak_speed * 0.04,
                    timestamp=now,
                    extension_ratio=self.stroke_max_ext
                )
            elif reach_delta < -8.0 and speed < self.min_punch_speed * 0.5:
                # Retracted without distinct impact
                self.state = PunchState.RECOVERY

        elif self.state == PunchState.IMPACT:
            # Immediate transition to recovery after emitting event
            self.state = PunchState.RECOVERY

        elif self.state == PunchState.RECOVERY:
            # Return to guard when hand moves back close to body
            if ext < 0.42 or (reach_delta <= 0 and speed < self.min_punch_speed * 0.45):
                self.state = PunchState.GUARD

        return event

    def force_impact_event(self, arm: ArmTrackingResult, now: Optional[float] = None) -> Optional[PunchEvent]:
        """
        Called when physical contact with an active target occurs during punch stroke.
        Emits single event and transitions state machine to recovery.
        """
        if now is None:
            now = time.time()

        if (now - self.last_punch_time) < self.cooldown_sec:
            return None

        self.last_punch_time = now
        self.state = PunchState.RECOVERY
        ptype = self._classify_punch(arm)
        return PunchEvent(
            arm_side=self.arm_side,
            punch_type=ptype,
            glove_pos=arm.glove_center,
            speed=max(self.min_punch_speed, arm.speed),
            speed_kmh=max(self.min_punch_speed, arm.speed) * 0.04,
            timestamp=now,
            extension_ratio=arm.extension_ratio
        )

    def _classify_punch(self, arm: ArmTrackingResult) -> PunchType:
        """Classifies punch style based on trajectory kinematics."""
        vx, vy = arm.velocity

        # Uppercut: strong upward vertical trajectory (-vy in screen coords)
        if -vy > 1.35 * abs(vx) and -vy > 180.0:
            return PunchType.UPPERCUT

        # Hook: strong lateral horizontal sweep with bent elbow
        if abs(vx) > 1.40 * abs(vy) and arm.extension_ratio < 0.82:
            return PunchType.HOOK

        # Cross: punch crossing torso midline
        if arm.is_cross_body:
            return PunchType.CROSS

        # Default: direct straight Jab
        return PunchType.JAB


class PunchDetector:
    """Manages punch state machines for both Left and Right arms."""

    def __init__(self, min_punch_speed: float = 240.0, min_extension: float = 0.58):
        self.left_sm = ArmPunchStateMachine("Left", min_punch_speed=min_punch_speed, min_extension=min_extension)
        self.right_sm = ArmPunchStateMachine("Right", min_punch_speed=min_punch_speed, min_extension=min_extension)

    def reset(self):
        self.left_sm.reset()
        self.right_sm.reset()

    def update(self, left_arm: ArmTrackingResult, right_arm: ArmTrackingResult,
               now: Optional[float] = None) -> List[PunchEvent]:
        """Updates both arms and returns any newly triggered PunchEvents."""
        events: List[PunchEvent] = []
        ev_l = self.left_sm.update(left_arm, now)
        if ev_l:
            events.append(ev_l)

        ev_r = self.right_sm.update(right_arm, now)
        if ev_r:
            events.append(ev_r)

        return events
