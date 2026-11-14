"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: TRACKING QUALITY EVALUATOR (tracking_quality.py)
Confidence, Visibility, and Dropout State Machine:
- Tracks presence, anatomical plausibility, and confidence metrics
- Evaluates states: TRACKING_OK, DEGRADED, LOST, REACQUIRING
- Smooth dropout fading (avoids sudden disappearance or teleportation)
- Safe attack gating: disables attack/slice generation during tracking degradation
=============================================================================
"""

import time
from enum import Enum
from typing import Optional, Tuple


class TrackingState(Enum):
    TRACKING_OK = "OK"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"


class TrackingQualityEvaluator:
    """
    Evaluates tracking confidence, maintains presence timers, and manages
    reacquisition states to prevent phantom swipes and sprite teleportation.
    """

    def __init__(self,
                 loss_timeout_sec: float = 0.15,
                 fade_duration_sec: float = 0.20,
                 reacquire_frames: int = 2,
                 min_ok_confidence: float = 0.50,
                 min_degraded_confidence: float = 0.25):
        self.loss_timeout = loss_timeout_sec
        self.fade_duration = fade_duration_sec
        self.reacquire_frames_needed = reacquire_frames
        self.min_ok_conf = min_ok_confidence
        self.min_deg_conf = min_degraded_confidence

        self.state = TrackingState.LOST
        self.last_seen_time: float = 0.0
        self.reacquire_counter: int = 0
        self.confidence: float = 0.0
        self.visibility: float = 0.0
        self.dropout_alpha: float = 0.0
        self.last_valid_pos: Optional[Tuple[float, float]] = None

    def reset(self):
        self.state = TrackingState.LOST
        self.last_seen_time = 0.0
        self.reacquire_counter = 0
        self.confidence = 0.0
        self.visibility = 0.0
        self.dropout_alpha = 0.0
        self.last_valid_pos = None

    def update(self, is_detected: bool, confidence: float, visibility: float,
               pos: Optional[Tuple[float, float]], now: Optional[float] = None) -> TrackingState:
        if now is None:
            now = time.time()

        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.visibility = max(0.0, min(1.0, float(visibility)))

        if is_detected and self.confidence >= self.min_deg_conf:
            self.last_seen_time = now
            if pos is not None:
                self.last_valid_pos = pos

            if self.state == TrackingState.LOST:
                # Entering reacquisition phase
                self.state = TrackingState.REACQUIRING
                self.reacquire_counter = 1
                self.dropout_alpha = 0.5
            elif self.state == TrackingState.REACQUIRING:
                self.reacquire_counter += 1
                if self.reacquire_counter >= self.reacquire_frames_needed:
                    self.state = TrackingState.TRACKING_OK if self.confidence >= self.min_ok_conf else TrackingState.DEGRADED
                self.dropout_alpha = min(1.0, self.dropout_alpha + 0.3)
            else:
                self.state = TrackingState.TRACKING_OK if self.confidence >= self.min_ok_conf else TrackingState.DEGRADED
                self.dropout_alpha = 1.0
        else:
            # Not detected or confidence too low
            time_since_seen = now - self.last_seen_time
            if time_since_seen > self.loss_timeout:
                self.state = TrackingState.LOST
                # Fade out gradually
                fade_elapsed = time_since_seen - self.loss_timeout
                self.dropout_alpha = max(0.0, 1.0 - (fade_elapsed / self.fade_duration))
            else:
                self.state = TrackingState.DEGRADED
                self.dropout_alpha = max(0.2, 1.0 - (time_since_seen / self.loss_timeout) * 0.5)

        return self.state

    @property
    def is_valid_for_attacks(self) -> bool:
        """Only emit new attack events (punches, slices) when tracking is rock solid."""
        return self.state == TrackingState.TRACKING_OK

    @property
    def is_visible(self) -> bool:
        return self.dropout_alpha > 0.05
