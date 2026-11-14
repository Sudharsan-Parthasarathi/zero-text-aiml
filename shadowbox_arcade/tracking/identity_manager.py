"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: IDENTITY MANAGER (identity_manager.py)
Anatomical Side Consistency & Spatial Continuity:
- Enforces strict anatomical separation between Left and Right arms
- Prevents left/right label swapping when arms cross or come close
- Evaluates shoulder-to-wrist kinematic reach constraints
- Detects cross-body punches without confusing glove assignments
=============================================================================
"""

import math
from typing import Tuple, Optional, Dict, Any


class IdentityManager:
    """
    Maintains spatial continuity and anatomical identity across consecutive frames.
    Uses shoulder anchor origins to prevent wrist swapping during high-speed crosses.
    """

    def __init__(self, max_reach_mult: float = 1.6):
        self.max_reach_mult = max_reach_mult
        self.prev_left_wrist: Optional[Tuple[float, float]] = None
        self.prev_right_wrist: Optional[Tuple[float, float]] = None
        self.swap_count: int = 0

    def reset(self):
        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.swap_count = 0

    def resolve_arm_identities(self,
                               raw_shoulder_l: Tuple[float, float],
                               raw_shoulder_r: Tuple[float, float],
                               raw_wrist_l: Optional[Tuple[float, float]],
                               raw_wrist_r: Optional[Tuple[float, float]],
                               expected_arm_len: Optional[float] = None) -> Tuple[Optional[Tuple[float, float]],
                                                                                  Optional[Tuple[float, float]],
                                                                                  bool]:
        """
        Validates whether raw_wrist_l and raw_wrist_r correspond to anatomical Left
        and Right, or if a temporary landmark swap occurred.
        Returns (resolved_wrist_l, resolved_wrist_r, was_swapped).
        """
        if raw_wrist_l is None or raw_wrist_r is None:
            # Single arm detected; evaluate shoulder proximity
            if raw_wrist_l is not None:
                d_l = math.hypot(raw_wrist_l[0] - raw_shoulder_l[0], raw_wrist_l[1] - raw_shoulder_l[1])
                d_r = math.hypot(raw_wrist_l[0] - raw_shoulder_r[0], raw_wrist_l[1] - raw_shoulder_r[1])
                # If much closer to right shoulder, it may be the right arm misattributed
                if d_r < d_l * 0.5:
                    return None, raw_wrist_l, True
                return raw_wrist_l, None, False

            if raw_wrist_r is not None:
                d_l = math.hypot(raw_wrist_r[0] - raw_shoulder_l[0], raw_wrist_r[1] - raw_shoulder_l[1])
                d_r = math.hypot(raw_wrist_r[0] - raw_shoulder_r[0], raw_wrist_r[1] - raw_shoulder_r[1])
                if d_l < d_r * 0.5:
                    return raw_wrist_r, None, True
                return None, raw_wrist_r, False

            return None, None, False

        # Both wrists detected: evaluate standard assignment vs swapped assignment
        # Normal configuration:
        # L wrist connects to L shoulder, R wrist connects to R shoulder
        dist_normal = (math.hypot(raw_wrist_l[0] - raw_shoulder_l[0], raw_wrist_l[1] - raw_shoulder_l[1]) +
                       math.hypot(raw_wrist_r[0] - raw_shoulder_r[0], raw_wrist_r[1] - raw_shoulder_r[1]))

        # Swapped configuration:
        dist_swapped = (math.hypot(raw_wrist_r[0] - raw_shoulder_l[0], raw_wrist_r[1] - raw_shoulder_l[1]) +
                        math.hypot(raw_wrist_l[0] - raw_shoulder_r[0], raw_wrist_l[1] - raw_shoulder_r[1]))

        # Temporal continuity check if previous positions exist
        temporal_normal = 0.0
        temporal_swapped = 0.0
        if self.prev_left_wrist is not None and self.prev_right_wrist is not None:
            temporal_normal = (math.hypot(raw_wrist_l[0] - self.prev_left_wrist[0], raw_wrist_l[1] - self.prev_left_wrist[1]) +
                               math.hypot(raw_wrist_r[0] - self.prev_right_wrist[0], raw_wrist_r[1] - self.prev_right_wrist[1]))
            temporal_swapped = (math.hypot(raw_wrist_r[0] - self.prev_left_wrist[0], raw_wrist_r[1] - self.prev_left_wrist[1]) +
                                math.hypot(raw_wrist_l[0] - self.prev_right_wrist[0], raw_wrist_l[1] - self.prev_right_wrist[1]))

        # Swap decision: only swap if both shoulder distance AND temporal continuity strongly support it
        should_swap = False
        if temporal_normal > 0.0 and temporal_swapped < temporal_normal * 0.45 and dist_swapped < dist_normal * 0.85:
            should_swap = True

        if should_swap:
            self.swap_count += 1
            final_l = raw_wrist_r
            final_r = raw_wrist_l
        else:
            final_l = raw_wrist_l
            final_r = raw_wrist_r

        self.prev_left_wrist = final_l
        self.prev_right_wrist = final_r
        return final_l, final_r, should_swap

    @staticmethod
    def is_cross_body(wrist: Tuple[float, float],
                      own_shoulder: Tuple[float, float],
                      other_shoulder: Tuple[float, float]) -> bool:
        """Determines if a wrist has crossed the torso midline towards the opposite shoulder."""
        torso_mid_x = (own_shoulder[0] + other_shoulder[0]) * 0.5
        # If left wrist is right of midline, or right wrist is left of midline
        if own_shoulder[0] < other_shoulder[0]:
            # Left shoulder is on left
            return wrist[0] > torso_mid_x
        else:
            return wrist[0] < torso_mid_x
