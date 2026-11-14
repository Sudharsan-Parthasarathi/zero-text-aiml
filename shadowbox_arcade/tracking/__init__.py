"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0 (tracking/__init__.py)
Unified Motion Tracking, Spatial Stabilization & Gesture Recognition Engine
=============================================================================
"""

from .filters import (
    OneEuroFilter,
    OneEuroFilter2D,
    OneEuroFilter3D,
    AngularFilter,
    OutlierDetector,
    MotionEstimator
)

from .coordinate_mapper import CoordinateMapper

from .tracking_quality import (
    TrackingState,
    TrackingQualityEvaluator
)

from .identity_manager import IdentityManager

from .camera_pipeline import CameraPipeline

from .pose_tracker import (
    ArmTrackingResult,
    PoseTrackingResult,
    PoseTracker
)

from .punch_detector import (
    PunchState,
    PunchType,
    PunchEvent,
    ArmPunchStateMachine,
    PunchDetector
)

from .player_calibration import (
    CalibrationData,
    CalibrationManager
)

from .tracking_debug import TrackingDebugVisualizer

__all__ = [
    "OneEuroFilter",
    "OneEuroFilter2D",
    "OneEuroFilter3D",
    "AngularFilter",
    "OutlierDetector",
    "MotionEstimator",
    "CoordinateMapper",
    "TrackingState",
    "TrackingQualityEvaluator",
    "IdentityManager",
    "CameraPipeline",
    "ArmTrackingResult",
    "PoseTrackingResult",
    "PoseTracker",
    "PunchState",
    "PunchType",
    "PunchEvent",
    "ArmPunchStateMachine",
    "PunchDetector",
    "CalibrationData",
    "CalibrationManager",
    "TrackingDebugVisualizer",
]
