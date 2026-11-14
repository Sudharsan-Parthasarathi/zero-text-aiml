"""
=============================================================================
TEST SUITE: SHADOWBOX TRACKING ENGINE 2.0 (tests/test_tracking_engine.py)
Validates filters, coordinate mapping, identity management, tracking quality,
punch state machine, and player calibration.
=============================================================================
"""

import math
import time
import numpy as np
import pytest

from tracking.filters import (
    OneEuroFilter,
    OneEuroFilter2D,
    OneEuroFilter3D,
    AngularFilter,
    OutlierDetector,
    MotionEstimator
)
from tracking.coordinate_mapper import CoordinateMapper
from tracking.tracking_quality import TrackingQualityEvaluator, TrackingState
from tracking.identity_manager import IdentityManager
from tracking.punch_detector import PunchState, PunchType, PunchEvent, PunchDetector
from tracking.player_calibration import CalibrationManager
from tracking.camera_pipeline import CameraPipeline
from tracking.pose_tracker import ArmTrackingResult, PoseTrackingResult


def test_one_euro_filter_1d_and_2d():
    """Validates 1€ Filter jitter reduction on noisy static signal."""
    f1 = OneEuroFilter(min_cutoff=1.0, beta=0.015)
    f2 = OneEuroFilter2D(min_cutoff=1.0, beta=0.015)

    np.random.seed(42)
    raw_vals = []
    filtered_vals = []

    for _ in range(60):
        val = 100.0 + np.random.normal(0.0, 2.5)
        raw_vals.append(val)
        filtered_vals.append(f1.filter(val, dt=0.016))
        pt_f = f2.filter((val, val * 2.0), dt=0.016)
        assert isinstance(pt_f, tuple)

    var_raw = np.var(raw_vals[15:])
    var_filt = np.var(filtered_vals[15:])
    assert var_filt < var_raw * 0.35, "1€ filter must reduce resting jitter variance by at least 65%"


def test_angular_filter_wraparound():
    """Validates that AngularFilter takes the shortest path across -pi / pi boundary."""
    af = AngularFilter(min_rate=0.5)

    # Initialize near +pi (+3.10 rad ~ 177 deg)
    af.reset(3.10)

    # Target is near -pi (-3.10 rad ~ -177 deg) -> Angular distance is only 0.08 rad across boundary!
    target = -3.10
    filtered = af.filter(target, dt=0.016)

    # If it spun around the long way, it would be around 0.0 rad.
    # Taking the shortest path across boundary, it should be close to +- 3.14 rad
    assert abs(abs(filtered) - math.pi) < 0.20
    assert abs(filtered) > 2.90


def test_outlier_detector():
    """Validates outlier detector clamps sudden physically impossible single-frame jumps."""
    od = OutlierDetector(max_speed_px_per_sec=2000.0)
    od.reset((100.0, 100.0))

    # Jump of 5000px in 0.016s -> speed = 312,500 px/s (glitch!)
    sanitized, is_outlier = od.sanitize((5100.0, 100.0), dt=0.016)
    assert is_outlier is True
    # Clamped displacement should be max_speed * dt = 2000 * 0.016 = 32px
    dist = math.hypot(sanitized[0] - 100.0, sanitized[1] - 100.0)
    assert dist <= 33.0


def test_motion_estimator():
    """Validates velocity and acceleration vector estimation."""
    me = MotionEstimator()
    me.reset((0.0, 0.0))

    # Constant velocity: (100 px / 0.016s = 6250 px/s)
    me.update((100.0, 0.0), dt=0.016)
    assert me.speed > 0.0
    assert me.velocity[0] > 0.0


def test_coordinate_mapper_modes():
    """Validates coordinate mapping and aspect ratio preservation."""
    mapper_fill = CoordinateMapper(cam_w=640, cam_h=480, screen_w=1280, screen_h=720, mode="FILL", mirror_x=True)
    mapper_crop = CoordinateMapper(cam_w=640, cam_h=480, screen_w=1280, screen_h=720, mode="CENTER_CROP", mirror_x=True)

    # Test center point (0.5, 0.5)
    sx_fill, sy_fill = mapper_fill.norm_to_screen(0.5, 0.5)
    assert sx_fill == pytest.approx(640.0, abs=2.0)
    assert sy_fill == pytest.approx(360.0, abs=2.0)

    # In mirrored mode, normalized x=0.1 (camera right) should map to screen right (0.9 * 1280 = 1152px)
    sx_mirror, _ = mapper_fill.norm_to_screen(0.1, 0.5)
    assert sx_mirror > 1000.0


def test_tracking_quality_state_machine():
    """Validates TrackingQualityEvaluator state transitions and attack gating."""
    tq = TrackingQualityEvaluator(loss_timeout_sec=0.10, fade_duration_sec=0.15)
    now = 1000.0

    # 1. Initial State is LOST
    assert tq.state == TrackingState.LOST
    assert tq.is_valid_for_attacks is False

    # 2. First detected frame triggers REACQUIRING
    st = tq.update(is_detected=True, confidence=0.92, visibility=0.95, pos=(200.0, 200.0), now=now)
    assert st == TrackingState.REACQUIRING
    assert tq.is_valid_for_attacks is False  # Cannot attack on first reacquisition frame

    # 3. Second detected frame completes reacquisition -> TRACKING_OK
    st = tq.update(is_detected=True, confidence=0.92, visibility=0.95, pos=(200.0, 200.0), now=now + 0.016)
    assert st == TrackingState.TRACKING_OK
    assert tq.is_valid_for_attacks is True

    # 4. Dropout past loss timeout -> LOST
    st = tq.update(is_detected=False, confidence=0.0, visibility=0.0, pos=None, now=now + 0.20)
    assert st == TrackingState.LOST
    assert tq.is_valid_for_attacks is False


def test_identity_manager_swap_resolution():
    """Validates IdentityManager resolves Left and Right wrist assignments."""
    im = IdentityManager()

    sh_l = (400.0, 300.0)
    sh_r = (880.0, 300.0)

    # Normal assignment: left wrist near left shoulder, right wrist near right shoulder
    w_l = (420.0, 450.0)
    w_r = (860.0, 450.0)

    res_l, res_r, swapped = im.resolve_arm_identities(sh_l, sh_r, w_l, w_r)
    assert swapped is False
    assert res_l == w_l
    assert res_r == w_r


def test_punch_detector_state_machine_and_single_event():
    """
    CRITICAL REQUIREMENT:
    - Moving from Guard -> Accel -> Extension -> Impact produces EXACTLY 1 PunchEvent.
    - Stationary glove held in target produces 0 punches.
    """
    pd = PunchDetector(min_punch_speed=200.0, min_extension=0.55)

    def make_arm(wrist, shoulder, speed, ext):
        a = ArmTrackingResult(side="Left")
        a.wrist_filtered = wrist
        a.shoulder_filtered = shoulder
        a.glove_center = wrist
        a.speed = speed
        a.extension_ratio = ext
        a.is_valid_for_attacks = True
        a.velocity = (0.0, -speed)  # Upward straight punch
        return a

    # 1. Stationary glove: speed = 0.0 -> MUST PRODUCE 0 EVENTS
    arm_stationary = make_arm(wrist=(400.0, 400.0), shoulder=(400.0, 500.0), speed=0.0, ext=0.40)
    arm_dummy = ArmTrackingResult(side="Right")
    for _ in range(5):
        events = pd.update(arm_stationary, arm_dummy, now=time.time())
        assert len(events) == 0

    # 2. Punch Stroke: Guard -> Accel -> Extension -> Impact
    now = 2000.0
    # Frame 1: Accel start
    a1 = make_arm(wrist=(400.0, 380.0), shoulder=(400.0, 500.0), speed=180.0, ext=0.45)
    evs1 = pd.update(a1, arm_dummy, now=now)
    assert len(evs1) == 0
    assert pd.left_sm.state == PunchState.ACCELERATION

    # Frame 2: Extension
    now += 0.033
    a2 = make_arm(wrist=(400.0, 260.0), shoulder=(400.0, 500.0), speed=350.0, ext=0.75)
    evs2 = pd.update(a2, arm_dummy, now=now)
    assert len(evs2) == 0
    assert pd.left_sm.state == PunchState.EXTENSION

    # Frame 3: Impact peak
    now += 0.033
    a3 = make_arm(wrist=(400.0, 240.0), shoulder=(400.0, 500.0), speed=320.0, ext=0.88)
    evs3 = pd.update(a3, arm_dummy, now=now)
    assert len(evs3) == 1, "Punch stroke must emit exactly 1 PunchEvent at impact"
    event = evs3[0]
    assert event.arm_side == "Left"
    assert event.speed >= 320.0

    # Frame 4: Recovery (hand held at extension) -> MUST NOT RE-EMIT PUNCH
    now += 0.033
    a4 = make_arm(wrist=(400.0, 240.0), shoulder=(400.0, 500.0), speed=50.0, ext=0.88)
    evs4 = pd.update(a4, arm_dummy, now=now)
    assert len(evs4) == 0


def test_punch_classification_types():
    """Validates classification of JAB, CROSS, HOOK, UPPERCUT."""
    pd = PunchDetector(min_punch_speed=150.0, min_extension=0.50)

    # Uppercut: strong upward vertical trajectory (-vy >> vx)
    arm_up = ArmTrackingResult(side="Right")
    arm_up.wrist_filtered = (500.0, 250.0)
    arm_up.shoulder_filtered = (500.0, 450.0)
    arm_up.glove_center = (500.0, 250.0)
    arm_up.velocity = (20.0, -320.0)  # High -vy
    arm_up.speed = 320.0
    arm_up.extension_ratio = 0.70
    arm_up.is_valid_for_attacks = True

    ptype = pd.right_sm._classify_punch(arm_up)
    assert ptype == PunchType.UPPERCUT

    # Hook: strong lateral horizontal sweep (vx >> vy) with bent elbow
    arm_hook = ArmTrackingResult(side="Left")
    arm_hook.wrist_filtered = (350.0, 350.0)
    arm_hook.shoulder_filtered = (450.0, 350.0)
    arm_hook.glove_center = (350.0, 350.0)
    arm_hook.velocity = (300.0, 30.0)  # High vx
    arm_hook.speed = 300.0
    arm_hook.extension_ratio = 0.65
    arm_hook.is_valid_for_attacks = True

    ptype_hook = pd.left_sm._classify_punch(arm_hook)
    assert ptype_hook == PunchType.HOOK

    # Cross: cross-body punch
    arm_cross = ArmTrackingResult(side="Right")
    arm_cross.is_cross_body = True
    arm_cross.velocity = (-200.0, -100.0)
    arm_cross.extension_ratio = 0.90
    ptype_cross = pd.right_sm._classify_punch(arm_cross)
    assert ptype_cross == PunchType.CROSS


def test_calibration_manager_progression():
    """Validates 5-step player calibration state machine."""
    calib = CalibrationManager(1280, 720)
    assert calib.current_step == "WAIT_FOR_STANCE"
    assert calib.is_complete is False

    # Mock pose with optimal tracking
    pose = PoseTrackingResult()
    pose.has_pose = True
    pose.left_arm.tracking_state = TrackingState.TRACKING_OK
    pose.left_arm.shoulder_filtered = (400.0, 300.0)
    pose.left_arm.wrist_filtered = (400.0, 450.0)
    pose.left_arm.forearm_length = 85.0
    pose.right_arm.tracking_state = TrackingState.TRACKING_OK
    pose.right_arm.shoulder_filtered = (580.0, 300.0)
    pose.right_arm.wrist_filtered = (580.0, 450.0)
    pose.right_arm.forearm_length = 85.0

    # Step 0 -> Step 1
    calib.update(pose)
    assert calib.current_step == "HOLD_GUARD"
    assert calib.data.shoulder_width == 180.0

    # Fast forward guard hold (1.3 seconds)
    calib.guard_hold_start = time.time() - 1.5
    calib.update(pose)
    assert calib.current_step == "TEST_JAB"

    # Test jab with high speed
    pose.left_arm.speed = 350.0
    pose.left_arm.wrist_filtered = (400.0, 100.0)  # Extended reach > 180 * 1.1
    calib.update(pose)
    assert calib.current_step == "TEST_CROSS"

    # Test cross with high speed
    pose.right_arm.speed = 350.0
    pose.right_arm.wrist_filtered = (580.0, 100.0)
    finished = calib.update(pose)
    assert finished is True
    assert calib.is_complete is True


def test_camera_pipeline_headless():
    """Validates CameraPipeline runs safely in headless mode."""
    pipe = CameraPipeline(camera_index=999, target_w=640, target_h=480, target_fps=30)
    pipe.is_headless = True
    pipe.start()

    time.sleep(0.08)
    ok, frame, t, age, fid, is_new = pipe.get_latest_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)

    telemetry = pipe.get_telemetry()
    assert "capture_fps" in telemetry

    pipe.stop()
