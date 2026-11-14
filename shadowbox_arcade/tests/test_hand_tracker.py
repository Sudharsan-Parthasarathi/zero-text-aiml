import time
import numpy as np
import pytest
from hand_tracker import BladeTrail, HandData, HandTracker

def test_blade_trail():
    trail = BladeTrail(max_points=5, max_age=0.2)
    now = time.time()
    trail.add_point(100, 100, now)
    trail.add_point(150, 150, now + 0.05)
    
    assert len(trail.points) == 2
    assert trail.velocity > 0
    segs = trail.get_segments()
    assert len(segs) == 1

    # After aging past max_age
    trail.clean(now + 0.3)
    assert len(trail.points) == 0

def test_hand_data():
    hand = HandData("Right")
    assert not hand.is_present
    
    class DummyLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Create 21 dummy landmarks
    landmarks = [DummyLM(0.5, 0.5) for _ in range(21)]
    landmarks[8] = DummyLM(0.6, 0.4)  # Index tip
    
    now = time.time()
    hand.update(landmarks, 1000, 800, 0.95, now)
    
    assert hand.is_present
    assert hand.index_tip == (600.0, 320.0)
    assert hand.confidence == 0.95

    # Mark absent
    hand.mark_absent(now + 0.3)
    assert not hand.is_present

def test_hand_tracker_init_and_process():
    tracker = HandTracker()
    assert tracker.detector is not None

    # Process empty/black frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left, right = tracker.process_frame(dummy_frame, 640, 480)
    assert not left.is_present
    assert not right.is_present
    assert not tracker.any_hand_present

def test_get_active_slices_strictly_laser_swords():
    """
    Validates that active slices are strictly sourced from active Laser Swords.
    Fingertip motion fallback is completely removed.
    """
    tracker = HandTracker()
    right_hand = tracker.right_hand

    # 1. Inactive sword, moving fingertip -> MUST PRODUCE 0 SLICES
    right_hand.is_present = True
    right_hand.prev_index_tip = (100.0, 100.0)
    right_hand.index_tip = (300.0, 300.0)
    right_hand.trail.velocity = 600.0
    right_hand.sword.is_active = False

    slices = tracker.get_active_slices(min_velocity=120.0)
    assert len(slices) == 0

    # 2. Active sword, high velocity -> MUST PRODUCE 1 SLICE
    right_hand.sword.is_active = True
    right_hand.sword.tracking_validity = True
    right_hand.sword.velocity = 350.0
    right_hand.sword.prev_tip = (200.0, 200.0)
    right_hand.sword.blade_tip = (250.0, 280.0)

    slices = tracker.get_active_slices(min_velocity=120.0)
    assert len(slices) == 1
    assert slices[0][0] == ((200.0, 200.0), (250.0, 280.0))
    assert slices[0][1] == "Right"

    tracker.close()

def test_blade_trail_follows_blade_tip():
    """
    Validates that hand.trail records the sword blade tip, not index fingertip.
    """
    hand = HandData("Right")
    class DummyLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # 21 dummy landmarks with distinct tip and knuckles
    landmarks = [DummyLM(0.5, 0.5) for _ in range(21)]
    landmarks[0] = DummyLM(0.5, 0.7)   # Wrist
    landmarks[8] = DummyLM(0.1, 0.1)   # Index tip (far away from blade)

    now = time.time()
    hand.update(landmarks, 1000, 1000, 0.95, now, auto_grip=True)

    # Trail point must correspond to sword.blade_tip, NOT index_tip
    assert len(hand.trail.points) >= 1
    trail_pt = hand.trail.points[-1]
    assert trail_pt[0] == pytest.approx(hand.sword.blade_tip[0], abs=1e-3)
    assert trail_pt[1] == pytest.approx(hand.sword.blade_tip[1], abs=1e-3)
    assert trail_pt[0] != pytest.approx(hand.index_tip[0], abs=10.0)

def test_draw_blade_effects_renders_cleanly():
    """
    Validates that draw_blade_effects executes cleanly on OpenCV frames.
    """
    tracker = HandTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tracker.right_hand.is_present = True
    tracker.right_hand.sword.is_present = True
    tracker.right_hand.sword.is_active = True
    tracker.right_hand.sword.pommel = (100.0, 150.0)
    tracker.right_hand.sword.guard_center = (120.0, 120.0)
    tracker.right_hand.sword.blade_root = (125.0, 115.0)
    tracker.right_hand.sword.blade_tip = (160.0, 60.0)
    tracker.right_hand.sword.extension_pct = 1.0

    tracker.draw_blade_effects(frame)
    tracker.close()

def test_set_sword_size():
    tracker = HandTracker()
    tracker.set_sword_size(1.5, 1.2)
    assert tracker.left_sword.length_multiplier == 1.5
    assert tracker.left_sword.width_multiplier == 1.2
    assert tracker.right_sword.length_multiplier == 1.5
    assert tracker.right_sword.width_multiplier == 1.2
    tracker.close()

def test_single_hand_mode_right_only():
    tracker = HandTracker()
    tracker.hand_mode = "RIGHT_ONLY"

    class DummyLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [DummyLM(0.5, 0.5) for _ in range(21)]
    landmarks[0] = DummyLM(0.5, 0.7)

    # Mock detection result with 2 hands
    class MockCategory:
        def __init__(self, name, score=0.95):
            self.category_name = name
            self.score = score

    class MockResult:
        def __init__(self):
            self.hand_landmarks = [landmarks, landmarks]
            # MediaPipe category_name "Left" means user's Right hand
            self.handedness = [[MockCategory("Left")], [MockCategory("Right")]]

    tracker.detector.detect = lambda img: MockResult()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left, right = tracker.process_frame(dummy_frame, 640, 480)

    assert right.is_present
    assert not left.is_present
    assert len(tracker.get_active_swords()) <= 1
    tracker.close()

def test_single_hand_mode_left_only():
    tracker = HandTracker()
    tracker.hand_mode = "LEFT_ONLY"

    class DummyLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [DummyLM(0.5, 0.5) for _ in range(21)]
    landmarks[0] = DummyLM(0.5, 0.7)

    class MockCategory:
        def __init__(self, name, score=0.95):
            self.category_name = name
            self.score = score

    class MockResult:
        def __init__(self):
            self.hand_landmarks = [landmarks, landmarks]
            self.handedness = [[MockCategory("Right")], [MockCategory("Left")]]

    tracker.detector.detect = lambda img: MockResult()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left, right = tracker.process_frame(dummy_frame, 640, 480)

    assert left.is_present
    assert not right.is_present
    tracker.close()
