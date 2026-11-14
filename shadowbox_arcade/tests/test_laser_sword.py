"""
=============================================================================
TEST SUITE: LASER SWORD SYSTEM & FRUIT NINJA 2.0 INTERACTION MODEL
=============================================================================
"""

import os
import math
import time
import pytest
import numpy as np

# Ensure headless execution for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
pygame.init()

from hand_tracker import HandData, HandTracker
from laser_sword import LaserSword, LaserSwordTrail, SWORD_PALETTES
from fruit_physics import FruitObject, WaveSpawner, swept_blade_collision
from fruit_ninja_game import FruitNinjaGame

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def make_mock_landmarks(wrist=(0.5, 0.7), mcps=None, tips=None):
    lms = [MockLandmark(0.5, 0.5) for _ in range(21)]
    lms[0] = MockLandmark(wrist[0], wrist[1])
    if mcps is None:
        mcps = [(0.46, 0.58), (0.49, 0.56), (0.52, 0.57), (0.55, 0.59)]
    for idx, pt in zip([5, 9, 13, 17], mcps):
        lms[idx] = MockLandmark(pt[0], pt[1])

    if tips is None:
        tips = [(0.42, 0.54), (0.46, 0.40), (0.49, 0.38), (0.52, 0.39), (0.55, 0.42)]
    for idx, pt in zip([4, 8, 12, 16, 20], tips):
        lms[idx] = MockLandmark(pt[0], pt[1])
    return lms

def test_laser_sword_initialization_and_palettes():
    left_sword = LaserSword("Left")
    right_sword = LaserSword("Right")

    assert left_sword.hand_label == "Left"
    assert right_sword.hand_label == "Right"
    assert left_sword.palette["name"] == "Electric Cyan"
    assert right_sword.palette["name"] == "Neon Amber"

    assert left_sword.is_gripped is False
    assert left_sword.is_active is False
    assert left_sword.extension_pct == 0.0
    assert left_sword.base_blade_length == 235.0

def test_fist_closure_metric_and_hysteresis():
    sword = LaserSword("Right")

    # 1. Open Hand: Tips extended high above knuckles
    open_mcps = [(0.46, 0.60), (0.49, 0.58), (0.52, 0.59), (0.55, 0.61)]
    open_tips = [(0.42, 0.55), (0.46, 0.35), (0.49, 0.33), (0.52, 0.34), (0.55, 0.37)]
    open_lms = make_mock_landmarks(wrist=(0.50, 0.75), mcps=open_mcps, tips=open_tips)

    for _ in range(5):
        sword.update_from_landmarks(open_lms, 1280, 720, confidence=0.95, dt=0.033, auto_grip=False)

    assert sword.is_gripped is False
    assert sword.is_active is False
    assert sword.extension_pct == 0.0

    # 2. Closed Fist: Tips curled close to MCP knuckles and palm center
    fist_mcps = [(0.46, 0.60), (0.49, 0.58), (0.52, 0.59), (0.55, 0.61)]
    fist_tips = [(0.48, 0.64), (0.48, 0.63), (0.50, 0.63), (0.52, 0.64), (0.54, 0.65)]
    fist_lms = make_mock_landmarks(wrist=(0.50, 0.75), mcps=fist_mcps, tips=fist_tips)

    for _ in range(12):
        sword.update_from_landmarks(fist_lms, 1280, 720, confidence=0.95, dt=0.033, auto_grip=False)

    assert sword.is_gripped is True
    assert sword.is_active is True
    assert sword.extension_pct > 0.5

    # 3. Test Hysteresis: Slightly relaxing fist does not immediately collapse sword
    mid_tips = [(0.45, 0.56), (0.46, 0.52), (0.49, 0.51), (0.52, 0.53), (0.54, 0.55)]
    mid_lms = make_mock_landmarks(wrist=(0.50, 0.75), mcps=fist_mcps, tips=mid_tips)
    sword.update_from_landmarks(mid_lms, 1280, 720, confidence=0.95, dt=0.033, auto_grip=False)
    assert sword.is_gripped is True

    # 4. Opening fully drops grip
    for _ in range(15):
        sword.update_from_landmarks(open_lms, 1280, 720, confidence=0.95, dt=0.033, auto_grip=False)
    assert sword.is_gripped is False

def test_auto_grip_override():
    sword = LaserSword("Left")
    open_mcps = [(0.46, 0.60), (0.49, 0.58), (0.52, 0.59), (0.55, 0.61)]
    open_tips = [(0.42, 0.55), (0.46, 0.35), (0.49, 0.33), (0.52, 0.34), (0.55, 0.37)]
    open_lms = make_mock_landmarks(wrist=(0.50, 0.75), mcps=open_mcps, tips=open_tips)

    # With auto_grip=True, sword ignites even with open hand
    for _ in range(12):
        sword.update_from_landmarks(open_lms, 1280, 720, confidence=0.95, dt=0.033, auto_grip=True)

    assert sword.is_active is True
    assert sword.extension_pct > 0.8

def test_sword_ignition_and_retraction_flags():
    sword = LaserSword("Right")
    fist_mcps = [(0.46, 0.60), (0.49, 0.58), (0.52, 0.59), (0.55, 0.61)]
    fist_tips = [(0.48, 0.64), (0.48, 0.63), (0.50, 0.63), (0.52, 0.64), (0.54, 0.65)]
    fist_lms = make_mock_landmarks(wrist=(0.50, 0.75), mcps=fist_mcps, tips=fist_tips)

    open_mcps = [(0.46, 0.60), (0.49, 0.58), (0.52, 0.59), (0.55, 0.61)]
    open_tips = [(0.42, 0.55), (0.46, 0.35), (0.49, 0.33), (0.52, 0.34), (0.55, 0.37)]
    open_lms = make_mock_landmarks(wrist=(0.50, 0.75), mcps=open_mcps, tips=open_tips)

    # 1. Update until gripped
    saw_just_ignited = False
    for _ in range(5):
        sword.update_from_landmarks(fist_lms, 1280, 720, confidence=0.95, dt=0.033)
        if sword.just_ignited:
            saw_just_ignited = True

    assert sword.is_gripped is True
    assert saw_just_ignited is True
    # In next frame, just_ignited should be False
    sword.update_from_landmarks(fist_lms, 1280, 720, confidence=0.95, dt=0.033)
    assert sword.just_ignited is False

    # 2. Update with open hand until retracted
    saw_just_retracted = False
    for _ in range(15):
        sword.update_from_landmarks(open_lms, 1280, 720, confidence=0.95, dt=0.033)
        if sword.just_retracted:
            saw_just_retracted = True

    assert sword.is_gripped is False
    assert saw_just_retracted is True
    # Next frame, just_retracted should be False
    sword.update_from_landmarks(open_lms, 1280, 720, confidence=0.95, dt=0.033)
    assert sword.just_retracted is False

def test_swept_blade_collision_and_stationary_rejection():
    sword = LaserSword("Right")
    fist_mcps = [(0.46, 0.60), (0.49, 0.58), (0.52, 0.59), (0.55, 0.61)]
    fist_tips = [(0.48, 0.64), (0.48, 0.63), (0.50, 0.63), (0.52, 0.64), (0.54, 0.65)]
    fist_lms = make_mock_landmarks(wrist=(0.50, 0.75), mcps=fist_mcps, tips=fist_tips)

    # Fully extend sword
    for _ in range(15):
        sword.update_from_landmarks(fist_lms, 1280, 720, confidence=0.95, dt=0.033)
    assert sword.is_active is True
    assert sword.extension_pct >= 0.95

    # Fruit positioned near blade tip
    tip_x, tip_y = sword.blade_tip
    fruit = FruitObject(1, "watermelon", tip_x, tip_y, 0, 0, radius=50)

    # 1. Stationary Blade Test: velocity is zero -> should NOT slice
    sword.velocity = 0.0
    hit, contact = sword.check_swept_slice((fruit.x, fruit.y), fruit.radius, min_swipe_velocity=140.0)
    assert hit is False

    # 2. Moving Blade Test: velocity > min_swipe_velocity -> SHOULD slice
    sword.velocity = 450.0
    hit, contact = sword.check_swept_slice((fruit.x, fruit.y), fruit.radius, min_swipe_velocity=140.0)
    assert hit is True
    assert contact is not None
    assert math.hypot(contact[0] - fruit.x, contact[1] - fruit.y) <= fruit.radius + sword.blade_width + 15.0

    # 3. Off-target fruit test
    hit_miss, _ = sword.check_swept_slice((50.0, 50.0), fruit.radius)
    assert hit_miss is False

def test_difficulty_parameters_in_wave_spawner():
    spawner = WaveSpawner(1280, 720)

    # Easy mode: floatier gravity = 840.0
    easy_fruit = spawner._spawn_single(bomb_chance=0.0, mode="CLASSIC", difficulty="EASY")
    assert easy_fruit.gravity == 840.0

    # Expert mode: faster gravity = 1040.0
    expert_fruit = spawner._spawn_single(bomb_chance=0.0, mode="CLASSIC", difficulty="EXPERT")
    assert expert_fruit.gravity == 1040.0

def test_fruit_ninja_tutorial_state_progression():
    game = FruitNinjaGame(use_fullscreen=False)
    game.start_tutorial()

    assert game.state == "TUTORIAL"
    assert game.tutorial_step == 1

    # Step 1: Advance by detecting hand
    game.tracker.right_hand.is_present = True
    assert game.tracker.any_hand_present is True
    game.update_tutorial(0.8)
    assert game.tutorial_step == 2

    # Step 2: Advance by closed fist or auto-grip
    game.auto_grip = True
    game.update_tutorial(0.9)
    assert game.tutorial_step == 3

    # Step 3: Advance by moving hand
    game.tutorial_movement_distance = 400.0
    game.update_tutorial(0.016)
    assert game.tutorial_step == 4
    assert game.tutorial_target is not None

    # Step 4: Slice practice fruit
    game.slice_fruit(game.tutorial_target)
    assert game.tutorial_target.is_sliced is True
    game.tutorial_success_time = time.time() - 2.0
    game.update_tutorial(0.016)
    assert game.tutorial_step == 5

    # Step 5: Complete tutorial
    game.complete_tutorial()
    assert game.completed_tutorial is True
    assert game.state == "MENU"

def test_settings_persistence_in_fruit_ninja():
    game = FruitNinjaGame(use_fullscreen=False)
    
    # Test difficulty cycling
    initial_diff = game.difficulty
    game.cycle_difficulty()
    assert game.difficulty != initial_diff

    # Test auto grip toggle
    initial_ag = game.auto_grip
    game.toggle_auto_grip()
    assert game.auto_grip != initial_ag

    # Verify profile manager stored changes
    active_p = game.profile_mgr.get_active_player()
    prefs = active_p.get("preferences", {})
    assert prefs.get("difficulty") == game.difficulty
    assert prefs.get("auto_grip") == game.auto_grip

    game.release_camera()

def test_laser_sword_velocity_vector_and_hand_mode():
    sword = LaserSword("Right")
    # Initial state
    vec = sword.velocity_vector
    assert isinstance(vec, tuple)
    assert len(vec) == 2

    # Movement state
    sword.prev_tip = (100.0, 150.0)
    sword.blade_tip = (140.0, 200.0)
    vec2 = sword.velocity_vector
    assert vec2 == (40.0, 50.0)

    # Hand tracker hand_mode support
    tracker = HandTracker()
    assert hasattr(tracker, "hand_mode")
    tracker.hand_mode = "LEFT_ONLY"
    assert tracker.hand_mode == "LEFT_ONLY"
    tracker.close()

def test_landmark_12_invariance():
    """
    CRITICAL INVARIANCE REQUIREMENT:
    Moving Landmark 12 (middle fingertip) while keeping Wrist and MCPs fixed
    must produce exactly ZERO shift in grip anchor, guard, blade root, or blade tip.
    """
    sword1 = LaserSword("Right")
    sword2 = LaserSword("Right")

    fixed_wrist = (0.50, 0.70)
    fixed_mcps = [(0.46, 0.58), (0.49, 0.56), (0.52, 0.57), (0.55, 0.59)]

    # Hand 1: Landmark 12 at normal position
    tips1 = [(0.42, 0.54), (0.46, 0.40), (0.49, 0.38), (0.52, 0.39), (0.55, 0.42)]
    lms1 = make_mock_landmarks(wrist=fixed_wrist, mcps=fixed_mcps, tips=tips1)

    # Hand 2: Landmark 12 displaced drastically (y from 0.38 to 0.15 and x from 0.49 to 0.65)
    tips2 = [(0.42, 0.54), (0.46, 0.40), (0.65, 0.15), (0.52, 0.39), (0.55, 0.42)]
    lms2 = make_mock_landmarks(wrist=fixed_wrist, mcps=fixed_mcps, tips=tips2)

    sword1.update_from_landmarks(lms1, 1280, 720, confidence=0.95, dt=0.033, auto_grip=True)
    sword2.update_from_landmarks(lms2, 1280, 720, confidence=0.95, dt=0.033, auto_grip=True)

    # Assert Grip Center, Guard Center, Blade Root, and Blade Tip are bitwise/numerically identical
    assert abs(sword1.raw_grip[0] - sword2.raw_grip[0]) < 1e-6
    assert abs(sword1.raw_grip[1] - sword2.raw_grip[1]) < 1e-6
    assert abs(sword1.grip_anchor[0] - sword2.grip_anchor[0]) < 1e-6
    assert abs(sword1.grip_anchor[1] - sword2.grip_anchor[1]) < 1e-6
    assert abs(sword1.guard_center[0] - sword2.guard_center[0]) < 1e-6
    assert abs(sword1.guard_center[1] - sword2.guard_center[1]) < 1e-6
    assert abs(sword1.blade_root[0] - sword2.blade_root[0]) < 1e-6
    assert abs(sword1.blade_root[1] - sword2.blade_root[1]) < 1e-6
    assert abs(sword1.blade_tip[0] - sword2.blade_tip[0]) < 1e-6
    assert abs(sword1.blade_tip[1] - sword2.blade_tip[1]) < 1e-6

def test_knuckle_and_grip_center_derivation():
    """
    Validates exact geometric formulas:
    - Knuckle Center K = 0.25 * sum(MCP 5, 9, 13, 17)
    - Grip Center G = W + 0.58 * (K - W)
    """
    sword = LaserSword("Right")
    w = (0.50, 0.70)
    mcps = [(0.44, 0.56), (0.48, 0.54), (0.52, 0.55), (0.56, 0.57)]
    lms = make_mock_landmarks(wrist=w, mcps=mcps)

    sw, sh = 1000, 1000
    sword.update_from_landmarks(lms, sw, sh, confidence=0.90, dt=0.033)

    expected_kx = 0.25 * (0.44 + 0.48 + 0.52 + 0.56) * sw
    expected_ky = 0.25 * (0.56 + 0.54 + 0.55 + 0.57) * sh
    assert abs(sword.knuckle_center[0] - expected_kx) < 1e-3
    assert abs(sword.knuckle_center[1] - expected_ky) < 1e-3

    expected_gx = (w[0] * sw) + 0.58 * (expected_kx - (w[0] * sw))
    expected_gy = (w[1] * sh) + 0.58 * (expected_ky - (w[1] * sh))
    assert abs(sword.raw_grip[0] - expected_gx) < 1e-3
    assert abs(sword.raw_grip[1] - expected_gy) < 1e-3

def test_proportional_palm_scaling():
    """
    Validates that weapon dimensions scale with palm depth:
    Hilt length, guard size, and blade dimensions must scale with palm_scale.
    """
    sword_close = LaserSword("Right")
    sword_far = LaserSword("Right")

    # Hand close to camera: large wrist to knuckle distance
    close_lms = make_mock_landmarks(wrist=(0.50, 0.85), mcps=[(0.44, 0.45), (0.48, 0.43), (0.52, 0.44), (0.56, 0.46)])
    # Hand far from camera: small wrist to knuckle distance
    far_lms = make_mock_landmarks(wrist=(0.50, 0.65), mcps=[(0.48, 0.55), (0.49, 0.54), (0.51, 0.54), (0.52, 0.55)])

    sword_close.update_from_landmarks(close_lms, 1000, 1000, confidence=0.95, dt=0.033, auto_grip=True)
    sword_far.update_from_landmarks(far_lms, 1000, 1000, confidence=0.95, dt=0.033, auto_grip=True)

    assert sword_close.palm_scale > sword_far.palm_scale * 2.0
    # Guard offset from grip must be larger for the close hand
    dist_guard_close = math.hypot(sword_close.guard_center[0] - sword_close.grip_anchor[0],
                                  sword_close.guard_center[1] - sword_close.grip_anchor[1])
    dist_guard_far = math.hypot(sword_far.guard_center[0] - sword_far.grip_anchor[0],
                                sword_far.guard_center[1] - sword_far.grip_anchor[1])
    assert dist_guard_close > dist_guard_far * 2.0

def test_one_euro_filter_jitter_reduction():
    """
    Validates that OneEuroFilter dynamically suppresses high-frequency landmark jitter.
    Variance of filtered grip coordinates must be significantly lower than raw noisy coordinates.
    """
    sword = LaserSword("Right")
    np.random.seed(42)

    raw_xs, filtered_xs = [], []
    base_wrist = [0.50, 0.70]
    base_mcps = [(0.46, 0.58), (0.49, 0.56), (0.52, 0.57), (0.55, 0.59)]

    for _ in range(50):
        # Inject jitter (+- 0.006 normalized coords ~ +- 6 px)
        noise_x = np.random.normal(0.0, 0.005)
        noise_y = np.random.normal(0.0, 0.005)
        noisy_mcps = [(x + noise_x, y + noise_y) for x, y in base_mcps]
        lms = make_mock_landmarks(wrist=(base_wrist[0] + noise_x, base_wrist[1] + noise_y), mcps=noisy_mcps)

        sword.update_from_landmarks(lms, 1000, 1000, confidence=0.95, dt=0.033, auto_grip=True)
        raw_xs.append(sword.raw_grip[0])
        filtered_xs.append(sword.grip_anchor[0])

    var_raw = np.var(raw_xs[10:])
    var_filtered = np.var(filtered_xs[10:])
    # Filter should reduce variance substantially (at least 60% reduction)
    assert var_filtered < var_raw * 0.40

def test_reacquisition_reset_prevents_velocity_spikes():
    """
    When hand tracking is lost and re-established at a far location,
    the velocity filter and pose history must reset to prevent false swipe triggers.
    """
    sword = LaserSword("Right")
    lms1 = make_mock_landmarks(wrist=(0.20, 0.70))
    sword.update_from_landmarks(lms1, 1000, 1000, confidence=0.95, dt=0.033, auto_grip=True)

    # Mark absent for several frames
    for _ in range(5):
        sword.mark_absent(dt=0.033)
    assert sword.is_present is False

    # Reappear on opposite side of screen (600px jump)
    lms2 = make_mock_landmarks(wrist=(0.80, 0.30))
    sword.update_from_landmarks(lms2, 1000, 1000, confidence=0.95, dt=0.033, auto_grip=True)

    # Velocity must be reset on reacquisition frame
    assert sword.velocity == 0.0
    assert sword.prev_tip is None

def test_symmetrical_visual_vs_collision_geometry():
    """
    Ensures that swept slice collision testing uses the EXACT segment
    rendered visually (from sword.blade_root to sword.blade_tip).
    """
    sword = LaserSword("Right")
    fist_mcps = [(0.46, 0.60), (0.49, 0.58), (0.52, 0.59), (0.55, 0.61)]
    fist_tips = [(0.48, 0.64), (0.48, 0.63), (0.50, 0.63), (0.52, 0.64), (0.54, 0.65)]
    fist_lms = make_mock_landmarks(wrist=(0.50, 0.75), mcps=fist_mcps, tips=fist_tips)

    for _ in range(15):
        sword.update_from_landmarks(fist_lms, 1280, 720, confidence=0.95, dt=0.033)
    assert sword.is_active is True

    # Midpoint of visual blade segment
    mid_x = 0.5 * (sword.blade_root[0] + sword.blade_tip[0])
    mid_y = 0.5 * (sword.blade_root[1] + sword.blade_tip[1])

    sword.velocity = 350.0
    hit, contact = sword.check_swept_slice((mid_x, mid_y), fruit_radius=20.0, min_swipe_velocity=100.0)
    assert hit is True
    assert contact is not None
    assert math.hypot(contact[0] - mid_x, contact[1] - mid_y) < 5.0

def test_slicing_disabled_when_tracking_invalid_or_retracted():
    """
    Validates that no slices can register if tracking validity is False
    or the sword has retracted.
    """
    sword = LaserSword("Right")
    sword.velocity = 500.0
    sword.blade_root = (500.0, 500.0)
    sword.blade_tip = (500.0, 300.0)

    # 1. Sword retracted: is_active is False
    sword.is_active = False
    sword.tracking_validity = True
    sword.extension_pct = 0.0
    hit, _ = sword.check_swept_slice((500.0, 400.0), fruit_radius=30.0)
    assert hit is False

    # 2. Tracking validity False
    sword.is_active = True
    sword.extension_pct = 1.0
    sword.tracking_validity = False
    hit, _ = sword.check_swept_slice((500.0, 400.0), fruit_radius=30.0)
    assert hit is False

def test_left_and_right_palettes_and_symmetry():
    """
    Validates Left and Right hand distinct color palettes and orthogonal frame axes.
    """
    left_sword = LaserSword("Left")
    right_sword = LaserSword("Right")

    assert left_sword.palette["name"] == "Electric Cyan"
    assert right_sword.palette["name"] == "Neon Amber"

    # Forward u and lateral n vectors must be perpendicular: u . n == 0
    lms = make_mock_landmarks(wrist=(0.50, 0.70))
    left_sword.update_from_landmarks(lms, 1000, 1000, confidence=0.9, dt=0.033)
    ux, uy = left_sword.primary_axis
    nx, ny = left_sword.lateral_axis
    dot = ux * nx + uy * ny
    assert abs(dot) < 1e-5

def test_fruit_ninja_debug_overlay_toggle_and_render():
    """
    Validates F3 debug overlay toggle and renders cleanly without exception.
    """
    game = FruitNinjaGame(use_fullscreen=False)
    assert game.show_debug_overlay is False

    # Toggle overlay
    game.show_debug_overlay = True
    assert game.show_debug_overlay is True

    # Test rendering debug overlay on surface
    test_surface = pygame.Surface((game.screen_w, game.screen_h))
    game.render_debug_overlay(test_surface)

    # Populate dummy landmarks on right sword
    lms = make_mock_landmarks(wrist=(0.50, 0.70))
    game.tracker.right_sword.update_from_landmarks(lms, game.screen_w, game.screen_h, confidence=0.92, dt=0.033, auto_grip=True)
    game.render_debug_overlay(test_surface)

    game.release_camera()
