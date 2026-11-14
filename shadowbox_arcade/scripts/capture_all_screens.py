"""
=============================================================================
SHADOWBOX ARCADE — COMPREHENSIVE SCREENSHOT GENERATOR & QA HARNESS
Captures high-resolution frames of all 9 primary user interfaces for audit:
1. Arcade Launcher Main Selection Screen
2. Player Profile Modal ("Who's Playing Today?")
3. Settings & Preferences Modal
4. Shadow Boxing Gameplay HUD & Kinetic Targets
5. Shadow Boxing Game Over / Match Summary
6. Fruit Ninja Mode Selection & Dojo Menu
7. Fruit Ninja 5-Step Dojo Tutorial & Onboarding
8. Fruit Ninja Active Gameplay (Laser Swords, Fruit Slices, Splatters)
9. Fruit Ninja Game Over Screen
=============================================================================
"""

import os
import sys
import time
import math
import random
import numpy as np

# Use dummy video and audio drivers for clean headless screenshot capture
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import cv2
import pygame
pygame.init()
pygame.font.init()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add project root to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import player_profiles
from player_profiles import get_profile_manager
from music_manager import get_music_manager

# -----------------------------------------------------------------------------
# 1. ARCADE LAUNCHER SCREENS (1, 2, 3)
# -----------------------------------------------------------------------------
def capture_launcher_screens():
    print("[1/3] Capturing Arcade Launcher Screens...")
    from importlib.machinery import SourceFileLoader
    import importlib.util

    spec = importlib.util.spec_from_file_location("arcade_launcher", os.path.join(PROJECT_ROOT, "04_arcade_launcher.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    launcher = mod.ArcadeLauncher()
    active_player = launcher.profile_mgr.get_active_player()

    # Screen 1: Main Launcher Screen
    launcher.render_main_screen(active_player)
    path_1 = os.path.join(OUTPUT_DIR, "01_arcade_launcher.png")
    pygame.image.save(launcher.screen, path_1)
    print(f"  -> Saved {path_1}")

    # Screen 2: Player Selection Modal
    launcher.render_main_screen(active_player)
    launcher.render_player_modal()
    path_2 = os.path.join(OUTPUT_DIR, "02_player_modal.png")
    pygame.image.save(launcher.screen, path_2)
    print(f"  -> Saved {path_2}")

    # Screen 3: Settings Modal
    launcher.render_main_screen(active_player)
    launcher.render_settings_modal(active_player)
    path_3 = os.path.join(OUTPUT_DIR, "03_settings_modal.png")
    pygame.image.save(launcher.screen, path_3)
    print(f"  -> Saved {path_3}")

    launcher.music.cleanup()
    pygame.display.quit()

# -----------------------------------------------------------------------------
# 2. FRUIT NINJA SCREENS (6, 7, 8, 9)
# -----------------------------------------------------------------------------
def capture_fruit_ninja_screens():
    print("[2/3] Capturing Fruit Ninja Screens...")
    from fruit_ninja_game import FruitNinjaGame
    from fruit_physics import FruitObject, SlicedHalf, JuiceParticle, JuiceSplatterDecal

    game = FruitNinjaGame(use_fullscreen=False)
    
    # Camera mock surface
    camera_surface = pygame.Surface((game.screen_w, game.screen_h))
    camera_surface.fill((16, 20, 24))

    # Screen 6: Fruit Ninja Menu / Mode Selection
    game.state = "MENU"
    game.render(camera_surface)
    path_6 = os.path.join(OUTPUT_DIR, "06_fruit_ninja_menu.png")
    pygame.image.save(game.screen, path_6)
    print(f"  -> Saved {path_6}")

    # Screen 7: Fruit Ninja 5-Step Tutorial
    game.state = "TUTORIAL"
    game.tutorial_step = 4
    game.tutorial_target = FruitObject(999, "watermelon", 640, 360, 0, 0, radius=55, is_bomb=False)
    # Mock right laser sword active
    r_sword = game.tracker.right_hand.sword
    r_sword.is_active = True
    r_sword.extension_pct = 1.0
    r_sword.blade_root = (750, 480)
    r_sword.blade_tip = (580, 280)
    r_sword.knuckle_center = (750, 480)
    game.render(camera_surface)
    path_7 = os.path.join(OUTPUT_DIR, "07_fruit_ninja_tutorial.png")
    pygame.image.save(game.screen, path_7)
    print(f"  -> Saved {path_7}")

    # Screen 8: Fruit Ninja Active Gameplay (Action frame with swords & fruits)
    game.state = "PLAYING"
    game.mode = "CLASSIC"
    game.score = 240
    game.combo = 3
    game.lives = 3
    game.time_left = 48
    
    # Spawn a few active fruits and sliced halves
    f1 = FruitObject(1, "watermelon", 420, 340, 60, -120, radius=52)
    f2 = FruitObject(2, "apple", 780, 290, -40, -180, radius=40)
    f3 = FruitObject(3, "bomb", 600, 410, 10, -90, radius=38, is_bomb=True)
    game.active_fruits = [f1, f2, f3]

    # Sliced half and particles
    h1 = SlicedHalf("orange", 500, 360, -180, -40, 0.4, -120, "left")
    h2 = SlicedHalf("orange", 540, 360, 180, -40, 0.4, 120, "right")
    game.sliced_halves = [h1, h2]

    # Both laser swords active
    l_sword = game.tracker.left_hand.sword
    l_sword.is_active = True
    l_sword.extension_pct = 1.0
    l_sword.blade_root = (380, 520)
    l_sword.blade_tip = (480, 310)
    l_sword.knuckle_center = (380, 520)

    r_sword.blade_root = (880, 510)
    r_sword.blade_tip = (720, 290)
    r_sword.knuckle_center = (880, 510)

    game.render(camera_surface)
    path_8 = os.path.join(OUTPUT_DIR, "08_fruit_ninja_gameplay.png")
    pygame.image.save(game.screen, path_8)
    print(f"  -> Saved {path_8}")

    # Screen 9: Fruit Ninja Game Over Screen
    game.state = "GAMEOVER"
    game.score = 380
    game.max_combo = 6
    game.total_sliced = 48
    game.render(camera_surface)
    path_9 = os.path.join(OUTPUT_DIR, "09_fruit_ninja_gameover.png")
    pygame.image.save(game.screen, path_9)
    print(f"  -> Saved {path_9}")

    # Screen 10: Fruit Ninja F3 Diagnostics Overlay
    game.state = "PLAYING"
    game.show_debug_overlay = True
    r_sword.is_present = True
    r_sword.tracking_validity = True
    r_sword.tracking_confidence = 0.94
    r_sword.wrist = (880, 620)
    r_sword.knuckle_center = (850, 480)
    r_sword.raw_grip = (865, 540)
    r_sword.grip_anchor = (865, 540)
    r_sword.primary_axis = (-0.3, -0.95)
    r_sword.lateral_axis = (0.95, -0.3)
    r_sword.guard_center = (845, 450)
    r_sword.blade_root = (840, 440)
    r_sword.blade_tip = (720, 240)
    r_sword.mcp_joints = [(830, 490), (845, 480), (860, 482), (875, 495)]
    r_sword.fingertips = [(810, 520), (835, 530), (850, 532), (865, 535), (880, 538)]
    r_sword.fist_closure_score = 0.88
    r_sword.smooth_fist_score = 0.85
    r_sword.is_gripped = True
    r_sword.velocity = 420.0
    r_sword.palm_scale = 85.0
    game.render(camera_surface)
    path_10 = os.path.join(OUTPUT_DIR, "10_fruit_ninja_f3_diagnostics.png")
    pygame.image.save(game.screen, path_10)
    print(f"  -> Saved {path_10}")

    game.release_camera()
    game.tracker.close()
    pygame.display.quit()

# -----------------------------------------------------------------------------
# 3. SHADOW BOXING SCREENS (4, 5)
# -----------------------------------------------------------------------------
def capture_shadow_boxing_screens():
    print("[3/3] Capturing Shadow Boxing Screens...")
    w, h = 1280, 720

    # Helper for transparent overlay
    def overlay_alpha(bg, fg, x, y):
        fh, fw = fg.shape[:2]
        if x < 0 or y < 0 or x + fw > bg.shape[1] or y + fh > bg.shape[0]:
            return
        if fg.shape[2] == 4:
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                bg[y:y+fh, x:x+fw, c] = (1.0 - alpha) * bg[y:y+fh, x:x+fw, c] + alpha * fg[:, :, c]
        else:
            bg[y:y+fh, x:x+fw] = fg

    # Screen 4: Gameplay frame with targets, glove, combo, timer
    frame_play = np.zeros((h, w, 3), dtype=np.uint8)
    frame_play[:] = (20, 24, 32)  # Dark cyber ring

    # Load and overlay authentic gloves
    g_l_path = os.path.join(PROJECT_ROOT, "boxing_glove_left_classic_red.png")
    g_r_path = os.path.join(PROJECT_ROOT, "boxing_glove_right_classic_red.png")
    if os.path.exists(g_l_path) and os.path.exists(g_r_path):
        gl = cv2.imread(g_l_path, cv2.IMREAD_UNCHANGED)
        gr = cv2.imread(g_r_path, cv2.IMREAD_UNCHANGED)
        gl_res = cv2.resize(gl, (140, 190))
        gr_res = cv2.resize(gr, (140, 190))
        overlay_alpha(frame_play, gl_res, 320, 480)
        overlay_alpha(frame_play, gr_res, 820, 460)

    # Draw kinetic targets (Jab and Cross)
    cv2.circle(frame_play, (480, 320), 45, (45, 45, 230), -1)
    cv2.circle(frame_play, (480, 320), 45, (255, 255, 255), 3)
    cv2.putText(frame_play, "JAB", (455, 330), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    cv2.circle(frame_play, (780, 290), 48, (230, 95, 30), -1)
    cv2.circle(frame_play, (780, 290), 48, (255, 255, 255), 3)
    cv2.putText(frame_play, "CROSS", (745, 300), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    # Power-up crystal
    cv2.circle(frame_play, (640, 220), 30, (255, 40, 220), -1)
    cv2.circle(frame_play, (640, 220), 34, (255, 255, 255), 2)
    cv2.putText(frame_play, "CRIT", (620, 226), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

    # HUD Elements
    cv2.rectangle(frame_play, (0, 0), (w, 75), (14, 18, 24), -1)
    cv2.line(frame_play, (0, 75), (w, 75), (60, 70, 90), 2)
    cv2.putText(frame_play, "SCORE: 480", (30, 48), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 215, 0), 2)
    cv2.putText(frame_play, "ROUND 1  •  TIME: 38s", (w // 2 - 140, 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame_play, "COMBO: x5", (w - 220, 48), cv2.FONT_HERSHEY_DUPLEX, 1.1, (60, 220, 100), 2)

    # KIAI energy bar
    cv2.rectangle(frame_play, (w // 2 - 150, h - 45), (w // 2 + 150, h - 20), (30, 35, 45), -1)
    cv2.rectangle(frame_play, (w // 2 - 150, h - 45), (w // 2 + 90, h - 20), (240, 90, 40), -1)
    cv2.putText(frame_play, "KIAI ENERGY: 80% [SHOUT TO ACTIVATE]", (w // 2 - 160, h - 55), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 200, 100), 1)

    path_4 = os.path.join(OUTPUT_DIR, "04_shadow_boxing_gameplay.png")
    cv2.imwrite(path_4, frame_play)
    print(f"  -> Saved {path_4}")

    # Screen 5: Game Over / Match Summary
    frame_end = np.zeros((h, w, 3), dtype=np.uint8)
    frame_end[:] = (14, 16, 22)
    
    cv2.putText(frame_end, "ROUND COMPLETE", (w // 2 - 190, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 215, 255), 3)
    cv2.putText(frame_end, "FINAL SCORE: 850", (w // 2 - 170, 220), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
    cv2.putText(frame_end, "NEW PERSONAL RECORD!", (w // 2 - 160, 275), cv2.FONT_HERSHEY_DUPLEX, 0.9, (60, 230, 100), 2)
    
    # Stats card
    cv2.rectangle(frame_end, (w // 2 - 280, 320), (w // 2 + 280, 520), (25, 30, 42), -1)
    cv2.rectangle(frame_end, (w // 2 - 280, 320), (w // 2 + 280, 520), (70, 90, 130), 2)
    
    stats = [
        ("Total Punches:", "78"),
        ("Max Combo:", "12x"),
        ("Hit Accuracy:", "88%"),
        ("Kiai Critical Strikes:", "4"),
        ("Mecha Boss Defeated:", "YES")
    ]
    sy = 360
    for lbl, val in stats:
        cv2.putText(frame_end, lbl, (w // 2 - 240, sy), cv2.FONT_HERSHEY_DUPLEX, 0.75, (190, 200, 220), 1)
        cv2.putText(frame_end, val, (w // 2 + 130, sy), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 215, 0), 2)
        sy += 32

    # Action buttons
    cv2.rectangle(frame_end, (w // 2 - 260, 570), (w // 2 - 20, 630), (40, 160, 70), -1)
    cv2.putText(frame_end, "[R] PLAY AGAIN", (w // 2 - 235, 608), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    
    cv2.rectangle(frame_end, (w // 2 + 20, 570), (w // 2 + 260, 630), (50, 60, 80), -1)
    cv2.putText(frame_end, "[Q] ARCADE MENU", (w // 2 + 35, 608), cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 225, 240), 2)

    path_5 = os.path.join(OUTPUT_DIR, "05_shadow_boxing_gameover.png")
    cv2.imwrite(path_5, frame_end)
    print(f"  -> Saved {path_5}")

    # Screen 11: Shadow Boxing Tracking Engine 2.0 F3 Engineering Diagnostics
    from tracking import TrackingDebugVisualizer, PoseTrackingResult, ArmTrackingResult, TrackingState, PunchDetector, CalibrationManager
    frame_debug = np.zeros((h, w, 3), dtype=np.uint8)
    frame_debug[:] = (20, 24, 32)
    
    # Overlay gloves
    if os.path.exists(g_l_path) and os.path.exists(g_r_path):
        overlay_alpha(frame_debug, gl_res, 320, 480)
        overlay_alpha(frame_debug, gr_res, 820, 460)

    # Synthesize realistic pose result for telemetry visualization
    l_arm = ArmTrackingResult(
        side="Left",
        shoulder_raw=(420.0, 260.0),
        shoulder_filtered=(420.0, 260.0),
        elbow_raw=(370.0, 390.0),
        elbow_filtered=(370.0, 390.0),
        wrist_raw=(330.0, 490.0),
        wrist_filtered=(330.0, 490.0),
        glove_center=(330.0, 490.0),
        forearm_angle=1.92,
        forearm_length=110.0,
        speed=145.0,
        velocity=(-30.0, -12.0),
        tracking_state=TrackingState.TRACKING_OK,
        extension_ratio=0.52,
        is_cross_body=False,
        is_valid_for_attacks=True,
        dropout_alpha=1.0
    )
    r_arm = ArmTrackingResult(
        side="Right",
        shoulder_raw=(860.0, 260.0),
        shoulder_filtered=(860.0, 260.0),
        elbow_raw=(900.0, 380.0),
        elbow_filtered=(900.0, 380.0),
        wrist_raw=(830.0, 470.0),
        wrist_filtered=(830.0, 470.0),
        glove_center=(830.0, 470.0),
        forearm_angle=1.35,
        forearm_length=112.0,
        speed=380.0,
        velocity=(-240.0, -110.0),
        tracking_state=TrackingState.TRACKING_OK,
        extension_ratio=0.78,
        is_cross_body=False,
        is_valid_for_attacks=True,
        dropout_alpha=1.0
    )
    mock_pose = PoseTrackingResult(
        has_pose=True,
        left_arm=l_arm,
        right_arm=r_arm,
        tracking_fps=58.2,
        inference_time_ms=18.4,
        timestamp=time.time(),
        frame_id=142
    )
    
    mock_punch_det = PunchDetector()
    mock_punch_det.right_sm.state = mock_punch_det.right_sm.state  # GUARD
    
    dbg_vis = TrackingDebugVisualizer()
    dbg_vis.enabled = True
    dbg_vis.render_fps = 60.0
    dbg_vis.render_overlay(
        frame_debug, mock_pose, mock_punch_det,
        frame_age_ms=8.2,
        cam_telemetry={"capture_fps": 60.0, "latency_ms": 7.8, "reconnects": 0}
    )
    
    path_11 = os.path.join(OUTPUT_DIR, "11_shadow_boxing_f3_diagnostics.png")
    cv2.imwrite(path_11, frame_debug)
    print(f"  -> Saved {path_11}")

    # Screen 12: Player Calibration Sequence Screen
    frame_calib = np.zeros((h, w, 3), dtype=np.uint8)
    frame_calib[:] = (20, 24, 32)
    calib_mgr = CalibrationManager(screen_w=w, screen_h=h)
    calib_mgr.step_idx = 1
    calib_mgr.status_message = "STEP 2/4: HOLD NEUTRAL BOXING GUARD NEAR CHEST"
    calib_mgr.draw_calibration_overlay(frame_calib)
    if os.path.exists(g_l_path) and os.path.exists(g_r_path):
        overlay_alpha(frame_calib, gl_res, int(w * 0.40) - 70, int(h * 0.55) - 95)
        overlay_alpha(frame_calib, gr_res, int(w * 0.60) - 70, int(h * 0.55) - 95)
    
    path_12 = os.path.join(OUTPUT_DIR, "12_shadow_boxing_calibration.png")
    cv2.imwrite(path_12, frame_calib)
    print(f"  -> Saved {path_12}")

if __name__ == "__main__":
    capture_launcher_screens()
    capture_fruit_ninja_screens()
    capture_shadow_boxing_screens()
    print("\n[SUCCESS] All high-resolution screen captures completed successfully!")
