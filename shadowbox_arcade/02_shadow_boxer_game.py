import cv2
import time
import math
import random
import os
import sys
import json
import numpy as np
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sound_fx import play_sfx, trigger_haptic, init_haptic
from player_profiles import get_profile_manager
from clap_detector import ClapDetector
from music_manager import get_music_manager
from tracking import (
    CameraPipeline, PoseTracker, PunchDetector, CalibrationManager,
    TrackingDebugVisualizer, TrackingState, ArmTrackingResult, PoseTrackingResult, PunchState
)

# -----------------------------------------------------------------------------
# SHADOW-BOXER: MULTIMODAL KINETIC & ACOUSTIC AR BOXING GAME (ULTIMATE EDITION)
# Major Enhancements:
# 1. Full-Screen Native Resolution with dynamic responsive UI scaling
# 2. Optimized CPU Pipeline: Fast downsampled pose inference + sub-pixel EMA smoothing
# 3. Massive Arcade Boxing Gloves + Glowing Fist Reticles & Swept Continuous Hitboxes
# 4. Multimodal KIAI Battle Cry (Microphone DSP) & Sonic Clap Forcefield Deflection
# 5. Dynamic Mini-Boss Waves with Armor, Floating Health Bars & Sound Fanfares
# 6. Glowing Power-Up Orbs: 2x Damage, Slow-Mo, 3x Score, Rapid Fury, Titanium Shield
# 7. Pause Menu (P key), Fullscreen Toggle (F key), and Local Persistence Ready
# -----------------------------------------------------------------------------

# 1. Screen & Window Configuration (Native Fullscreen Detection)
try:
    import ctypes
    user32 = ctypes.windll.user32
    SCREEN_WIDTH = user32.GetSystemMetrics(0)
    SCREEN_HEIGHT = user32.GetSystemMetrics(1)
except Exception:
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720

# Dynamic layout scale factor (baseline is 720p)
SCALE = max(0.8, min(2.5, SCREEN_HEIGHT / 720.0))
WINDOW_NAME = "SHADOWBOX ARCADE: Shadow Boxing"
is_fullscreen = True

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 2. Initialize Tracking Engine 2.0 (Threaded Camera Acquisition + Biomechanical Pose Tracking)
CAM_W = 640
CAM_H = 480
camera_pipe = CameraPipeline(camera_index=0, target_w=CAM_W, target_h=CAM_H, target_fps=60)
camera_pipe.start()

pose_tracker = PoseTracker(cam_w=CAM_W, cam_h=CAM_H, screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT, mirror_x=True)
punch_detector = PunchDetector(min_punch_speed=200.0, min_extension=0.55)
calibration_mgr = CalibrationManager(screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT)
debug_visualizer = TrackingDebugVisualizer()

# -----------------------------------------------------------------------------
# GAME STATE & VARIABLES
# -----------------------------------------------------------------------------
GAME_STATE = "WAITING"   # "WAITING", "PLAYING", "PAUSED", "GAMEOVER"
GAME_DURATION = 90       # 90 seconds round
round_start_time = 0.0
pause_start_time = 0.0
total_paused_duration = 0.0

score = 0
combo = 0
max_combo = 0
total_punches = 0
dodges_completed = 0
kiai_strikes = 0
bosses_defeated = 0
powerups_collected = 0
last_hit_text = ""
last_hit_time = 0.0
screen_shake = 0

# -----------------------------------------------------------------------------
# KIAI ENERGY METER (Punch-Charged Shout Mode)
# -----------------------------------------------------------------------------
KIAI_MAX = 100
kiai_energy = 0
KIAI_ENERGY_PER_HIT = 8       # Energy gained per successful target hit
KIAI_ENERGY_BOSS_HIT = 15     # Extra energy per mini-boss hit
KIAI_MODE_DURATION = 10.0     # Kiai bonus mode lasts 10 seconds
SHOUT_THRESHOLD = 0.038       # Calibrated acoustic RMS threshold for Kiai battle cry
kiai_mode = False
kiai_mode_end = 0.0
kiai_mode_activated_time = 0.0  # For activation flash display

# -----------------------------------------------------------------------------
# PLAYER PROFILE & PERSISTENT RECORDS (player_profiles.py)
# -----------------------------------------------------------------------------
explicit_player = None
if "--player-id" in sys.argv:
    try:
        p_idx = sys.argv.index("--player-id")
        if p_idx + 1 < len(sys.argv):
            explicit_player = sys.argv[p_idx + 1]
    except Exception:
        pass

profile_mgr = get_profile_manager()
if explicit_player and explicit_player in profile_mgr.data.get("players", {}):
    active_player = profile_mgr.data["players"][explicit_player]
else:
    active_player = profile_mgr.get_active_player()
player_id = active_player["id"]
player_name = active_player.get("name", "Guest")
player_prefs = active_player.get("preferences", {})

high_score = active_player.get("shadow_boxing", {}).get("high_score", 0)
prev_high_score = high_score
is_new_high_score = False

def record_player_match():
    """Safely persist player stats to player_profiles.json."""
    global is_new_high_score, high_score
    is_new, best = profile_mgr.record_shadow_boxing_match(
        player_id=player_id,
        score=score,
        combo=max_combo,
        punches=total_punches,
        kiai=kiai_strikes,
        bosses=bosses_defeated,
        dodges=dodges_completed,
        shields=shields_deflected
    )
    if is_new:
        is_new_high_score = True
        high_score = best

# Background Music Coordinator
music_mgr = get_music_manager(player_prefs)
music_mgr.play_boxing_theme()

# -----------------------------------------------------------------------------
# MULTIMODAL AUDIO DSP & TRANSIENT CLAP DETECTOR
# -----------------------------------------------------------------------------
AUDIO_AVAILABLE = False
show_diagnostic = False
mic_volume = 0.0
KIAI_THRESHOLD = 0.038

clap_detector = ClapDetector(sample_rate=44100, block_size=512, cooldown_sec=1.5)
clap_detector.sensitivity = player_prefs.get("mic_sensitivity", 1.0)

def audio_callback(indata, frames, time_info, status):
    """Audio DSP callback for transient clap detection & shout tracking."""
    global mic_volume
    if status:
        pass
    clap_detector.process_audio_block(indata)
    mic_volume = clap_detector.vocal_rms

audio_stream = None

def init_audio_stream():
    """Robustly attempt to initialize the microphone input stream with multiple fallbacks."""
    global audio_stream, AUDIO_AVAILABLE
    attempts = [
        {"channels": 1, "samplerate": 44100, "blocksize": 512, "desc": "44.1 kHz Mono"},
        {"channels": 2, "samplerate": 44100, "blocksize": 512, "desc": "44.1 kHz Stereo Array"},
        {"channels": 1, "samplerate": 16000, "blocksize": 256, "desc": "16 kHz Mono"},
        {"channels": 2, "samplerate": 16000, "blocksize": 256, "desc": "16 kHz Stereo"}
    ]
    for cfg in attempts:
        try:
            stream = sd.InputStream(
                channels=cfg["channels"],
                samplerate=cfg["samplerate"],
                callback=audio_callback,
                blocksize=cfg["blocksize"]
            )
            stream.start()
            audio_stream = stream
            AUDIO_AVAILABLE = True
            clap_detector.mic_available = True
            print(f"[Audio DSP] Active microphone stream: {cfg['desc']} (sensitivity: {clap_detector.sensitivity}x)")
            return True
        except Exception:
            continue
    return False

if not init_audio_stream():
    print("[Audio DSP Warning] Standard microphone stream unavailable. Falling back to Vision.")
    audio_stream = None
    AUDIO_AVAILABLE = False
    clap_detector.mic_available = False

# Initialize optional Arduino haptic stub
init_haptic()

# -----------------------------------------------------------------------------
# TRACKING & SMOOTHING (EMA filter for zero-lag, jitter-free hands)
# -----------------------------------------------------------------------------
smooth_wrist_l = None
smooth_wrist_r = None
smooth_elbow_l = None
smooth_elbow_r = None
smooth_shoulder_l = None
smooth_shoulder_r = None
smooth_nose = None

prev_fist_l = None
prev_fist_r = None
last_pose_time = time.time()

# Radii scaled to native resolution (proportionate, natural glove sizing)
TARGET_RADIUS = int(48 * SCALE)
BASE_GLOVE_RADIUS = int(45 * SCALE)
GLOVE_RADIUS = BASE_GLOVE_RADIUS

# Laser Hazard
laser_active = False
laser_y = int(SCREEN_HEIGHT * 0.38)
laser_start_time = 0.0
laser_duration = 3.5
next_laser_time = time.time() + random.uniform(8, 14)
laser_dodged = False
laser_deflected = False

# Sonic Forcefield Shield (Clap Detection)
shield_active = False
shield_start_time = 0.0
shield_duration = 1.4
shields_deflected = 0
shield_center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

# Particle System
particles = []

# -----------------------------------------------------------------------------
# POWER-UP SYSTEM
# -----------------------------------------------------------------------------
POWERUP_TYPES = {
    "DOUBLE_DMG": {
        "name": "DOUBLE DAMAGE",
        "label": "2X DMG",
        "color": (0, 120, 255),       # Fiery Orange/Red
        "duration": 8.0,
        "icon": "2X"
    },
    "SLOW_MO": {
        "name": "SLOW MOTION",
        "label": "SLOW-MO",
        "color": (255, 240, 50),      # Ice Cyan
        "duration": 7.5,
        "icon": "SL"
    },
    "SCORE_BOOST": {
        "name": "3X SCORE MULTIPLIER",
        "label": "3X SCORE",
        "color": (0, 215, 255),       # Golden Sun
        "duration": 8.0,
        "icon": "3X"
    },
    "RAPID_FURY": {
        "name": "RAPID FURY GLOVES",
        "label": "FURY",
        "color": (240, 50, 220),      # Neon Violet
        "duration": 8.0,
        "icon": "FR"
    },
    "SHIELD": {
        "name": "TITANIUM SHIELD",
        "label": "SHIELD",
        "color": (60, 255, 80),       # Emerald Neon
        "duration": 8.0,
        "icon": "SH"
    }
}

active_powerup = None
powerup_end_time = 0.0
powerup_target = None
next_powerup_spawn_time = 0.0

# -----------------------------------------------------------------------------
# MINI-BOSS SYSTEM
# -----------------------------------------------------------------------------
boss_active = False
boss_data = None
# Waves scheduled at seconds remaining in round
BOSS_WAVE_TIMERS = [42, 20]
boss_waves_spawned = set()

def spawn_target():
    """Spawn a regular target matching appropriate reach zone."""
    hand_type = random.choice(["LEFT", "RIGHT"])
    margin_x = int(100 * SCALE)
    margin_y_top = int(140 * SCALE)
    margin_y_bottom = int(160 * SCALE)
    
    if hand_type == "LEFT":
        x = random.randint(margin_x, int(SCREEN_WIDTH * 0.44))
    else:
        x = random.randint(int(SCREEN_WIDTH * 0.56), SCREEN_WIDTH - margin_x)
        
    y = random.randint(margin_y_top, SCREEN_HEIGHT - margin_y_bottom)
    return {"x": x, "y": y, "radius": TARGET_RADIUS, "hand": hand_type, "spawn_time": time.time()}

def spawn_powerup_target():
    """Spawn a special glowing power-up orb."""
    ptype = random.choice(list(POWERUP_TYPES.keys()))
    margin_x = int(120 * SCALE)
    x = random.randint(margin_x, SCREEN_WIDTH - margin_x)
    y = random.randint(int(150 * SCALE), int(SCREEN_HEIGHT * 0.68))
    radius = int(42 * SCALE)
    return {
        "x": x,
        "y": y,
        "radius": radius,
        "type": ptype,
        "spawn_time": time.time(),
        "duration": 9.0  # Disappears if not hit in 9s
    }

def spawn_mini_boss(wave_index=1):
    """Spawn a multi-hit mechanical mini-boss encounter."""
    max_hp = 6 if wave_index == 1 else 10
    name = "MECHA-BRUTE" if wave_index == 1 else "CYBER-TITAN"
    radius = int(82 * SCALE)
    boss = {
        "x": SCREEN_WIDTH // 2,
        "y": int(SCREEN_HEIGHT * 0.40),
        "radius": radius,
        "hp": max_hp,
        "max_hp": max_hp,
        "name": name,
        "hit_flash_time": 0.0,
        "spawn_time": time.time(),
        "wave": wave_index,
        "base_x": SCREEN_WIDTH // 2,
        "base_y": int(SCREEN_HEIGHT * 0.40),
    }
    trigger_haptic("boss")
    return boss

target = spawn_target()

# -----------------------------------------------------------------------------
# GRAPHICS & VISUAL FX FUNCTIONS
# -----------------------------------------------------------------------------
def create_explosion(x, y, color, count=24, speed_max=14, is_kiai=False):
    """Create flying glowing particle sparks when strikes land."""
    if is_kiai:
        count = 42
        speed_max = 19
        
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(4, speed_max)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        size = int(random.randint(4, 9) * SCALE) if is_kiai else int(random.randint(3, 6) * SCALE)
        life = random.randint(14, 26) if is_kiai else random.randint(10, 18)
        
        if is_kiai:
            p_color = random.choice([(0, 215, 255), (0, 100, 255), (255, 255, 255), (0, 255, 255)])
        else:
            p_color = color
            
        particles.append({"x": x, "y": y, "vx": vx, "vy": vy, "size": size, "life": life, "max_life": life, "color": p_color})

def update_and_draw_particles(frame):
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1
        alpha = p["life"] / p["max_life"]
        radius = max(1, int(p["size"] * alpha))
        cv2.circle(frame, (int(p["x"]), int(p["y"])), radius, p["color"], -1)
        if p["life"] <= 0:
            particles.remove(p)

def dist_point_to_segment(p1, p2, c):
    """Calculate shortest distance from center circle point to line segment (p1->p2)."""
    if p1 is None and p2 is None:
        return 999999.0
    if p1 is None:
        return math.hypot(c[0] - p2[0], c[1] - p2[1])
    if p2 is None:
        return math.hypot(c[0] - p1[0], c[1] - p1[1])
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = c
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(cx - x1, cy - y1)
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(cx - proj_x, cy - proj_y)

# -----------------------------------------------------------------------------
# AUTHENTIC REFERENCE BOXING GLOVE SPRITES & FIRST-PERSON PROJECTION
# -----------------------------------------------------------------------------
SPRITE_DIR = os.path.dirname(__file__)
skin_name = player_prefs.get("glove_skin", "Classic Red")
skin_tag = skin_name.lower().replace(" ", "_")
if skin_tag == "neon_violet":
    skin_tag = "neon_purple"

SPRITE_PATH_L = os.path.join(SPRITE_DIR, f"boxing_glove_left_{skin_tag}.png")
SPRITE_PATH_R = os.path.join(SPRITE_DIR, f"boxing_glove_right_{skin_tag}.png")

if not os.path.exists(SPRITE_PATH_L) or not os.path.exists(SPRITE_PATH_R):
    import generate_gloves
    generate_gloves.generate_all_skins(SPRITE_DIR)

GLOVE_SPRITE_L = cv2.imread(SPRITE_PATH_L, cv2.IMREAD_UNCHANGED)
GLOVE_SPRITE_R = cv2.imread(SPRITE_PATH_R, cv2.IMREAD_UNCHANGED)
if GLOVE_SPRITE_L is None:
    GLOVE_SPRITE_L = cv2.imread(os.path.join(SPRITE_DIR, "boxing_glove_left.png"), cv2.IMREAD_UNCHANGED)
if GLOVE_SPRITE_R is None:
    GLOVE_SPRITE_R = cv2.imread(os.path.join(SPRITE_DIR, "boxing_glove_right.png"), cv2.IMREAD_UNCHANGED)

# Dynamic Glove Animation State (First-Person Perspective, Extension & Recoil)
glove_anim_l = {"extend": 0.0, "recoil": 0.0, "twist": 0.0, "scale_boost": 0.0}
glove_anim_r = {"extend": 0.0, "recoil": 0.0, "twist": 0.0, "scale_boost": 0.0}

def draw_boxing_glove(frame, wrist=None, elbow=None, is_left=True, speed=0.0, is_fury=False, anim=None, arm=None):
    """Render authentic 2D professional boxing gloves with customizable size (Large 1.30x default)
    and cosmetic skins, with first-person perspective and smooth 360-degree rotation.
    Supports Tracking Engine 2.0 ArmTrackingResult for zero-jitter, calibrated placement."""
    dropout = 1.0
    if arm is not None:
        if arm.dropout_alpha <= 0.04:
            return (0, 0)
        dropout = arm.dropout_alpha
        wx, wy = int(arm.wrist_filtered[0]), int(arm.wrist_filtered[1])
        rad = arm.forearm_angle
        speed = arm.speed_kmh
        is_left = (arm.side == "Left")
    else:
        if wrist is None:
            return (0, 0)
        wx, wy = wrist
        if elbow:
            ex, ey = elbow
            rad = math.atan2(wy - ey, wx - ex)
        else:
            rad = -math.pi / 2

    ext = anim["extend"] if anim else 0.0
    rec = anim["recoil"] if anim else 0.0
    twist = anim["twist"] if anim else 0.0
    scale_boost = anim["scale_boost"] if anim else 0.0

    # First-person forward projection: glove sits naturally over player's fist
    if arm is not None:
        ax = int(arm.glove_center[0] + (ext - rec) * math.cos(rad))
        ay = int(arm.glove_center[1] + (ext - rec) * math.sin(rad))
    else:
        proj_dist = (24.0 * SCALE) + ext - rec
        ax = int(wx + proj_dist * math.cos(rad))
        ay = int(wy + proj_dist * math.sin(rad))

    # Proportionate glove sizing: Normal (1.30x) or Large (1.45x)
    size_setting = player_prefs.get("glove_size", "Normal")
    if size_setting in ("Large", "Extra Large"):
        size_mult = 1.45
    elif size_setting in ("Normal", "Standard"):
        size_mult = 1.30
    else:
        # Fallback to Normal (1.30x) if previously set to Custom or unmapped
        size_mult = 1.30

    base_width = 120.0 * SCALE * size_mult
    g_scale = (base_width / 360.0) * (1.0 + scale_boost * 0.5)
    gw = max(20, int(360 * g_scale))
    gh = max(26, int(480 * g_scale))

    sprite_src = GLOVE_SPRITE_L if is_left else GLOVE_SPRITE_R
    resized = cv2.resize(sprite_src, (gw, gh), interpolation=cv2.INTER_LINEAR)

    # In Fury power-up mode: apply glowing violet tint to glove
    if is_fury:
        resized = resized.copy()
        resized[:, :, 0] = np.clip(resized[:, :, 0] * 1.6 + 45, 0, 255).astype(np.uint8)
        resized[:, :, 1] = np.clip(resized[:, :, 1] * 0.35, 0, 255).astype(np.uint8)

    # Cuff anchor inside sprite (where glove secures to forearm)
    cuff_x = int(gw * 0.48)
    cuff_y = int(gh * 0.86)

    # Calculate padded canvas to allow full 360-degree rotation without clipping
    d_max = int(math.hypot(max(cuff_x, gw - cuff_x), max(cuff_y, gh - cuff_y))) + 12
    pad_size = 2 * d_max
    padded = np.zeros((pad_size, pad_size, 4), dtype=np.uint8)

    x_in = d_max - cuff_x
    y_in = d_max - cuff_y
    padded[y_in:y_in+gh, x_in:x_in+gw] = resized

    # Rotate around cuff anchor so glove points forward along forearm punch vector
    forearm_deg = math.degrees(rad)
    rot_deg = -(forearm_deg + 90) + (twist * (-1.0 if is_left else 1.0))
    M = cv2.getRotationMatrix2D((d_max, d_max), rot_deg, 1.0)
    rotated = cv2.warpAffine(padded, M, (pad_size, pad_size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Speed trails when throwing fast jabs or hooks matching skin theme
    if speed > 15 or is_fury:
        if is_fury:
            trail_col = (255, 50, 200)
        elif skin_name == "Electric Blue":
            trail_col = (255, 180, 20)
        elif skin_name == "Neon Purple":
            trail_col = (220, 40, 200)
        elif skin_name == "Golden":
            trail_col = (0, 215, 255)
        else: # Classic Red
            trail_col = (0, 215, 255) if is_left else (0, 140, 255)

        for offset_px in (int(18 * SCALE), int(34 * SCALE)):
            tx = int(ax - offset_px * math.cos(rad))
            ty = int(ay - offset_px * math.sin(rad))
            cv2.ellipse(frame, (tx, ty), (int(46 * SCALE), int(32 * SCALE)), int(forearm_deg), 0, 360, trail_col, 2)

    # Fast Alpha blend rotated glove onto frame with dropout fading
    rx1 = ax - d_max
    ry1 = ay - d_max
    rx2 = rx1 + pad_size
    ry2 = ry1 + pad_size

    fx1, fy1 = max(0, rx1), max(0, ry1)
    fx2, fy2 = min(frame.shape[1], rx2), min(frame.shape[0], ry2)
    sx1, sy1 = fx1 - rx1, fy1 - ry1
    sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)

    if (fx2 > fx1) and (fy2 > fy1):
        alpha = (rotated[sy1:sy2, sx1:sx2, 3:4] / 255.0) * dropout
        color = rotated[sy1:sy2, sx1:sx2, :3]
        frame[fy1:fy2, fx1:fx2] = (1.0 - alpha) * frame[fy1:fy2, fx1:fx2] + alpha * color

    fist_x = ax
    fist_y = ay

    # 1. Detected hand tracking anchor ring on player's wrist with clear L / R label
    lbl = "L" if is_left else "R"
    cv2.circle(frame, (wx, wy), int(15 * SCALE), (255, 255, 255), 2)
    cv2.circle(frame, (wx, wy), int(8 * SCALE), (35, 35, 220) if is_left else (220, 35, 35), -1)
    cv2.putText(frame, lbl, (wx - int(5 * SCALE), wy + int(5 * SCALE)),
                cv2.FONT_HERSHEY_DUPLEX, 0.42 * SCALE, (255, 255, 255), 1)

    return (fist_x, fist_y)

def draw_sonic_shield(frame, cx, cy, progress):
    """Draw electric arcade forcefield hex-shield when player claps."""
    alpha = max(0.0, 1.0 - progress)
    shield_radius = int((110 + math.sin(time.time() * 16) * 10) * SCALE)

    shock_r = int(shield_radius + progress * 55 * SCALE)
    cv2.circle(frame, (cx, cy), shock_r, (255, 255, 200), max(1, int(3 * alpha)))
    cv2.circle(frame, (cx, cy), shield_radius, (255, 240, 0), int(4 * SCALE))
    cv2.circle(frame, (cx, cy), shield_radius - int(18 * SCALE), (255, 255, 255), int(2 * SCALE))
    cv2.circle(frame, (cx, cy), max(8, shield_radius - int(38 * SCALE)), (255, 200, 50), int(2 * SCALE))

    for angle_deg in (0, 60, 120, 180, 240, 300):
        rad = math.radians(angle_deg)
        x1 = int(cx + (shield_radius - int(38 * SCALE)) * math.cos(rad))
        y1 = int(cy + (shield_radius - int(38 * SCALE)) * math.sin(rad))
        x2 = int(cx + shield_radius * math.cos(rad))
        y2 = int(cy + shield_radius * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), int(2 * SCALE))

    cv2.putText(frame, "SONIC SHIELD ACTIVE", (cx - int(135 * SCALE), cy - shield_radius - int(16 * SCALE)),
                cv2.FONT_HERSHEY_DUPLEX, 0.7 * SCALE, (255, 255, 0), 2)

def draw_powerup_target(frame, pu, current_time):
    """Draw an animated, pulsing cyberpunk power-up crystal."""
    x = pu["x"]
    y = pu["y"]
    r = pu["radius"]
    cfg = POWERUP_TYPES.get(pu["type"], POWERUP_TYPES["DOUBLE_DMG"])
    p_color = cfg["color"]

    # Rotating energy ring
    rot_speed = current_time * 4.0
    pulse = int(math.sin(rot_speed * 1.5) * 5 * SCALE)
    outer_r = r + pulse

    cv2.circle(frame, (x, y), outer_r + int(6 * SCALE), p_color, 2)
    cv2.circle(frame, (x, y), outer_r, (255, 255, 255), 2)
    cv2.circle(frame, (x, y), max(6, outer_r - int(10 * SCALE)), p_color, -1)

    # Hex spokes
    for i in range(4):
        ang = rot_speed + i * (math.pi / 2)
        px = int(x + outer_r * math.cos(ang))
        py = int(y + outer_r * math.sin(ang))
        cv2.circle(frame, (px, py), int(4 * SCALE), (255, 255, 255), -1)

    # Power-up badge label
    label = cfg["icon"]
    cv2.putText(frame, label, (x - int(14 * SCALE), y + int(7 * SCALE)),
                cv2.FONT_HERSHEY_DUPLEX, 0.65 * SCALE, (10, 10, 10), 2)
    cv2.putText(frame, cfg["label"], (x - int(40 * SCALE), y + outer_r + int(18 * SCALE)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * SCALE, p_color, 1)

def draw_mini_boss(frame, boss, current_time):
    """Draw a towering mechanical mini-boss with rotating spiked shield and glowing health bar."""
    bx = boss["x"]
    by = boss["y"]
    r = boss["radius"]
    is_hit_flash = (current_time - boss["hit_flash_time"]) < 0.15

    # Flashing damaged or vibrant armored color
    core_color = (255, 255, 255) if is_hit_flash else (0, 0, 220)
    armor_color = (220, 220, 220) if is_hit_flash else (0, 165, 255)

    # 1. Spiked rotating energy perimeter
    num_spikes = 8
    rot = current_time * 2.5
    spike_pts = []
    for i in range(num_spikes * 2):
        ang = rot + i * (math.pi / num_spikes)
        dist = (r + int(16 * SCALE)) if (i % 2 == 0) else (r - int(6 * SCALE))
        sx = int(bx + dist * math.cos(ang))
        sy = int(by + dist * math.sin(ang))
        spike_pts.append((sx, sy))

    pts_arr = np.array(spike_pts, np.int32)
    cv2.polylines(frame, [pts_arr], True, armor_color, int(3 * SCALE))

    # 2. Heavy armored shell
    cv2.circle(frame, (bx, by), r, (25, 25, 25), -1)
    cv2.circle(frame, (bx, by), r, armor_color, int(4 * SCALE))
    cv2.circle(frame, (bx, by), r - int(12 * SCALE), (45, 45, 45), -1)

    # 3. Pulsing Cybernetic Core Eye
    core_pulse = int(math.sin(current_time * 8.0) * 6 * SCALE)
    cv2.circle(frame, (bx, by), max(10, int(32 * SCALE) + core_pulse), core_color, -1)
    cv2.circle(frame, (bx, by), max(5, int(14 * SCALE)), (255, 255, 255), -1)

    # 4. Boss Name Banner
    cv2.putText(frame, f"[ {boss['name']} ]", (bx - int(75 * SCALE), by - r - int(32 * SCALE)),
                cv2.FONT_HERSHEY_DUPLEX, 0.65 * SCALE, (0, 215, 255), 2)

    # 5. FLOATING SEGMENTED HEALTH BAR
    bar_w = int(180 * SCALE)
    bar_h = int(18 * SCALE)
    bar_x = bx - bar_w // 2
    bar_y = by - r - int(24 * SCALE)

    cv2.rectangle(frame, (bar_x - 2, bar_y - 2), (bar_x + bar_w + 2, bar_y + bar_h + 2), (20, 20, 20), -1)
    cv2.rectangle(frame, (bar_x - 2, bar_y - 2), (bar_x + bar_w + 2, bar_y + bar_h + 2), (200, 200, 200), 1)

    hp_pct = max(0.0, boss["hp"] / boss["max_hp"])
    fill_w = int(bar_w * hp_pct)

    # Color shifts from Green -> Yellow -> Red as HP drops
    if hp_pct > 0.5:
        bar_color = (0, 255, 0)
    elif hp_pct > 0.25:
        bar_color = (0, 215, 255)
    else:
        bar_color = (0, 0, 255)

    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    # Segment divider lines
    for seg in range(1, boss["max_hp"]):
        seg_x = bar_x + int(bar_w * (seg / boss["max_hp"]))
        cv2.line(frame, (seg_x, bar_y), (seg_x, bar_y + bar_h), (20, 20, 20), 1)

    # HP text
    hp_str = f"HP: {boss['hp']}/{boss['max_hp']}"
    cv2.putText(frame, hp_str, (bx - int(32 * SCALE), bar_y + bar_h - int(4 * SCALE)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * SCALE, (255, 255, 255), 1)

# -----------------------------------------------------------------------------
# MAIN GAME LOOP
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("  SHADOW-BOXER ULTIMATE EDITION RUNNING!")
print(f"  - Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT} (Full-Screen)")
print("  - SPACEBAR: Start Round  |  P: Pause/Resume  |  F: Toggle Fullscreen")
print("  - Hit Blue targets with LEFT hand, Red targets with RIGHT hand.")
print("  - Collect Glowing Orbs for Power-Ups (2x Damage, Slow-Mo, Shield)!")
print("  - Smash Mini-Bosses before the timer expires!")
print("  - SHOUT into your mic for 5x KIAI CRITICAL HITS!")
print("  - R: Restart  |  Q: Quit")
print("="*60 + "\n")

prev_frame_time = time.time()
current_pose = PoseTrackingResult()
frame_age_ms = 0.0

try:
    while True:
        success, raw_cam, frame_t, frame_age_ms, frame_id, is_new = camera_pipe.get_latest_frame()
        if not success or raw_cam is None:
            time.sleep(0.005)
            continue

        current_time = time.time()
        dt = current_time - last_pose_time if (current_time - last_pose_time) > 0 else 0.033
        last_pose_time = current_time

        # FPS Calculation
        fps = 1.0 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 30
        prev_frame_time = current_time

        # Process pose tracking on new camera frame via Tracking Engine 2.0
        if is_new or not current_pose.has_pose:
            current_pose = pose_tracker.process_frame(raw_cam, dt=dt, frame_id=frame_id)
            punch_events = punch_detector.update(current_pose.left_arm, current_pose.right_arm, now=current_time)
            debug_visualizer.log_sample(current_pose, punch_detector)

        # Mirror and crop/resize camera feed with guaranteed coordinate matching
        frame = pose_tracker.coord_mapper.prepare_camera_frame(raw_cam)

        # Dynamic Screen Shake
        if screen_shake > 0:
            dx = random.randint(-int(7 * SCALE), int(7 * SCALE))
            dy = random.randint(-int(7 * SCALE), int(7 * SCALE))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (SCREEN_WIDTH, SCREEN_HEIGHT))
            screen_shake -= 1

        # Convenience variables for game logic
        wrist_l = (int(current_pose.left_arm.wrist_filtered[0]), int(current_pose.left_arm.wrist_filtered[1])) if current_pose.left_arm.is_valid_for_attacks else None
        wrist_r = (int(current_pose.right_arm.wrist_filtered[0]), int(current_pose.right_arm.wrist_filtered[1])) if current_pose.right_arm.is_valid_for_attacks else None
        elbow_l = (int(current_pose.left_arm.elbow_filtered[0]), int(current_pose.left_arm.elbow_filtered[1])) if current_pose.left_arm.is_valid_for_attacks else None
        elbow_r = (int(current_pose.right_arm.elbow_filtered[0]), int(current_pose.right_arm.elbow_filtered[1])) if current_pose.right_arm.is_valid_for_attacks else None
        shoulder_l = (int(current_pose.left_arm.shoulder_filtered[0]), int(current_pose.left_arm.shoulder_filtered[1])) if current_pose.left_arm.is_valid_for_attacks else None
        shoulder_r = (int(current_pose.right_arm.shoulder_filtered[0]), int(current_pose.right_arm.shoulder_filtered[1])) if current_pose.right_arm.is_valid_for_attacks else None
        nose = (int(current_pose.nose[0]), int(current_pose.nose[1])) if current_pose.nose else None

        speed_l = current_pose.left_arm.speed_kmh
        speed_r = current_pose.right_arm.speed_kmh

        # Update Glove Punch Animation Physics (Extension, Perspective Zoom & Fist Pronation)
        for spd, anim in [(speed_l, glove_anim_l), (speed_r, glove_anim_r)]:
            if spd > 13:
                target_ext = min(42.0 * SCALE, (spd - 13) * 1.8 * SCALE)
                anim["extend"] = anim["extend"] * 0.65 + target_ext * 0.35
                anim["scale_boost"] = min(0.24, (spd - 13) * 0.012)
                anim["twist"] = min(16.0, (spd - 13) * 0.8)
            else:
                anim["extend"] = max(0.0, anim["extend"] - dt * 90.0 * SCALE)
                anim["scale_boost"] = max(0.0, anim["scale_boost"] - dt * 1.5)
                anim["twist"] = max(0.0, anim["twist"] - dt * 60.0)

            if anim["recoil"] > 0:
                anim["recoil"] = max(0.0, anim["recoil"] - dt * 240.0 * SCALE)

        is_fury = (active_powerup == "RAPID_FURY" and current_time < powerup_end_time)

        # Draw enlarged pro boxing gloves with zero-jitter ArmTrackingResult
        fist_l = draw_boxing_glove(frame, arm=current_pose.left_arm, is_fury=is_fury, anim=glove_anim_l)
        fist_r = draw_boxing_glove(frame, arm=current_pose.right_arm, is_fury=is_fury, anim=glove_anim_r)
    
        # -------------------------------------------------------------------------
        # MULTIMODAL CLAP DETECTION (Acoustic Transient + Biomechanical Pose Context)
        # -------------------------------------------------------------------------
        is_clap, clap_conf, _ = clap_detector.evaluate_clap(
            wrist_l=wrist_l,
            wrist_r=wrist_r,
            shoulder_l=shoulder_l,
            shoulder_r=shoulder_r,
            is_playing=(GAME_STATE == "PLAYING")
        )
    
        if is_clap:
            shield_active = True
            shield_start_time = current_time
            if wrist_l and wrist_r:
                shield_center = ((wrist_l[0] + wrist_r[0]) // 2, (wrist_l[1] + wrist_r[1]) // 2)
            else:
                shield_center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            create_explosion(shield_center[0], shield_center[1], (255, 255, 0), count=32, is_kiai=True)
            play_sfx("shield")
            trigger_haptic("shield")
            last_hit_text = "SONIC SHIELD ACTIVATED!"
            last_hit_time = current_time
    
        # -------------------------------------------------------------------------
        # POWER-UP DURATION & STATUS
        # -------------------------------------------------------------------------
        if active_powerup and current_time >= powerup_end_time:
            active_powerup = None
            GLOVE_RADIUS = BASE_GLOVE_RADIUS
    
        if active_powerup == "RAPID_FURY":
            GLOVE_RADIUS = int(BASE_GLOVE_RADIUS * 1.55)  # Massive hitboxes in fury mode!
        else:
            GLOVE_RADIUS = BASE_GLOVE_RADIUS
    
        # -------------------------------------------------------------------------
        # GAMEPLAY STATE MACHINE
        # -------------------------------------------------------------------------
        if GAME_STATE == "WAITING":
            # =====================================================================
            # FULL-SCREEN OPAQUE START SCREEN (Dark Arcade Theme)
            # =====================================================================
            frame[:] = (25, 25, 30)  # Solid dark charcoal background
    
            # Subtle red accent border
            cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1), (0, 0, 180), 3)
    
            # Top decorative bar
            bar_h = int(6 * SCALE)
            cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH, bar_h), (0, 0, 220), -1)
    
            # Title Section
            title_y = int(SCREEN_HEIGHT * 0.18)
            cv2.putText(frame, "SHADOW BOXING", (SCREEN_WIDTH // 2 - int(210 * SCALE), title_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5 * SCALE, (0, 0, 235), 3)
            cv2.putText(frame, "KINETIC AR BOXING", (SCREEN_WIDTH // 2 - int(130 * SCALE), title_y + int(42 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * SCALE, (200, 200, 200), 1)
    
            # Divider line
            div_y = title_y + int(65 * SCALE)
            cv2.line(frame, (SCREEN_WIDTH // 2 - int(200 * SCALE), div_y),
                     (SCREEN_WIDTH // 2 + int(200 * SCALE), div_y), (0, 0, 140), 2)
    
            # High Score Display
            hs_y = div_y + int(45 * SCALE)
            if high_score > 0:
                cv2.putText(frame, f"PERSONAL BEST: {high_score}", (SCREEN_WIDTH // 2 - int(115 * SCALE), hs_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7 * SCALE, (0, 215, 255), 2)
            else:
                cv2.putText(frame, "NO HIGH SCORE YET", (SCREEN_WIDTH // 2 - int(110 * SCALE), hs_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * SCALE, (140, 140, 140), 1)
    
            # Instructions
            instructions = [
                "Hit CYAN targets with LEFT glove, RED with RIGHT",
                "SHOUT into mic while punching for KIAI CRITICAL hits!",
                "Punches & battle cries charge KIAI OVERDRIVE (10s bonus)",
                "Collect POWER-UP ORBS for special abilities",
                "SQUAT or CLAP to deflect laser hazards",
                "Smash MINI-BOSSES for big bonus points",
            ]
    
            inst_y = hs_y + int(55 * SCALE)
            for line in instructions:
                cv2.putText(frame, line, (SCREEN_WIDTH // 2 - int(240 * SCALE), inst_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * SCALE, (180, 180, 180), 1)
                inst_y += int(32 * SCALE)
    
            # Controls bar at bottom
            ctrl_y = int(SCREEN_HEIGHT * 0.82)
            cv2.rectangle(frame, (0, ctrl_y - int(5 * SCALE)), (SCREEN_WIDTH, ctrl_y + int(40 * SCALE)), (35, 35, 40), -1)
            cv2.putText(frame, "[SPACE] Start  |  [C] Calibrate  |  [P] Pause  |  [F3] Telemetry  |  [Q] Quit",
                        (SCREEN_WIDTH // 2 - int(320 * SCALE), ctrl_y + int(22 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * SCALE, (160, 160, 160), 1)
    
            # Pulsing "PRESS SPACE TO FIGHT" button
            pulse_alpha = 0.6 + 0.4 * math.sin(current_time * 4)
            btn_color = (0, int(80 * pulse_alpha), int(220 * pulse_alpha))
            btn_w = int(380 * SCALE)
            btn_h = int(55 * SCALE)
            btn_x = SCREEN_WIDTH // 2 - btn_w // 2
            btn_y = int(SCREEN_HEIGHT * 0.72)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), btn_color, -1)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 0, 255), 2)
            cv2.putText(frame, "PRESS [SPACE] TO FIGHT!", (btn_x + int(30 * SCALE), btn_y + int(36 * SCALE)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8 * SCALE, (255, 255, 255), 2)

        elif GAME_STATE == "CALIBRATION":
            calibration_mgr.update(current_pose)
            calibration_mgr.draw_calibration_overlay(frame)
            if calibration_mgr.is_complete:
                GAME_STATE = "WAITING"
    
        elif GAME_STATE == "PAUSED":
            # =====================================================================
            # FULL-SCREEN OPAQUE PAUSE SCREEN
            # =====================================================================
            frame[:] = (25, 25, 30)  # Solid dark background
    
            # Red accent top/bottom bars
            cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH, int(6 * SCALE)), (0, 0, 220), -1)
            cv2.rectangle(frame, (0, SCREEN_HEIGHT - int(6 * SCALE)), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 220), -1)
    
            # Title
            pause_y = int(SCREEN_HEIGHT * 0.35)
            cv2.putText(frame, "GAME PAUSED", (SCREEN_WIDTH // 2 - int(180 * SCALE), pause_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4 * SCALE, (255, 255, 255), 3)
    
            # Accent line
            cv2.line(frame, (SCREEN_WIDTH // 2 - int(150 * SCALE), pause_y + int(20 * SCALE)),
                     (SCREEN_WIDTH // 2 + int(150 * SCALE), pause_y + int(20 * SCALE)), (0, 0, 200), 2)
    
            # Current progress info during pause
            info_y = pause_y + int(70 * SCALE)
            cv2.putText(frame, f"CURRENT SCORE: {score}", (SCREEN_WIDTH // 2 - int(120 * SCALE), info_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7 * SCALE, (0, 215, 255), 2)
            cv2.putText(frame, f"TIME REMAINING: {time_left}s", (SCREEN_WIDTH // 2 - int(120 * SCALE), info_y + int(35 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * SCALE, (200, 200, 200), 1)
    
            # Resume / Quit prompt
            opt_y = int(SCREEN_HEIGHT * 0.68)
            cv2.putText(frame, "Press 'P' to Resume Match", (SCREEN_WIDTH // 2 - int(160 * SCALE), opt_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75 * SCALE, (0, 255, 100), 2)
            cv2.putText(frame, "Press 'Q' to Quit", (SCREEN_WIDTH // 2 - int(90 * SCALE), opt_y + int(42 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * SCALE, (160, 160, 160), 1)
    
        elif GAME_STATE == "PLAYING":
            elapsed = current_time - round_start_time - total_paused_duration
            time_left = max(0, int(GAME_DURATION - elapsed))
    
            if time_left <= 0:
                GAME_STATE = "GAMEOVER"
                kiai_mode = False  # End kiai mode on round end
                # Record match stats to persistent profile and save high score
                prev_high_score = high_score
                record_player_match()
                save_high_score(high_score)
    
            # Kiai Mode Expiration Check
            if kiai_mode and current_time >= kiai_mode_end:
                kiai_mode = False
    
            # Check Active Sonic Shield
            is_shield_up = (current_time - shield_start_time) < shield_duration or (active_powerup == "SHIELD" and current_time < powerup_end_time)
            if is_shield_up and (current_time - shield_start_time) < shield_duration:
                prog = (current_time - shield_start_time) / shield_duration
                draw_sonic_shield(frame, shield_center[0], shield_center[1], prog)
    
            # ---------------------------------------------------------------------
            # MINI-BOSS WAVE SPAWN LOGIC
            # ---------------------------------------------------------------------
            for wave_idx, trigger_sec in enumerate(BOSS_WAVE_TIMERS, start=1):
                if time_left <= trigger_sec and wave_idx not in boss_waves_spawned and not boss_active:
                    boss_waves_spawned.add(wave_idx)
                    boss_active = True
                    boss_data = spawn_mini_boss(wave_idx)
                    screen_shake = 12
    
            # Update & Draw Mini-Boss if Active
            if boss_active and boss_data:
                # Smooth bobbing movement
                bob_speed = 2.2
                boss_data["x"] = int(boss_data["base_x"] + math.sin(current_time * bob_speed) * (140 * SCALE))
                boss_data["y"] = int(boss_data["base_y"] + math.cos(current_time * (bob_speed * 1.3)) * (45 * SCALE))
    
                draw_mini_boss(frame, boss_data, current_time)
    
                # Check Boss Collision with both fists
                bx, by, br = boss_data["x"], boss_data["y"], boss_data["radius"]
                boss_hit = False
                boss_speed = 0.0
    
                # Forgiving collision check with swept fists
                if fist_l and prev_fist_l and current_pose.left_arm.is_valid_for_attacks:
                    d_l = dist_point_to_segment(prev_fist_l, fist_l, (bx, by))
                    if d_l < (br + GLOVE_RADIUS) and (current_pose.left_arm.speed > 160.0 or punch_detector.left_sm.state in (PunchState.EXTENSION, PunchState.IMPACT)):
                        boss_hit = True
                        boss_speed = speed_l
                        punch_detector.left_sm.force_impact_event(current_pose.left_arm, now=current_time)
    
                if not boss_hit and fist_r and prev_fist_r and current_pose.right_arm.is_valid_for_attacks:
                    d_r = dist_point_to_segment(prev_fist_r, fist_r, (bx, by))
                    if d_r < (br + GLOVE_RADIUS) and (current_pose.right_arm.speed > 160.0 or punch_detector.right_sm.state in (PunchState.EXTENSION, PunchState.IMPACT)):
                        boss_hit = True
                        boss_speed = speed_r
                        punch_detector.right_sm.force_impact_event(current_pose.right_arm, now=current_time)
    
                if boss_hit and (current_time - boss_data["hit_flash_time"] > 0.22):
                    boss_data["hit_flash_time"] = current_time
                    # Multimodal Kiai: active if player shouted recently or during Kiai Mode
                    shout_active = clap_detector.has_recent_shout(window_sec=0.6) or (mic_volume > SHOUT_THRESHOLD)
                    is_kiai = shout_active or kiai_mode
    
                    # Apply impact recoil animation to hitting glove
                    if fist_l and dist_point_to_segment(prev_fist_l, fist_l, (bx, by)) < (br + GLOVE_RADIUS):
                        glove_anim_l["recoil"] = 22.0 * SCALE
                    else:
                        glove_anim_r["recoil"] = 22.0 * SCALE
    
                    # Damage Calculation
                    dmg = 1
                    if active_powerup == "DOUBLE_DMG" and current_time < powerup_end_time:
                        dmg *= 2
                    if is_kiai:
                        dmg *= 3
    
                    boss_data["hp"] = max(0, boss_data["hp"] - dmg)
                    total_punches += 1
                    combo += 1
                    if combo > max_combo:
                        max_combo = combo
    
                    # Balanced boss hit scoring
                    boss_pts = 5 if (is_kiai and shout_active) else (4 if is_kiai else (3 if boss_speed > 16 else 2))
                    if combo >= 3:
                        boss_pts += 1
                    if is_kiai:
                        kiai_strikes += 1
                    if active_powerup == "SCORE_BOOST" and current_time < powerup_end_time:
                        boss_pts += 2
                    score += boss_pts

                    # Charge kiai energy on boss hits (+ extra charge for battle cry)
                    if not kiai_mode:
                        hit_charge = KIAI_ENERGY_BOSS_HIT + (15 if shout_active else 0)
                        kiai_energy = min(KIAI_MAX, kiai_energy + hit_charge)
                        if kiai_energy >= KIAI_MAX:
                            kiai_mode = True
                            kiai_mode_end = current_time + KIAI_MODE_DURATION
                            kiai_mode_activated_time = current_time
                            kiai_energy = 0

                    if is_kiai:
                        screen_shake = 18 if shout_active else 12
                        create_explosion(bx, by, (0, 215, 255), count=42 if shout_active else 30, is_kiai=True)
                        last_hit_text = f"KIAI CRITICAL BOSS STRIKE! +{boss_pts} -{dmg} HP" if shout_active else f"KIAI BOSS STRIKE! +{boss_pts} -{dmg} HP"
                        play_sfx("critical")
                        trigger_haptic("kiai")
                        if shout_active:
                            clap_detector.consume_shout()
                    else:
                        play_sfx("boss_hit")
                        trigger_haptic("boss")
                        screen_shake = 7
                        create_explosion(bx, by, (0, 165, 255), count=18)
                        last_hit_text = f"BOSS HIT! +{boss_pts} -{dmg} HP"

                    last_hit_time = current_time
    
                    # Check Boss Defeat
                    if boss_data["hp"] <= 0:
                        boss_active = False
                        bosses_defeated += 1
                        screen_shake = 22
                        create_explosion(bx, by, (0, 215, 255), count=85, is_kiai=True)
                        play_sfx("boss_defeat")
                        boss_bonus = 10
                        score += boss_bonus
                        last_hit_text = f"MINI-BOSS DESTROYED! +{boss_bonus} PTS!"
                        last_hit_time = current_time
                        target = spawn_target()
    
            # ---------------------------------------------------------------------
            # POWER-UP ORB SPAWN & COLLECTION
            # ---------------------------------------------------------------------
            if not active_powerup and powerup_target is None and not boss_active:
                if current_time >= next_powerup_spawn_time:
                    powerup_target = spawn_powerup_target()
                    next_powerup_spawn_time = current_time + random.uniform(14, 20)
    
            if powerup_target:
                # Check expiration
                if current_time - powerup_target["spawn_time"] > powerup_target["duration"]:
                    powerup_target = None
                else:
                    draw_powerup_target(frame, powerup_target, current_time)
                    pux, puy, pur = powerup_target["x"], powerup_target["y"], powerup_target["radius"]
    
                    # Collision check with either hand
                    pu_hit = False
                    if fist_l and dist_point_to_segment(prev_fist_l, fist_l, (pux, puy)) < (pur + GLOVE_RADIUS):
                        pu_hit = True
                    if not pu_hit and fist_r and dist_point_to_segment(prev_fist_r, fist_r, (pux, puy)) < (pur + GLOVE_RADIUS):
                        pu_hit = True
    
                    if pu_hit:
                        # Apply recoil animation to hitting glove
                        if fist_l and dist_point_to_segment(prev_fist_l, fist_l, (pux, puy)) < (pur + GLOVE_RADIUS):
                            glove_anim_l["recoil"] = 16.0 * SCALE
                        else:
                            glove_anim_r["recoil"] = 16.0 * SCALE
    
                        active_powerup = powerup_target["type"]
                        cfg = POWERUP_TYPES[active_powerup]
                        powerup_end_time = current_time + cfg["duration"]
                        powerups_collected += 1
                        score += 5  # Special bonus for grabbing power-up
                        create_explosion(pux, puy, cfg["color"], count=32, is_kiai=True)
                        play_sfx("powerup")
                        trigger_haptic("powerup")
                        last_hit_text = f"POWER-UP: {cfg['label']}! +5"
                        last_hit_time = current_time
                        powerup_target = None
    
            # ---------------------------------------------------------------------
            # LASER HAZARD (Squat / Duck or Parrying Shield)
            # ---------------------------------------------------------------------
            if not laser_active and current_time >= next_laser_time and not boss_active:
                laser_active = True
                laser_start_time = current_time
                laser_dodged = False
                laser_deflected = False
    
            if laser_active:
                laser_elapsed = current_time - laser_start_time
                laser_speed_mod = 0.6 if (active_powerup == "SLOW_MO" and current_time < powerup_end_time) else 1.0
                eff_duration = laser_duration / laser_speed_mod
    
                laser_y = int(SCREEN_HEIGHT * 0.38)
                laser_color = (0, 0, 255) if int(current_time * 8) % 2 == 0 else (0, 255, 255)
                cv2.line(frame, (0, laser_y), (SCREEN_WIDTH, laser_y), laser_color, int(4 * SCALE))
                cv2.putText(frame, "! DUCK OR CLAP SHIELD !", (SCREEN_WIDTH // 2 - int(190 * SCALE), laser_y - int(16 * SCALE)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75 * SCALE, laser_color, 2)
    
                if nose and nose[1] > laser_y:
                    laser_dodged = True
                    cv2.putText(frame, "NICE DODGE!", (SCREEN_WIDTH // 2 - int(75 * SCALE), laser_y + int(40 * SCALE)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75 * SCALE, (0, 255, 0), 2)
    
                if is_shield_up:
                    laser_deflected = True
                    laser_dodged = True
                    cv2.putText(frame, "SHIELD DEFLECTING LASER!", (SCREEN_WIDTH // 2 - int(180 * SCALE), laser_y + int(40 * SCALE)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.72 * SCALE, (255, 255, 0), 2)
    
                if laser_elapsed >= eff_duration:
                    laser_active = False
                    next_laser_time = current_time + random.uniform(10, 16)
                    if laser_deflected:
                        score += 75
                        shields_deflected += 1
                        last_hit_text = "SHIELD PARRY! +75 PTS!"
                        last_hit_time = current_time
                        screen_shake = 10
                        create_explosion(SCREEN_WIDTH // 2, laser_y, (255, 255, 0), count=42, is_kiai=True)
                        play_sfx("parry")
                    elif laser_dodged:
                        score += 5
                        dodges_completed += 1
                        last_hit_text = "PERFECT DODGE! +5"
                        play_sfx("dodge")
                        last_hit_time = current_time
                    else:
                        score = max(0, score - 3)
                        combo = 0
                        last_hit_text = "LASER HIT! -3"
                        play_sfx("block")
                        last_hit_time = current_time
    
            # ---------------------------------------------------------------------
            # REGULAR TARGET DRAW & SWEPT COLLISION CHECK
            # ---------------------------------------------------------------------
            if target and not boss_active:
                pulse = int(math.sin(current_time * 6) * 4 * SCALE)
                r = target["radius"] + pulse
                t_color = (255, 200, 0) if target["hand"] == "LEFT" else (0, 80, 255)
                label = "L" if target["hand"] == "LEFT" else "R"
    
                # Draw Arcade Target Pad
                cv2.circle(frame, (target["x"], target["y"]), r, t_color, int(4 * SCALE))
                cv2.circle(frame, (target["x"], target["y"]), max(4, r - int(14 * SCALE)), t_color, -1)
                cv2.putText(frame, label, (target["x"] - int(10 * SCALE), target["y"] + int(10 * SCALE)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85 * SCALE, (0, 0, 0), 2)
    
                # Continuous Swept Collision Detection (Prevents tunneling through targets!)
                hit = False
                hit_speed = 0.0
                hit_hand = None
                tgt_center = (target["x"], target["y"])
    
                # Left Hand Check
                if target["hand"] == "LEFT" and fist_l and current_pose.left_arm.is_valid_for_attacks:
                    d_swept = dist_point_to_segment(prev_fist_l, fist_l, tgt_center)
                    if d_swept < (r + GLOVE_RADIUS + 15) and (current_pose.left_arm.speed > 160.0 or punch_detector.left_sm.state in (PunchState.EXTENSION, PunchState.IMPACT)):
                        hit = True
                        hit_speed = speed_l
                        hit_hand = "LEFT"
                        punch_detector.left_sm.force_impact_event(current_pose.left_arm, now=current_time)
    
                # Right Hand Check
                elif target["hand"] == "RIGHT" and fist_r and current_pose.right_arm.is_valid_for_attacks:
                    d_swept = dist_point_to_segment(prev_fist_r, fist_r, tgt_center)
                    if d_swept < (r + GLOVE_RADIUS + 15) and (current_pose.right_arm.speed > 160.0 or punch_detector.right_sm.state in (PunchState.EXTENSION, PunchState.IMPACT)):
                        hit = True
                        hit_speed = speed_r
                        hit_hand = "RIGHT"
                        punch_detector.right_sm.force_impact_event(current_pose.right_arm, now=current_time)
    
                # Forgiving Cross/Fury Hand Check: In Rapid Fury mode, either hand hits!
                if not hit and active_powerup == "RAPID_FURY":
                    for h_name, f_pt, p_pt, spd, arm_obj, sm_obj in [
                        ("LEFT", fist_l, prev_fist_l, speed_l, current_pose.left_arm, punch_detector.left_sm),
                        ("RIGHT", fist_r, prev_fist_r, speed_r, current_pose.right_arm, punch_detector.right_sm)
                    ]:
                        if f_pt and arm_obj.is_valid_for_attacks and dist_point_to_segment(p_pt, f_pt, tgt_center) < (r + GLOVE_RADIUS + 25):
                            if arm_obj.speed > 140.0 or sm_obj.state in (PunchState.EXTENSION, PunchState.IMPACT):
                                hit = True
                                hit_speed = spd
                                hit_hand = h_name
                                sm_obj.force_impact_event(arm_obj, now=current_time)
                                break
    
                if hit:
                    # Apply impact recoil animation to hitting glove
                    if hit_hand == "LEFT":
                        glove_anim_l["recoil"] = 20.0 * SCALE
                    else:
                        glove_anim_r["recoil"] = 20.0 * SCALE
    
                    # Multimodal Kiai shout detection: recent shout or active Kiai Mode
                    shout_active = clap_detector.has_recent_shout(window_sec=0.6) or (mic_volume > SHOUT_THRESHOLD)
                    is_kiai = shout_active or kiai_mode

                    if is_kiai:
                        kiai_strikes += 1
                        screen_shake = 12 if shout_active else 8
                        create_explosion(target["x"], target["y"], t_color, count=36 if shout_active else 26, is_kiai=True)
                        play_sfx("critical")
                        trigger_haptic("kiai")
                        if shout_active:
                            clap_detector.consume_shout()
                    else:
                        create_explosion(target["x"], target["y"], t_color, count=16)
                        play_sfx("punch")
                        trigger_haptic("punch")

                    total_punches += 1
                    combo += 1
                    if combo > max_combo:
                        max_combo = combo

                    # ---- BALANCED SCORING SYSTEM ----
                    # Base points: 2 normal, 3 for fast/accurate punch
                    base_pts = 3 if hit_speed > 16 else 2

                    # Consecutive punch bonus: +1 if combo >= 3
                    combo_bonus = 1 if combo >= 3 else 0
                    base_pts += combo_bonus

                    # Kiai shout bonus: 5 pts for acoustic shout, 4 for Kiai Mode (+ combo bonus)
                    if is_kiai:
                        base_pts = (5 if shout_active else 4) + combo_bonus

                    # Score Boost power-up: +2 points bonus
                    if active_powerup == "SCORE_BOOST" and current_time < powerup_end_time:
                        base_pts += 2

                    pts = base_pts
                    score += pts

                    # Charge kiai energy on every hit (+ extra charge when shouting!)
                    if not kiai_mode:
                        hit_charge = KIAI_ENERGY_PER_HIT + (12 if shout_active else 0)
                        kiai_energy = min(KIAI_MAX, kiai_energy + hit_charge)
                        if kiai_energy >= KIAI_MAX:
                            kiai_mode = True
                            kiai_mode_end = current_time + KIAI_MODE_DURATION
                            kiai_mode_activated_time = current_time
                            kiai_energy = 0

                    tag = "KIAI CRITICAL HIT!" if (is_kiai and shout_active) else ("KIAI MODE HIT!" if is_kiai else "CLEAN HIT!")
                    last_hit_text = f"{tag} +{pts}"
                    last_hit_time = current_time

                    target = spawn_target()
    
            # Update flying particle system
            update_and_draw_particles(frame)
    
            # ---------------------------------------------------------------------
            # HEADS UP DISPLAY (HUD) - Full-Screen Arcade Layout
            # ---------------------------------------------------------------------
            # ---- TOP BAR: Score, Timer, Combo ----
            header_h = int(58 * SCALE)
            cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH, header_h), (20, 20, 25), -1)
            # Red accent line under header
            cv2.line(frame, (0, header_h), (SCREEN_WIDTH, header_h), (0, 0, 200), 2)
    
            # Score (left)
            cv2.putText(frame, f"SCORE", (int(20 * SCALE), int(20 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * SCALE, (160, 160, 160), 1)
            cv2.putText(frame, f"{score}", (int(20 * SCALE), int(46 * SCALE)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9 * SCALE, (255, 255, 255), 2)
    
            # Timer (center) - MM:SS format
            timer_mins = time_left // 60
            timer_secs = time_left % 60
            timer_str = f"{timer_mins:01d}:{timer_secs:02d}"
            timer_color = (255, 255, 255) if time_left > 15 else (0, 0, 255)
            cv2.putText(frame, timer_str, (SCREEN_WIDTH // 2 - int(45 * SCALE), int(44 * SCALE)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1 * SCALE, timer_color, 2)
            cv2.putText(frame, "TIME", (SCREEN_WIDTH // 2 - int(22 * SCALE), int(16 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * SCALE, (160, 160, 160), 1)
    
            # Combo (right)
            combo_c = (0, 255, 0) if combo < 5 else ((0, 165, 255) if combo < 12 else (0, 0, 255))
            cv2.putText(frame, "COMBO", (SCREEN_WIDTH - int(130 * SCALE), int(20 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * SCALE, (160, 160, 160), 1)
            cv2.putText(frame, f"x{combo}", (SCREEN_WIDTH - int(120 * SCALE), int(46 * SCALE)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9 * SCALE, combo_c, 2)
    
            # High Score (right of score)
            hs_x = int(200 * SCALE)
            cv2.putText(frame, f"BEST: {high_score}", (hs_x, int(44 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * SCALE, (0, 200, 220), 1)
    
            # ---- ACTIVE POWER-UP HUD BADGE ----
            if active_powerup and current_time < powerup_end_time:
                cfg = POWERUP_TYPES[active_powerup]
                rem = powerup_end_time - current_time
                pct = rem / cfg["duration"]
                badge_w = int(220 * SCALE)
                badge_h = int(32 * SCALE)
                bx = SCREEN_WIDTH // 2 - badge_w // 2
                by = header_h + int(10 * SCALE)
    
                cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), (25, 25, 25), -1)
                cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), cfg["color"], 2)
                cv2.rectangle(frame, (bx + 2, by + 2), (bx + int(badge_w * pct), by + badge_h - 2), cfg["color"], -1)
                cv2.putText(frame, f"{cfg['name']} ({rem:.1f}s)", (bx + int(10 * SCALE), by + int(21 * SCALE)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48 * SCALE, (255, 255, 255), 1)
    
            # ---- KIAI ENERGY METER (Bottom Left) ----
            kiai_bar_x = int(15 * SCALE)
            kiai_bar_y = SCREEN_HEIGHT - int(80 * SCALE)
            kiai_bar_w = int(220 * SCALE)
            kiai_bar_h = int(22 * SCALE)

            # Background panel
            cv2.rectangle(frame, (kiai_bar_x - 4, kiai_bar_y - int(24 * SCALE)),
                          (kiai_bar_x + kiai_bar_w + 4, kiai_bar_y + kiai_bar_h + int(14 * SCALE)),
                          (20, 20, 25), -1)

            if kiai_mode:
                # Kiai Mode ACTIVE: Show countdown
                kiai_remaining = max(0, kiai_mode_end - current_time)
                kiai_pct = kiai_remaining / KIAI_MODE_DURATION

                # Flashing "KIAI MODE ACTIVE" label
                flash = int(current_time * 6) % 2
                label_color = (0, 215, 255) if flash else (0, 100, 255)
                cv2.putText(frame, f"KIAI MODE! {kiai_remaining:.1f}s", (kiai_bar_x, kiai_bar_y - int(6 * SCALE)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55 * SCALE, label_color, 2)

                # Countdown bar (draining)
                cv2.rectangle(frame, (kiai_bar_x, kiai_bar_y), (kiai_bar_x + kiai_bar_w, kiai_bar_y + kiai_bar_h), (40, 40, 50), -1)
                fill_w = int(kiai_bar_w * kiai_pct)
                if fill_w > 0:
                    cv2.rectangle(frame, (kiai_bar_x, kiai_bar_y), (kiai_bar_x + fill_w, kiai_bar_y + kiai_bar_h), (0, 215, 255), -1)
                cv2.rectangle(frame, (kiai_bar_x, kiai_bar_y), (kiai_bar_x + kiai_bar_w, kiai_bar_y + kiai_bar_h), (0, 180, 255), 2)
            else:
                # Energy charging mode
                energy_pct = kiai_energy / KIAI_MAX
                cv2.putText(frame, "KIAI ENERGY", (kiai_bar_x, kiai_bar_y - int(6 * SCALE)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * SCALE, (180, 180, 180), 1)

                # Bar background
                cv2.rectangle(frame, (kiai_bar_x, kiai_bar_y), (kiai_bar_x + kiai_bar_w, kiai_bar_y + kiai_bar_h), (40, 40, 50), -1)

                # Filled portion with color gradient (gray -> yellow -> red as it fills)
                fill_w = int(kiai_bar_w * energy_pct)
                if energy_pct < 0.5:
                    bar_color = (0, 180, 255)  # Orange
                elif energy_pct < 0.85:
                    bar_color = (0, 100, 255)  # Red-orange
                else:
                    bar_color = (0, 0, 255)    # Bright red (almost ready!)
                if fill_w > 0:
                    cv2.rectangle(frame, (kiai_bar_x, kiai_bar_y), (kiai_bar_x + fill_w, kiai_bar_y + kiai_bar_h), bar_color, -1)
                cv2.rectangle(frame, (kiai_bar_x, kiai_bar_y), (kiai_bar_x + kiai_bar_w, kiai_bar_y + kiai_bar_h), (100, 100, 110), 2)

                # Percentage text
                cv2.putText(frame, f"{int(energy_pct * 100)}%", (kiai_bar_x + kiai_bar_w + int(8 * SCALE), kiai_bar_y + int(16 * SCALE)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * SCALE, (200, 200, 200), 1)

            # Live Acoustic / Shout Status Meter (Visible in both modes)
            mic_indicator_h = int(6 * SCALE)
            eff_thresh = getattr(clap_detector, "vocal_threshold", SHOUT_THRESHOLD) / max(0.2, getattr(clap_detector, "sensitivity", 1.0))
            is_shout_active = clap_detector.has_recent_shout(window_sec=0.5) or (mic_volume > eff_thresh)
            mic_fill = int(min(1.0, mic_volume / (eff_thresh * 1.5)) * kiai_bar_w)
            mic_color = (0, 255, 100) if is_shout_active else ((0, 180, 255) if mic_volume > (eff_thresh * 0.4) else (70, 75, 85))

            cv2.rectangle(frame, (kiai_bar_x, kiai_bar_y + kiai_bar_h + 3),
                          (kiai_bar_x + kiai_bar_w, kiai_bar_y + kiai_bar_h + mic_indicator_h + 3), (35, 38, 48), -1)
            if mic_fill > 0:
                cv2.rectangle(frame, (kiai_bar_x, kiai_bar_y + kiai_bar_h + 3),
                              (kiai_bar_x + mic_fill, kiai_bar_y + kiai_bar_h + mic_indicator_h + 3), mic_color, -1)

            if AUDIO_AVAILABLE:
                if is_shout_active:
                    status_lbl = "KIAI READY!"
                    lbl_c = (0, 215, 255)
                else:
                    status_lbl = "MIC ACTIVE [SHOUT]"
                    lbl_c = (150, 160, 175)
            else:
                status_lbl = "MIC: OFF (VISION)"
                lbl_c = (120, 120, 140)

            cv2.putText(frame, status_lbl, (kiai_bar_x + kiai_bar_w + int(8 * SCALE), kiai_bar_y + kiai_bar_h + int(8 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36 * SCALE, lbl_c, 1)

            # ---- KIAI MODE ACTIVATION FLASH ----
            if kiai_mode and (current_time - kiai_mode_activated_time) < 1.5:
                flash_alpha = max(0, 1.0 - (current_time - kiai_mode_activated_time) / 1.5)
                overlay_flash = frame.copy()
                cv2.putText(overlay_flash, "KIAI MODE ACTIVATED!", (SCREEN_WIDTH // 2 - int(210 * SCALE), SCREEN_HEIGHT // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2 * SCALE, (0, 215, 255), 3)
                cv2.addWeighted(overlay_flash, flash_alpha, frame, 1.0 - flash_alpha, 0, frame)
    
            # ---- Floating Hit Impact Toast ----
            if current_time - last_hit_time < 1.2:
                t_col = (0, 215, 255) if "KIAI" in last_hit_text or "BOSS" in last_hit_text else (0, 255, 255)
                cv2.putText(frame, last_hit_text, (SCREEN_WIDTH // 2 - int(120 * SCALE), header_h + int(60 * SCALE)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75 * SCALE, t_col, 2)
    
        elif GAME_STATE == "GAMEOVER":
            # =====================================================================
            # FULL-SCREEN OPAQUE RESULTS SCREEN
            # =====================================================================
            frame[:] = (25, 25, 30)  # Solid dark background
    
            # Top accent bar
            cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH, int(6 * SCALE)), (0, 0, 220), -1)
    
            # "ROUND COMPLETE" header
            header_y = int(SCREEN_HEIGHT * 0.10)
            cv2.putText(frame, "ROUND COMPLETE", (SCREEN_WIDTH // 2 - int(175 * SCALE), header_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2 * SCALE, (255, 255, 255), 2)
    
            # Divider
            div_y = header_y + int(25 * SCALE)
            cv2.line(frame, (SCREEN_WIDTH // 2 - int(180 * SCALE), div_y),
                     (SCREEN_WIDTH // 2 + int(180 * SCALE), div_y), (0, 0, 140), 2)
    
            # ---- FINAL SCORE (prominent but balanced) ----
            score_y = div_y + int(65 * SCALE)
            cv2.putText(frame, "FINAL SCORE", (SCREEN_WIDTH // 2 - int(85 * SCALE), score_y - int(18 * SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * SCALE, (160, 160, 160), 1)
            cv2.putText(frame, f"{score}", (SCREEN_WIDTH // 2 - int(55 * SCALE), score_y + int(25 * SCALE)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4 * SCALE, (255, 255, 255), 3)
    
            # ---- NEW HIGH SCORE celebration or Personal Best display ----
            hs_y = score_y + int(65 * SCALE)
            if is_new_high_score:
                # Pulsing celebration
                flash = int(current_time * 5) % 2
                hs_color = (0, 0, 255) if flash else (0, 215, 255)
                cv2.putText(frame, "NEW HIGH SCORE!", (SCREEN_WIDTH // 2 - int(145 * SCALE), hs_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9 * SCALE, hs_color, 2)
                # Decorative stars
                for offset in [-int(170 * SCALE), int(155 * SCALE)]:
                    star_x = SCREEN_WIDTH // 2 + offset
                    cv2.circle(frame, (star_x, hs_y - int(8 * SCALE)), int(6 * SCALE), (0, 215, 255), -1)
                if prev_high_score > 0:
                    cv2.putText(frame, f"PREVIOUS BEST: {prev_high_score}", (SCREEN_WIDTH // 2 - int(105 * SCALE), hs_y + int(26 * SCALE)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * SCALE, (160, 160, 160), 1)
            else:
                cv2.putText(frame, f"PERSONAL BEST: {high_score}", (SCREEN_WIDTH // 2 - int(120 * SCALE), hs_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65 * SCALE, (0, 200, 220), 2)
    
            # ---- Stats Section ----
            stats_start_y = hs_y + int(55 * SCALE)
            left_col_x = SCREEN_WIDTH // 2 - int(220 * SCALE)
            right_col_x = SCREEN_WIDTH // 2 + int(30 * SCALE)
            row_h = int(34 * SCALE)
    
            # Left column stats
            left_stats = [
                ("PUNCHES LANDED", f"{total_punches}", (230, 230, 230)),
                ("MINI-BOSSES DEFEATED", f"{bosses_defeated}", (0, 165, 255)),
                ("KIAI SHOUT STRIKES", f"{kiai_strikes}", (0, 215, 255)),
            ]
    
            # Right column stats
            right_stats = [
                ("POWER-UPS COLLECTED", f"{powerups_collected}", (230, 230, 230)),
                ("LASERS DEFLECTED", f"{shields_deflected}", (0, 255, 255)),
                ("MAX COMBO STREAK", f"x{max_combo}", (0, 255, 0)),
            ]
    
            for i, (label, value, color) in enumerate(left_stats):
                y = stats_start_y + i * row_h
                cv2.putText(frame, label, (left_col_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * SCALE, (140, 140, 140), 1)
                cv2.putText(frame, value, (left_col_x + int(200 * SCALE), y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55 * SCALE, color, 1)
    
            for i, (label, value, color) in enumerate(right_stats):
                y = stats_start_y + i * row_h
                cv2.putText(frame, label, (right_col_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * SCALE, (140, 140, 140), 1)
                cv2.putText(frame, value, (right_col_x + int(200 * SCALE), y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55 * SCALE, color, 1)
    
            # ---- Bottom Buttons ----
            btn_y = int(SCREEN_HEIGHT * 0.84)
            # Play Again button
            btn_w = int(260 * SCALE)
            btn_h = int(50 * SCALE)
            btn_x1 = SCREEN_WIDTH // 2 - btn_w - int(20 * SCALE)
            cv2.rectangle(frame, (btn_x1, btn_y), (btn_x1 + btn_w, btn_y + btn_h), (0, 0, 160), -1)
            cv2.rectangle(frame, (btn_x1, btn_y), (btn_x1 + btn_w, btn_y + btn_h), (0, 0, 220), 2)
            cv2.putText(frame, "[R] PLAY AGAIN", (btn_x1 + int(40 * SCALE), btn_y + int(33 * SCALE)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65 * SCALE, (255, 255, 255), 2)
    
            # Quit button
            btn_x2 = SCREEN_WIDTH // 2 + int(20 * SCALE)
            cv2.rectangle(frame, (btn_x2, btn_y), (btn_x2 + btn_w, btn_y + btn_h), (40, 40, 45), -1)
            cv2.rectangle(frame, (btn_x2, btn_y), (btn_x2 + btn_w, btn_y + btn_h), (100, 100, 110), 2)
            cv2.putText(frame, "[Q] QUIT", (btn_x2 + int(65 * SCALE), btn_y + int(33 * SCALE)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65 * SCALE, (200, 200, 200), 2)
    
            # Bottom accent bar
            cv2.rectangle(frame, (0, SCREEN_HEIGHT - int(6 * SCALE)), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 220), -1)
    
        # Store previous fist positions for swept continuous collision detection
        prev_fist_l = fist_l
        prev_fist_r = fist_r
    
        # Acoustic Clap Diagnostic HUD (toggle with TAB)
        if show_diagnostic:
            clap_detector.draw_diagnostic_hud(frame, x=int(20 * SCALE), y=int(140 * SCALE), scale=SCALE)
    
        # Real-time tracking status badge
        st_text, st_col = calibration_mgr.get_tracking_status(current_pose)
        cv2.putText(frame, f"TRACK: {st_text}", (SCREEN_WIDTH - int(220 * SCALE), int(35 * SCALE)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48 * SCALE, st_col, 1)
    
        # F3 / Telemetry overlay
        debug_visualizer.update_render_fps()
        if debug_visualizer.enabled:
            debug_visualizer.render_overlay(
                frame, current_pose, punch_detector,
                frame_age_ms=frame_age_ms,
                cam_telemetry=camera_pipe.get_telemetry()
            )
    
        # Footer: FPS & System Status
        cv2.putText(frame, f"FPS: {int(fps)} | RES: {SCREEN_WIDTH}x{SCREEN_HEIGHT} | [C] Calibrate | [D] Telemetry | TAB: Audio | S: Shield | R: Restart | Q: Quit",
                    (int(15 * SCALE), SCREEN_HEIGHT - int(12 * SCALE)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * SCALE, (120, 255, 120), 1)
    
        cv2.imshow(WINDOW_NAME, frame)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('d'), ord('D'), 192, 0x72, 114):  # D or F3 key toggles telemetry HUD
            debug_visualizer.toggle()
        elif key in (ord('c'), ord('C')):  # C key triggers calibration
            if GAME_STATE != "CALIBRATION":
                calibration_mgr.start()
                GAME_STATE = "CALIBRATION"
        elif key == 9:  # TAB key toggles acoustic diagnostic HUD
            show_diagnostic = not show_diagnostic
        elif key == ord('f'):  # Toggle Fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('p'):  # Pause / Resume Toggle
            if GAME_STATE == "PLAYING":
                GAME_STATE = "PAUSED"
                pause_start_time = time.time()
            elif GAME_STATE == "PAUSED":
                GAME_STATE = "PLAYING"
                play_sfx("bell")
                total_paused_duration += (time.time() - pause_start_time)
        elif key == 32:  # SPACEBAR to start
            if GAME_STATE == "WAITING":
                GAME_STATE = "PLAYING"
                play_sfx("bell")
                round_start_time = time.time()
                total_paused_duration = 0.0
                score = 0
                combo = 0
                max_combo = 0
                total_punches = 0
                kiai_strikes = 0
                bosses_defeated = 0
                powerups_collected = 0
                shields_deflected = 0
                laser_deflected = False
                shield_start_time = 0.0
                dodges_completed = 0
                boss_active = False
                boss_data = None
                boss_waves_spawned.clear()
                active_powerup = None
                powerup_target = None
                kiai_energy = 0
                kiai_mode = False
                kiai_mode_end = 0.0
                is_new_high_score = False
                clap_detector.reset()
                target = spawn_target()
        elif key == ord('r'):  # Restart
            GAME_STATE = "PLAYING"
            play_sfx("bell")
            round_start_time = time.time()
            total_paused_duration = 0.0
            score = 0
            combo = 0
            max_combo = 0
            total_punches = 0
            kiai_strikes = 0
            bosses_defeated = 0
            powerups_collected = 0
            shields_deflected = 0
            laser_deflected = False
            shield_start_time = 0.0
            dodges_completed = 0
            boss_active = False
            boss_data = None
            boss_waves_spawned.clear()
            active_powerup = None
            powerup_target = None
            kiai_energy = 0
            kiai_mode = False
            kiai_mode_end = 0.0
            is_new_high_score = False
            clap_detector.reset()
            target = spawn_target()
        elif key == ord('s'):  # Manual debug shortcut for Sonic Shield
            if GAME_STATE == "PLAYING":
                shield_start_time = current_time
                create_explosion(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, (255, 255, 0), count=32, is_kiai=True)
        elif key == ord('k'):  # Manual debug shortcut for Kiai Shout
            if GAME_STATE == "PLAYING":
                clap_detector.last_shout_time = current_time
                mic_volume = SHOUT_THRESHOLD * 2.0
                kiai_energy = min(KIAI_MAX, kiai_energy + 25)
                if kiai_energy >= KIAI_MAX and not kiai_mode:
                    kiai_mode = True
                    kiai_mode_end = current_time + KIAI_MODE_DURATION
                    kiai_mode_activated_time = current_time
                    kiai_energy = 0
                print("[Debug] Manual KIAI shout triggered via 'K' key!")
    
finally:
    try:
        camera_pipe.stop()
    except Exception:
        pass
    try:
        pose_tracker.close()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    if AUDIO_AVAILABLE and audio_stream:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass
    try:
        music_mgr.cleanup()
    except Exception:
        pass
