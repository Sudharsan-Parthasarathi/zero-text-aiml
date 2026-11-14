# 🕹️ SHADOWBOX ARCADE

> **YOUR BODY IS THE CONTROLLER.**  
> **Zero-Controller, Real-Time Computer Vision Arcade Platform for Laptop Webcams**  
> Built with **Python 3.11**, **OpenCV**, **MediaPipe**, **NumPy**, and **Pygame**.  
> Runs at **50–60 FPS entirely offline on local CPU / Integrated Graphics**.

---

## 🌟 The Arcade Experience

**SHADOWBOX ARCADE** features two complete, controller-free AR arcade games operated entirely through physical movement, body posture, and natural hand gestures:

```
                                  ┌─────────────────────────────┐
                                  │    04_arcade_launcher.py    │
                                  │     (SHADOWBOX ARCADE)      │
                                  │ "YOUR BODY IS THE CONTROLLER."│
                                  └──────────────┬──────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │       1. SHADOW BOXING        │                 │        2. FRUIT NINJA         │
        │    02_shadow_boxer_game.py    │                 │    03_fruit_ninja_game.py     │
        ├───────────────────────────────┤                 ├───────────────────────────────┤
        │ • Skeletal Pose Tracking      │                 │ • Dual Hand Landmark Tracking │
        │ • 33 BlazePose Keypoints      │                 │ • 21 Hand Joints (Fingertips) │
        │ • Multimodal Transient Clap   │                 │ • Continuous Swept Slicing    │
        │ • Microphone Acoustic Kiai    │                 │ • Rare Heart Extra-Life Fruit │
        │ • Mecha Mini-Boss Waves       │                 │ • Sliced Halves & Splatters   │
        │ • Sonic Shield & Laser Parry  │                 │ • Bombs, Combos, Slow-Mo      │
        │ • 4 Glove Skins & Sizing      │                 │ • Classic, Zen & Arcade Modes │
        └───────────────────────────────┘                 └───────────────────────────────┘
```

---

## 🥊 Game 1: SHADOW BOXING (AR Kinetic Combat)

A multimodal kinetic combat game fusing skeletal pose tracking, biomechanical punch physics, and acoustic sound analysis:
- **Tracking Engine 2.0 (`tracking/`):**
  - **Threaded Camera Pipeline (`CameraPipeline`):** Decoupled background acquisition eliminates hardware driver stalls, delivering a smooth **60 FPS locked rendering frame rate**.
  - **1€ Filter & Forearm Vector Stabilization:** Dual-axis 1€ filter and shortest-path angular filtering eliminate resting landmark jitter while maintaining instant responsiveness during lightning-fast strikes.
  - **Proportional Glove Centering:** Gloves are anatomically centered over the player's physical clenched fists along the elbow-wrist vector ($C_{glove} = W + 0.35 L_{forearm} \hat{v}$).
  - **6-Stage Punch State Machine:** Transitions through `GUARD` $\to$ `ACCELERATION` $\to$ `EXTENSION` $\to$ `IMPACT` $\to$ `RECOVERY`, emitting exactly 1 scoring event per authentic punch stroke.
  - **Punch Style Classification:** Classifies kinematics into `JAB`, `CROSS`, `HOOK`, and `UPPERCUT`.
  - **5-Step Interactive Player Calibration (`[C]` key):** Dynamically measures shoulder width, neutral guard hold, and maximum reach to normalize difficulty for all body sizes.
  - **F3 Diagnostics HUD (`[D]` / `[F3]`):** Real-time telemetry displaying skeletal vectors, punch states, tracking quality %, and frame age.
- **Multimodal Transient Clap Forcefield:** Real-time audio DSP analyzes peak amplitude, crest factor, and energy onset combined with body-relative wrist convergence to trigger the **Sonic Shield** (TAB key displays on-screen acoustic diagnostic HUD).
- **Laser Deflection & Parry Reward:** Ducking under lasers dodges the hazard; deflecting the laser with an active Sonic Shield rewards **+75 parry points** and sub-bass shockwaves!
- **Acoustic "KIAI" Battle Cry:** Shouting into your microphone on punch impact triggers **5x Critical Strikes**, golden explosions, and heavy boss damage. Punches and battle cries charge your 10-second **Kiai Overdrive** bonus mode! (Acoustic diagnostics on `[TAB]`, manual test on `[K]`, and audio audit via `python scripts/check_mic.py`).
- **Customizable Gloves & Skins:** Visual glove scaling decoupled from collision hitboxes, featuring 4 unlockable skins: *Classic Red*, *Electric Blue*, *Neon Purple*, and *Golden Champion*.
- **Mecha Mini-Boss Encounters:** Face off against *MECHA-BRUTE* and *CYBER-TITAN* with segmented floating HP bars, rotating armor shields, and dynamic impact recoil.

---

## 🍉 Game 2: FRUIT NINJA 2.0 (Fist-Controlled Laser Sword Edition)

The classic fruit-slicing arcade experience transformed into an authentic webcam-powered sci-fi spectacle:
- **Natural Fist-Grip Sword & Biomechanical Palm Frame (`laser_sword.py`):**
  - Grip Anchor rigidly bound to palm and knuckle region: $G = W + 0.58(K - W)$.
  - **Landmark 12 Invariance:** Position and blade vector are derived exclusively from wrist and MCP joints 5, 9, 13, 17; moving fingertips has zero influence on sword anchor or blade angle.
  - **1€ Filter (One Euro Filter) Stabilization:** Dynamic cutoff frequency eliminates resting landmark jitter ($>65\%$ variance reduction) while preserving instant response during high-speed swings.
  - **Symmetrical Collision Geometry:** Slicing hitbox and visual plasma beam share identical $(P_{base}, P_{tip})$ coordinates, eliminating any finger-pointer artifacts.
  - Left Hand: **Electric Cyan Energy Blade**
  - Right Hand: **Neon Amber / Ruby Energy Blade**
  - Dual-wielding enabled: swing with both hands for maximum score multipliers!
  - Dynamic ignition snap-hiss and power-down collapse audio synthesis.
- **Closed-Fist Grip Engine & Auto-Grip Mode:**
  - Biomechanical closed-fist classifier evaluates normalized fingertip-to-palm distances scaled to hand geometry.
  - Temporal smoothing with hysteresis ($0.52 / 0.36$) and $80\text{ms}$ debounce prevents state flickering.
  - Optional **Auto-Grip Mode** (`[A]` key or settings) keeps blades permanently ignited whenever hands are tracked.
- **In-Game F3 Diagnostics HUD:**
  - Press `[F3]` at any time to toggle skeletal landmarks, knuckle center $K$, grip anchor $G$, forward axis $\vec{u}$, lateral axis $\vec{n}$, and real-time tracking metrics (fist score, angle, velocity, and filter deltas).
  - Temporal smoothing with hysteresis prevents state flickering.
  - Optional **Auto-Grip Mode** (`[A]` key or settings) keeps blades permanently ignited whenever hands are tracked.
- **Continuous Swept Blade Collision (Anti-Tunneling):**
  - Multi-sample capsule sweeping accounts for blade length, width, and velocity between consecutive frames, guaranteeing 100% slice registration at any swipe speed.
  - Intentional velocity gate rejects stationary blade camping.
- **5-Step Interactive Dojo Onboarding & Calibration:**
  - Step 1: Detect Hand in Camera
  - Step 2: Clench Fist to Ignite Energy Blade
  - Step 3: Move & Swing Sword to Build Momentum
  - Step 4: Slice Practice Watermelon
  - Step 5: Certified Dojo Master — Enter the Dojo!
- **3 Dynamic Difficulty Settings:**
  - 🟢 **EASY (Default):** $1.20\times$ enlarged fruits, floatier gravity, slower wave intervals, and lower hazard rates.
  - 🟡 **CLASSIC:** Authentic arcade physics, timing, and density.
  - 🔴 **EXPERT:** $0.90\times$ smaller fruit hitboxes, accelerated gravity, dense rapid-fire waves, and frequent hazard bombs.
- **Modernized UI/UX Redesign:**
  - Real-time **Tracking Status Badge** (`🟢 SWORD ACTIVE (FIST GRIPPED)`, `🟡 CLOSE FIST TO IGNITE`, `⚪ NO HAND DETECTED`).
  - Interactive clickable and sliceable 3-card mode selector (`[1] Classic`, `[2] Zen`, `[3] Arcade`).
  - High-tech arcade game over scoreboard with full accuracy and combo breakdown.
- **Physical Fruit Halves & Splatters:** Slicing splits fruits into rotating halves governed by parabolic gravity, spitting juice particles and stamping cutting board decals.
- **Three Game Modes:**
  - 🍉 **Classic Mode:** 60-second timer, 3 lives, bombs, and rare **Heart Extra-Life Fruits** (restores a lost life or awards +50 pts).
  - 🍊 **Zen Mode:** Relaxed slicing with no bombs or life penalties.
  - 🌟 **Arcade Mode:** 60-second high-density frenzy with special bonus fruits and non-lethal bomb penalties (-25 pts).
- **Special Fruits & Hazards:**
  - 💣 **Classic Iron Bombs:** Burning spark fuse; detonates with violent screen shake and sub-bass shockwave.
  - 🌟 **Golden Bonus Fruit:** Worth +30 points with golden fireworks.
  - ❄️ **Frozen Slow-Mo Fruit:** Slows physics to $0.45\times$ speed for 4.5 seconds with frosty particle auras.
  - 💖 **Heart Extra-Life Fruit:** Rare pink glowing heart fruit in Classic mode that recovers a lost life.

---

## 👤 Persistent Player Profiles & Settings

- **Local Multi-Profile System (`player_profiles.py` / `player_profiles.json`):**
  - Prompt: *"WHO'S PLAYING TODAY?"* on startup.
  - Case-insensitive profile normalization, sanitized names (up to 18 characters).
  - Individual records tracked per game: personal bests, total matches, punches landed, Kiai strikes, bosses defeated, shields deflected, fruit sliced, and combos.
  - Safe atomic writes via temporary files with `.bak` recovery.
  - Automatic migration from legacy `high_score.json` and `fruit_ninja_high_score.json` into default Guest profile.
- **In-Game Settings (`04_arcade_launcher.py`):**
  - Independent volume sliders for Music and SFX.
  - Global Mute toggle (`M`).
  - Glove Size selection (Large, Extra Large, Custom).
  - Glove Skin selection (Classic Red, Electric Blue, Neon Purple, Golden).
  - Fullscreen toggle (`F`).

---

## 🎵 Centralized Music & Soundtracks

Procedural synthwave and dojo soundtracks generated via `generate_music_tracks.py` and coordinated through `music_manager.py`:
- `menu_theme.wav`: Driving synthwave rhythm for the arcade launcher.
- `boxing_theme.wav`: High-octane electronic beat for boxing matches.
- `fruit_ninja_theme.wav`: Pentatonic eastern dojo melody for fruit slicing.
- Seamless looping, smooth fade transitions, and clean process release when switching games.

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.9 – 3.11
- Laptop webcam and microphone
- Windows, macOS, or Linux

### 2. Install Dependencies
```bash
# Clone the repository
git clone https://github.com/Noobi-Programmer/Shadowbox.git
cd Shadowbox

# Install required packages
pip install opencv-python mediapipe numpy pygame sounddevice pytest pillow
```

### 3. Launch the Games

* **Launch the Unified Arcade Portal (Recommended):**
  ```bash
  python 04_arcade_launcher.py
  ```
  *(Press `1` for Shadow Boxing, `2` for Fruit Ninja, `P` for Profiles, `S` for Settings, `Q` to Quit)*

* **Or Launch Shadow Boxing directly:**
  ```bash
  python 02_shadow_boxer_game.py [--player-id <id>]
  ```
  *(Press `TAB` for Acoustic Diagnostic HUD, `S` to test shield, `K` to test Kiai shout, `R` to restart, `Q` to quit)*

* **Or Launch Fruit Ninja directly:**
  ```bash
  python 03_fruit_ninja_game.py [--player-id <id>] [--fullscreen]
  ```

---

## 🧪 Automated Testing Suite

All 74 unit and integration tests run in $< 11$ seconds across 10 test modules:

```bash
pytest tests/ -v
```

Tests cover:
- **Tracking Engine 2.0 (`test_tracking_engine.py`):** 1€ filters (1D/2D), angular shortest-path wrapping, outlier detector, coordinate mapper, tracking quality state machine, anatomical left/right swap resolution, punch detector 6-state machine, punch classification, player calibration, and threaded camera pipeline.
- **Fist-Controlled Laser Swords (`test_laser_sword.py`):** Landmark 12 invariance, knuckle & grip centers, palm scaling, 1€ filter stabilization, reacquisition reset, symmetrical collision vs rendering geometry, auto-grip, and F3 overlay.
- **Hand Tracker (`test_hand_tracker.py`):** Laser sword active slices, blade trail tip tracking, OpenCV rendering.
- **Arcade Polish & Mechanics (`test_arcade_polish.py`):** Heart fruit mechanics, profile logging, clap convergence, laser parries.
- **Audio Engines (`test_audio_engine.py`, `test_music_manager.py`):** Dynamic mixing, volume sync, mute toggles, procedural soundtracks.
- **Multimodal Claps & Kiai DSP (`test_clap_detector.py`):** Dynamic thresholding, speech rejection, refractory periods, acoustic shout tracking, punch impact latch windows, sensitivity scaling, and stereo downmixing.
- **Fruit Physics (`test_fruit_physics.py`):** Swept slice collision, parabolic motion, sliced halves, wave spawner.
- **Player Profiles (`test_player_profiles.py`):** Atomic persistence, per-player stats, legacy migrations.
- **Headless Lifecycles (`test_game_loop.py`, `test_launcher.py`):** State transitions and clean teardowns.
