# 🎧 TASK BRIEF: SOUND DESIGN OVERHAUL & BROWSER-AGENT ASSET PROCUREMENT

**Repository:** `Noobi-Programmer/Shadowbox`  
**Target Environment:** Local CPU / Integrated Graphics Laptop Arcade System  
**Audio Stack:** Python 3.11, Pygame 2.6.1 Mixer (44.1 kHz, 16-bit stereo, buffer=512), NumPy Procedural DSP  
**Assigned Agent Role:** OpenCode Browser-Capable Sound Engineer & Audio DSP Specialist  

---

## 1. Mission & Scope

You are the lead sound designer and audio software engineer for **SHADOWBOX ARCADE**. Your mission is to conduct a professional sound design overhaul across both integrated arcade games (**SHADOW BOXING** and **FRUIT NINJA**) as well as the **ARCADE LAUNCHER**.

You have access to a live web browser and local repository file access. You must research, audition, download, verify, and document free, public-domain, and open-source game audio assets that elevate the game feel into an arcade-quality production.

---

## 2. Audio Identity Directives

### A. SHADOW BOXING (AR Kinetic Combat Dojo)
*Theme: Energetic, visceral, arcade combat with physical punch impact.*
- **Punch Impacts (Light, Medium, Heavy):**
  - Light (Jab/Flick): Snappy leather glove slap, minimal sub-bass, fast decay (< 120ms).
  - Medium (Cross/Hook): Resonant foam compression thud with clean mid-frequency crack (180–350ms).
  - Heavy / Critical (KIAI Power Strike): Visceral sub-bass (45–60 Hz) chest thump with dramatic impact saturation (400–650ms).
- **Movement & Defensive Actions:**
  - Whoosh: Aerodynamic air displacement for fast punch extensions and dodges.
  - Block / Deflection: Padded leather-on-glove soft heavy thud.
  - Sonic Shield & Laser Parry: Resonant forcefield harmonic activation followed by a high-energy metallic laser deflection ping and sub-bass shockwave.
- **Combat Milestones & Feedback:**
  - Boxing Round Bell: Resonant metallic bell signaling start of match and mini-boss arrivals.
  - Mecha Boss Impacts: Metallic armor crunch and hydraulic stress impacts.
  - Boss Defeat: Dramatic explosive destruction transient with triumphant ascending fanfare.

### B. FRUIT NINJA (Webcam AR Dojo)
*Theme: Satisfying, crisp, juicy, physical cutting action.*
- **Blade & Slicing Action:**
  - Blade Whoosh: Aerodynamic katana swipe with pitch variation.
  - Laser Sword Ignition / Retraction: Crisp sci-fi energy blade snap-hiss on fist clench and collapse on release.
  - Laser Sword Hum: Subtle resonant FM plasma hum on high-speed sweeps ($> 380\text{ px/s}$). No annoying continuous drone when stationary.
  - Distinct Fruit Impacts:
    - *Watermelon*: Deep fleshy thwack with wet pulp squelch.
    - *Apple*: Crisp, brittle snapping skin crunch.
    - *Coconut*: Hollow, resonant hard shell crack.
    - *Citrus / Soft Fruits*: Juicy pulp burst and droplet splatter.
- **Game Events & Hazards:**
  - Combo Milestones: Ascending pentatonic chime arpeggios ($2\times$ through $7\times$ multipliers).
  - Bomb Fuse & Blast: Authentic burning spark rope hiss followed by sub-bass detonation shockwave.
  - Countdown & Game Over: Authentic dojo temple bell countdown and dramatic minor descent string.

### C. ARCADE LAUNCHER & SYSTEM-WIDE UI
*Theme: Tactile, clean, modern arcade cabinet.*
- Card Hover: Subtle high-frequency blip / tick.
- Card Select / Game Launch: Clean affirmative confirmation fanfare.
- Settings Toggle: Distinct ON (rising pitch) and OFF (descending pitch) mechanical toggles.
- Modal Open / Dismiss: Smooth acoustic aperture sweeps.
- Volume Step / Feedback: Immediate real-time auditory confirmation of current volume level.

---

## 3. Approved Sound Discovery Sources

You must exclusively source candidate audio from verified CC0 (Public Domain) or clearly permissive open-source repositories:
1. **Kenney.nl (Asset packs & GitHub repositories)**
   - UI Audio, Interface Sounds, Impact Sounds, Digital Audio, RPG Audio
   - License: **CC0 1.0 Universal**
2. **OpenGameArt.org**
   - Public domain and CC0 submissions (filter explicitly by CC0).
3. **Freesound.org**
   - Filter strictly by Creative Commons 0.
4. **Pixabay Audio**
   - Verify compatibility with local offline game distribution.

### 🚫 Strict Disqualifications:
- **NO** ripped assets from commercial titles (e.g. halfbrick Fruit Ninja, Capcom, Nintendo).
- **NO** copyrighted YouTube, Spotify, or SoundCloud rips.
- **NO** non-commercial licenses (CC-NC) or ShareAlike (CC-BY-SA) that conflict with project licensing.
- **NO** unverified assets without explicit license text.

---

## 4. Step-by-Step Execution Protocol

1. **Browser Auditioning**:
   - Open candidate audio URLs in browser.
   - Listen to and preview each candidate audio file at standard volume.
   - Verify sample rate (44.1 kHz or 48 kHz preferred), bit depth (16-bit or 24-bit), and format (WAV or OGG preferred).
   - Reject files with background hiss, clipping distortion, long silent tails, or mismatched character.
2. **Local Installation**:
   - Install selected files into the organized repository paths:
     - `assets/audio/boxing/`
     - `assets/audio/fruit_ninja/`
     - `assets/audio/ui/`
     - `assets/audio/music/`
   - Use standardized, descriptive snake_case filenames.
3. **Audio Integrity & Pygame Compatibility Check**:
   - Run verification script:
     ```powershell
     .venv\Scripts\python.exe -c "import pygame; pygame.init(); pygame.mixer.init(); s = pygame.mixer.Sound('assets/audio/boxing/<file>'); print(s.get_length())"
     ```
   - Ensure the asset loads cleanly in SDL2 / Pygame without decoding exceptions.
4. **Licensing & Attribution Recording**:
   - Update `assets/audio/SOURCES.md` with:
     - File name & local path
     - Direct source URL
     - Creator / Artist
     - License type (e.g. CC0 1.0 Universal)
     - Gameplay event mapping
5. **Deliverable Report**:
   - Generate `docs/SOUND_ASSET_RESEARCH_REPORT.md` following the required structure.

---

## 5. Audition & Acceptance Matrix

| Event Name | Primary Frequency | Ideal Duration | Accepted Format | Source Pack |
|---|---|---|---|---|
| `punch_medium` | 180–350 Hz + leather transient | 0.35–0.55s | OGG / WAV | Kenney Impact Sounds |
| `punch_heavy` | 45–60 Hz sub + 1.2 kHz crack | 0.45–0.65s | OGG / WAV | Kenney Impact Sounds |
| `punch_soft_block` | 110–220 Hz low-Q thud | 0.30–0.50s | OGG / WAV | Kenney Impact Sounds |
| `boxing_bell` | 780 Hz + 1560 Hz harmonic bell | 1.2–1.8s | OGG / WAV | Kenney Impact Sounds |
| `shield_laser_parry` | 2.4 kHz ping + 50 Hz impulse | 0.40–0.60s | OGG / WAV | Kenney Impact Sounds |
| `boss_metal_hit` | 650–900 Hz metallic stress | 0.15–0.35s | OGG / WAV | Kenney Impact Sounds |
| `ui_click` | 1.4 kHz short tick | 0.05–0.12s | WAV | Kenney Interface Sounds |
| `ui_select` | 880–1200 Hz clean pip | 0.05–0.08s | WAV | Kenney Interface Sounds |
| `ui_toggle_on` | Rising pitch chirps | 0.12–0.16s | WAV | Kenney Interface Sounds |
| `ui_toggle_off` | Descending pitch chirps | 0.12–0.16s | WAV | Kenney Interface Sounds |
| `ui_confirm` | Two-tone rising chord | 0.25–0.55s | WAV | Kenney Interface Sounds |
| `ui_modal_open` | Soft acoustic aperture open | 0.12–0.18s | WAV | Kenney Interface Sounds |
| `ui_modal_close` | Soft acoustic aperture close | 0.12–0.18s | WAV | Kenney Interface Sounds |
| `ui_back` | Gentle drop transient | 0.06–0.10s | WAV | Kenney Interface Sounds |
| `slice_fruit_impact`| Crisp blade edge + squelch | 0.18–0.30s | OGG / WAV | Kenney Match-3 / FPS |
| `bomb_explosion` | 30–80 Hz detonation + debris | 0.90–1.40s | OGG / WAV | Kenney FPS Starter Kit |

---

## 6. Handoff Instructions

When execution begins:
1. Verify internet connectivity and repository write access.
2. Review the existing procedural DSP implementations in `sound_fx.py` and `fruit_sound_fx.py`.
3. Execute the procurement pipeline or download candidates via browser.
4. Execute `pytest tests/` to confirm that audio additions cause zero test regressions.
5. Compile and deliver `docs/SOUND_ASSET_RESEARCH_REPORT.md`.
