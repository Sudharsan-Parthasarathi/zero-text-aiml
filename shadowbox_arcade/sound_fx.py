"""
=============================================================================
SHADOWBOX ARCADE — ADVANCED BOXING SOUND ENGINE (sound_fx.py)
Hybrid Studio Sample & Procedural NumPy Audio DSP Architecture
=============================================================================
"""

import os
import time
import random
import numpy as np

# -----------------------------------------------------------------------------
# 1. MIXER INITIALIZATION & CHANNEL ALLOCATION
# -----------------------------------------------------------------------------

SOUND_AVAILABLE = False
_sounds = {}
_punch_medium_bank = []
_punch_heavy_bank = []
_punch_soft_bank = []
_whoosh_bank = []
_boss_metal_bank = []
_last_sfx_time = {}

ASSET_BOXING_DIR = os.path.join(os.path.dirname(__file__), "assets", "audio", "boxing")

try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(16)
    SOUND_AVAILABLE = True
except Exception as e:
    print(f"[Sound FX Warning] Could not initialize pygame.mixer: {e}. Silent mode enabled.")

def _to_sound(audio_mono, sample_rate=44100, volume=0.92):
    """Normalize and convert 1D float numpy array into 16-bit stereo pygame Sound."""
    if not SOUND_AVAILABLE:
        return None
    import pygame
    max_val = np.max(np.abs(audio_mono))
    if max_val > 0:
        audio_mono = audio_mono / max_val
    audio_int = (audio_mono * 32767 * volume).astype(np.int16)
    stereo = np.column_stack((audio_int, audio_int))
    return pygame.sndarray.make_sound(stereo)

# -----------------------------------------------------------------------------
# 2. HIGH-FIDELITY PROCEDURAL BOXING AUDIO SYNTHESIZERS (ZERO-LATENCY FALLBACK)
# -----------------------------------------------------------------------------

def _build_realistic_punch(variation=0):
    """Procedural physical model of leather glove on padded bag impact."""
    sr = 44100
    dur = 0.16 + (variation * 0.015)
    t = np.linspace(0, dur, int(sr * dur), False)
    
    crack_freqs = [1250, 1100, 1400, 980]
    sub_freqs = [52, 46, 58, 48]
    mid_starts = [190, 175, 210, 165]
    mid_ends = [65, 58, 72, 60]
    
    cf = crack_freqs[variation % 4]
    sf = sub_freqs[variation % 4]
    ms = mid_starts[variation % 4]
    me = mid_ends[variation % 4]
    
    # 1. Leather crack transient (initial contact snap)
    crack = np.sin(2 * np.pi * cf * t) * np.exp(-110 * t)
    noise_burst = np.random.uniform(-0.45, 0.45, len(t)) * np.exp(-130 * t)
    
    # 2. Resonant mid-body punch (foam compression thud)
    body_pitch = np.linspace(ms, me, len(t))
    body_tone = np.sin(2 * np.pi * body_pitch * t) * np.exp(-24 * t)
    
    # 3. Visceral sub-bass thump (heavy bag impulse)
    sub_bass = np.sin(2 * np.pi * sf * t) * np.exp(-12 * t)
    
    raw = 0.28 * crack + 0.22 * noise_burst + 0.48 * body_tone + 0.85 * sub_bass
    wave = np.tanh(1.75 * raw)
    return _to_sound(wave, volume=0.92)

def _build_critical_punch():
    """Heavy explosive impact for KIAI / Battle Cry power punches."""
    sr = 44100
    dur = 0.38
    t = np.linspace(0, dur, int(sr * dur), False)
    crack = np.sin(2 * np.pi * 1600 * t) * np.exp(-120 * t)
    noise = np.random.uniform(-0.55, 0.55, len(t)) * np.exp(-80 * t)
    sub = np.sin(2 * np.pi * np.linspace(95, 38, len(t)) * t) * np.exp(-7.5 * t)
    wave = np.tanh(2.2 * (0.35 * crack + 0.25 * noise + 1.1 * sub))
    return _to_sound(wave, volume=0.98)

def _build_whoosh_sound(variation=0):
    """Aerodynamic fist displacement through air."""
    sr = 44100
    dur = 0.14 + variation * 0.02
    t = np.linspace(0, dur, int(sr * dur), False)
    freq = np.linspace(850 + variation * 100, 220, len(t))
    wind = np.random.uniform(-0.5, 0.5, len(t))
    tone = np.sin(2 * np.pi * freq * t)
    env = np.sin(np.pi * (t / dur)) ** 2.2
    wave = np.tanh(1.5 * (0.5 * tone + 0.5 * wind) * env)
    return _to_sound(wave, volume=0.75)

def _build_shield_sound():
    """Sonic Forcefield activation resonant hum."""
    sr = 44100
    dur = 0.35
    t = np.linspace(0, dur, int(sr * dur), False)
    f1 = 580.0
    f2 = 1160.0
    hum = (np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)) * np.exp(-6.5 * t)
    shimmer = 0.2 * np.sin(2 * np.pi * 2320 * t) * np.exp(-14 * t)
    wave = np.tanh(1.6 * (hum + shimmer))
    return _to_sound(wave, volume=0.88)

def _build_parry_sound():
    """Laser deflection parry ping with sub-bass detonation."""
    sr = 44100
    dur = 0.45
    t = np.linspace(0, dur, int(sr * dur), False)
    ping = np.sin(2 * np.pi * np.linspace(2400, 950, len(t)) * t) * np.exp(-16 * t)
    sub = np.sin(2 * np.pi * np.linspace(120, 42, len(t)) * t) * np.exp(-6.0 * t)
    sparkle = np.random.uniform(-0.4, 0.4, len(t)) * np.exp(-35 * t)
    wave = np.tanh(1.8 * (0.6 * ping + 0.7 * sub + 0.3 * sparkle))
    return _to_sound(wave, volume=0.95)

def _build_bell_sound():
    """Resonant boxing ring bell for round start and match alerts."""
    sr = 44100
    dur = 1.2
    t = np.linspace(0, dur, int(sr * dur), False)
    f1, f2, f3 = 1046.5, 1318.5, 2093.0
    bell = (np.sin(2 * np.pi * f1 * t) + 0.6 * np.sin(2 * np.pi * f2 * t) + 0.3 * np.sin(2 * np.pi * f3 * t)) * np.exp(-3.5 * t)
    return _to_sound(bell, volume=0.90)

def _build_boss_hit_sound():
    """Metallic crunch and armor resonance."""
    sr = 44100
    duration = 0.22
    t = np.linspace(0, duration, int(sr * duration), False)
    bass = np.sin(2 * np.pi * np.linspace(160, 42, len(t)) * t)
    metal = np.sin(2 * np.pi * 780 * t) * np.exp(-22 * t)
    crunch = np.clip(np.random.normal(0, 0.75, len(t)), -0.9, 0.9)
    env = np.exp(-11 * t)
    return _to_sound(np.tanh(1.8 * (0.5 * bass + 0.35 * metal + 0.45 * crunch) * env))

def _build_boss_defeat_sound():
    """Triumphant chord and explosive destruction."""
    sr = 44100
    duration = 0.90
    t = np.linspace(0, duration, int(sr * duration), False)
    boom = np.sin(2 * np.pi * np.linspace(140, 25, len(t)) * t) * np.exp(-3.2 * t)
    noise = np.random.uniform(-0.5, 0.5, len(t)) * np.exp(-4.5 * t)
    chord = (np.sin(2 * np.pi * 523.25 * t) + np.sin(2 * np.pi * 659.25 * t) + np.sin(2 * np.pi * 783.99 * t)) / 3.0
    chord_env = np.clip((t - 0.15) * 4.0, 0, 1) * np.exp(-2.2 * (t - 0.15))
    chord_env[t < 0.15] = 0
    return _to_sound((0.6 * boom + 0.4 * noise + 0.7 * chord * chord_env))

def _build_powerup_sound():
    """Ascending harmonic chime."""
    sr = 44100
    duration = 0.35
    t = np.linspace(0, duration, int(sr * duration), False)
    freq = np.linspace(400, 1200, len(t))
    tone = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(4 * np.pi * freq * t)
    sparkle = 0.3 * np.sin(2 * np.pi * 2400 * t) * np.sin(2 * np.pi * 32 * t)
    env = np.clip(1.0 - t / duration, 0, 1) ** 0.8
    return _to_sound((tone + sparkle) * env)

# -----------------------------------------------------------------------------
# 3. SOUND ASSET LOADER (HYBRID KENNEY CC0 + PROCEDURAL FALLBACK)
# -----------------------------------------------------------------------------

def _load_or_synth():
    global _sounds, _punch_medium_bank, _punch_heavy_bank, _punch_soft_bank, _whoosh_bank, _boss_metal_bank
    if not SOUND_AVAILABLE:
        return

    def load_file(fname):
        p = os.path.join(ASSET_BOXING_DIR, fname)
        if os.path.exists(p):
            try:
                return pygame.mixer.Sound(p)
            except Exception:
                pass
        return None

    # 1. Medium punch bank (Jabs & Crosses)
    m_files = ["impact_punch_medium_000.ogg", "impact_punch_medium_001.ogg", "impact_punch_medium_002.ogg"]
    loaded_m = [load_file(f) for f in m_files]
    _punch_medium_bank = [s for s in loaded_m if s is not None]
    if not _punch_medium_bank:
        _punch_medium_bank = [_build_realistic_punch(i) for i in range(3)]

    # 2. Heavy punch bank (Hooks & Power Hits)
    h_files = ["impact_punch_heavy_000.ogg", "impact_punch_heavy_001.ogg", "impact_punch_heavy_002.ogg"]
    loaded_h = [load_file(f) for f in h_files]
    _punch_heavy_bank = [s for s in loaded_h if s is not None]
    if not _punch_heavy_bank:
        _punch_heavy_bank = [_build_realistic_punch(i + 2) for i in range(3)]

    # 3. Soft heavy / Block bank (Padded glove thud)
    s_files = ["impact_soft_heavy_000.ogg", "impact_soft_heavy_001.ogg"]
    loaded_s = [load_file(f) for f in s_files]
    _punch_soft_bank = [s for s in loaded_s if s is not None]
    if not _punch_soft_bank:
        _punch_soft_bank = [_build_realistic_punch(0)]

    # 4. Boss Metal Impact Bank
    bm_files = ["impact_metal_heavy_000.ogg", "impact_metal_heavy_001.ogg"]
    loaded_bm = [load_file(f) for f in bm_files]
    _boss_metal_bank = [s for s in loaded_bm if s is not None]
    if not _boss_metal_bank:
        _boss_metal_bank = [_build_boss_hit_sound()]

    # 5. Whoosh Bank
    _whoosh_bank = [_build_whoosh_sound(i) for i in range(3)]

    # 6. Specific Audio Mappings
    bell_file = load_file("impact_bell_heavy_000.ogg")
    _sounds["bell"] = bell_file if bell_file else _build_bell_sound()

    plate_file = load_file("impact_plate_heavy_000.ogg")
    _sounds["shield"] = plate_file if plate_file else _build_shield_sound()
    _sounds["parry"] = _build_parry_sound()

    _sounds["punch"] = _punch_medium_bank[0]
    _sounds["critical"] = _build_critical_punch()
    _sounds["boss_hit"] = _boss_metal_bank[0]
    _sounds["boss_defeat"] = _build_boss_defeat_sound()
    _sounds["powerup"] = _build_powerup_sound()
    _sounds["dodge"] = _whoosh_bank[0]
    _sounds["whoosh"] = _whoosh_bank[0]

if SOUND_AVAILABLE:
    try:
        _load_or_synth()
    except Exception as e:
        print(f"[Sound FX Error] Pre-loading audio failed: {e}")

# -----------------------------------------------------------------------------
# 4. HARDWARE HAPTIC INTEGRATION STUB
# -----------------------------------------------------------------------------

_arduino_serial = None
HAPTIC_ENABLED = False

def init_haptic(port=None, baudrate=115200):
    global _arduino_serial, HAPTIC_ENABLED
    try:
        import serial
        if port:
            _arduino_serial = serial.Serial(port, baudrate, timeout=0.05)
            HAPTIC_ENABLED = True
            print(f"[Haptic] Connected to Arduino on {port}!")
    except Exception:
        HAPTIC_ENABLED = False

def trigger_haptic(event_name="punch"):
    if HAPTIC_ENABLED and _arduino_serial:
        try:
            cmd = {"punch": b'P', "kiai": b'K', "shield": b'S', "boss": b'B', "powerup": b'U'}.get(event_name, b'P')
            _arduino_serial.write(cmd)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# 5. ASYNC AUDIO PLAYBACK & MIXING ENGINE
# -----------------------------------------------------------------------------

def play_sfx(name, combo_idx=0, volume_mult=1.0):
    """
    Play sound effect asynchronously with rate limiting, channel allocation,
    and dynamic master SFX volume synchronization.
    """
    if not SOUND_AVAILABLE:
        return
    now = time.time()
    
    # Rate limit rapid repeated sounds to avoid clipping / acoustic distortion
    if name in ("punch", "critical"):
        last_t = _last_sfx_time.get("punch", 0.0)
        if (now - last_t) < 0.065:
            return
        _last_sfx_time["punch"] = now
    elif name == "whoosh":
        last_t = _last_sfx_time.get("whoosh", 0.0)
        if (now - last_t) < 0.140:
            return
        _last_sfx_time["whoosh"] = now

    # Retrieve active SFX volume from centralized MusicManager
    try:
        from music_manager import get_music_manager
        mm = get_music_manager()
        if mm.is_muted:
            return
        sfx_vol = mm.sfx_volume * volume_mult
    except Exception:
        sfx_vol = 0.85 * volume_mult

    try:
        snd = None
        if name == "punch":
            # Select variation between medium punch impacts
            if _punch_medium_bank:
                snd = random.choice(_punch_medium_bank)
            else:
                snd = _sounds.get("punch")
        elif name == "heavy_punch":
            if _punch_heavy_bank:
                snd = random.choice(_punch_heavy_bank)
            else:
                snd = _sounds.get("critical")
        elif name == "block":
            if _punch_soft_bank:
                snd = random.choice(_punch_soft_bank)
            else:
                snd = _sounds.get("punch")
        elif name == "boss_hit":
            if _boss_metal_bank:
                snd = random.choice(_boss_metal_bank)
            else:
                snd = _sounds.get("boss_hit")
        elif name == "whoosh":
            if _whoosh_bank:
                snd = random.choice(_whoosh_bank)
            else:
                snd = _sounds.get("whoosh")
        elif name in _sounds:
            snd = _sounds[name]

        if snd:
            snd.set_volume(max(0.0, min(1.0, sfx_vol)))
            # Dedicated channel playback avoids cutting off other sounds
            ch = pygame.mixer.find_channel()
            if ch:
                ch.play(snd)
            else:
                snd.play()
    except Exception:
        pass
