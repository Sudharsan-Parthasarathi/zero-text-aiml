"""
=============================================================================
SHADOWBOX ARCADE — AUDIO ENGINE & DSP UNIT TESTS (test_audio_engine.py)
Tests boxing sound fx, fruit ninja audio manager, and UI audio feedback.
=============================================================================
"""

import os
import time
import pytest
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sound_fx
from sound_fx import (
    play_sfx,
    _build_realistic_punch,
    _build_critical_punch,
    _build_shield_sound,
    _build_parry_sound,
    _build_bell_sound,
    _build_whoosh_sound,
    _build_boss_hit_sound,
    _build_boss_defeat_sound,
)
from fruit_sound_fx import FruitAudioManager, get_audio_manager
from ui_audio import UIAudio, get_ui_audio
from music_manager import get_music_manager

@pytest.fixture(autouse=True)
def init_pygame_audio():
    if not pygame.get_init():
        pygame.init()
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    yield

# -----------------------------------------------------------------------------
# 1. SHADOW BOXING AUDIO ENGINE TESTS
# -----------------------------------------------------------------------------

def test_boxing_sound_fx_initialization():
    assert sound_fx.SOUND_AVAILABLE is True
    assert pygame.mixer.get_num_channels() >= 16

    # Verify procedural buffers return valid Sound objects
    assert isinstance(_build_realistic_punch(variation=0), pygame.mixer.Sound)
    assert isinstance(_build_realistic_punch(variation=1), pygame.mixer.Sound)
    assert isinstance(_build_critical_punch(), pygame.mixer.Sound)
    assert isinstance(_build_shield_sound(), pygame.mixer.Sound)
    assert isinstance(_build_parry_sound(), pygame.mixer.Sound)
    assert isinstance(_build_bell_sound(), pygame.mixer.Sound)
    assert isinstance(_build_whoosh_sound(), pygame.mixer.Sound)
    assert isinstance(_build_boss_hit_sound(), pygame.mixer.Sound)
    assert isinstance(_build_boss_defeat_sound(), pygame.mixer.Sound)

def test_boxing_sound_fx_playback():
    # Test all actions via play_sfx
    for action in ["punch", "heavy_punch", "critical", "shield", "parry", "dodge", "block", "bell", "whoosh", "boss_hit", "boss_defeat"]:
        play_sfx(action)

def test_boxing_punch_rate_limiting():
    # Reset punch timestamp
    sound_fx._last_sfx_time["punch"] = 0.0
    
    # First punch should update timestamp
    play_sfx("punch")
    t1 = sound_fx._last_sfx_time.get("punch", 0.0)
    assert t1 > 0

    # Rapid punch right after (within 65ms cooldown) should be gated
    play_sfx("punch")
    t2 = sound_fx._last_sfx_time.get("punch", 0.0)
    assert t1 == t2

# -----------------------------------------------------------------------------
# 2. FRUIT NINJA AUDIO MANAGER TESTS
# -----------------------------------------------------------------------------

def test_fruit_audio_manager_initialization():
    fam = FruitAudioManager()
    assert fam is not None
    assert len(fam.swoosh_bank) > 0
    assert "watermelon" in fam.slice_bank
    assert "apple" in fam.slice_bank
    assert "coconut" in fam.slice_bank
    assert "orange" in fam.slice_bank
    assert len(fam.sword_swings) > 0

def test_fruit_audio_playback():
    fam = get_audio_manager()
    # Per-fruit slicing
    fam.play_slice("watermelon")
    fam.play_slice("apple")
    fam.play_slice("coconut")
    fam.play_slice("orange")

    # Combos, bombs, and utilities
    fam.play_combo(2)
    fam.play_combo(5)
    fam.play_bomb()
    fam.play_swoosh()
    fam.play_countdown()
    fam.play_game_over()
    fam.play_bonus()
    fam.play_ui_click()

    # Laser sword audio
    fam.play_sword_ignite()
    fam.play_sword_retract()
    fam.play_sword_swing(speed=450.0)

# -----------------------------------------------------------------------------
# 3. UI AUDIO SYSTEM TESTS
# -----------------------------------------------------------------------------

def test_ui_audio_initialization_and_actions():
    ui = UIAudio()
    expected_actions = ["click", "select", "toggle", "open", "close", "confirm", "launch", "back", "error"]
    for act in expected_actions:
        assert act in ui.sounds
        assert len(ui.sounds[act]) > 0

    # Test playing each action
    for act in expected_actions:
        ui.play(act)

def test_audio_dynamic_volume_and_mute_sync():
    music_mgr = get_music_manager()
    ui = get_ui_audio()

    # When muted, playing UI sound should be a clean no-op
    music_mgr.is_muted = True
    ui.play("click")

    # When unmuted and volume > 0, it should execute cleanly
    music_mgr.is_muted = False
    music_mgr.set_sfx_volume(0.8)
    assert music_mgr.sfx_volume == 0.8
    ui.play("click")
