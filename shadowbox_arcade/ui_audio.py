"""
=============================================================================
SHADOWBOX ARCADE — UI AUDIO SYSTEM (ui_audio.py)
Premium Tactile Sound Feedback for Arcade Navigation & Settings
=============================================================================
"""

import os
import random
import pygame
from music_manager import get_music_manager

UI_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "assets", "audio", "ui")

class UIAudio:
    """Manages tactile sound effects for arcade menus, settings, and modal dialogues."""

    def __init__(self, audio_dir=None):
        self.audio_dir = audio_dir or UI_AUDIO_DIR
        self.sounds = {}
        self._load_sounds()

    def _load_sounds(self):
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            except Exception:
                return

        sound_mappings = {
            "click": ["click_001.wav"],
            "select": ["select_001.wav", "select_002.wav"],
            "toggle": ["toggle_001.wav", "toggle_002.wav"],
            "open": ["open_001.wav"],
            "close": ["close_001.wav"],
            "confirm": ["confirmation_001.wav", "confirmation_002.wav"],
            "launch": ["confirmation_002.wav"],
            "back": ["back_001.wav"],
            "error": ["error_001.wav"],
        }

        for action, filenames in sound_mappings.items():
            loaded = []
            for fname in filenames:
                fpath = os.path.join(self.audio_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        snd = pygame.mixer.Sound(fpath)
                        loaded.append(snd)
                    except Exception as e:
                        print(f"[UIAudio] Warning: Failed to load {fpath}: {e}")
            if loaded:
                self.sounds[action] = loaded

    def play(self, action):
        """Play a UI sound effect matching the given action name."""
        if not pygame.mixer.get_init():
            return

        music_mgr = get_music_manager()
        if music_mgr.is_muted:
            return

        sfx_vol = music_mgr.sfx_volume
        if sfx_vol <= 0.001:
            return

        samples = self.sounds.get(action)
        if not samples:
            return

        snd = random.choice(samples) if len(samples) > 1 else samples[0]
        snd.set_volume(sfx_vol)

        chan = pygame.mixer.find_channel()
        if chan:
            chan.play(snd)
        else:
            snd.play()

_global_ui_audio = None

def get_ui_audio():
    global _global_ui_audio
    if _global_ui_audio is None:
        _global_ui_audio = UIAudio()
    return _global_ui_audio
