"""
=============================================================================
CENTRALIZED ARCADE MUSIC & AUDIO MANAGER (music_manager.py)
Unified Background Music & SFX Volume Coordinator with Seamless Looping,
Fade Transitions, and Clean Cross-Game Process Handoff.
=============================================================================
"""

import os
import pygame

MUSIC_DIR = os.path.join(os.path.dirname(__file__), "assets", "music")

TRACK_FILES = {
    "menu": os.path.join(MUSIC_DIR, "menu_theme.wav"),
    "boxing": os.path.join(MUSIC_DIR, "boxing_theme.wav"),
    "fruit_ninja": os.path.join(MUSIC_DIR, "fruit_ninja_theme.wav")
}

class MusicManager:
    def __init__(self, music_volume=0.70, sfx_volume=0.85, is_muted=False):
        self.music_volume = max(0.0, min(1.0, float(music_volume)))
        self.sfx_volume = max(0.0, min(1.0, float(sfx_volume)))
        self.is_muted = bool(is_muted)
        self.current_track = None
        self.is_playing = False
        self._ensure_mixer_init()
        self._apply_volume()

    def _ensure_mixer_init(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(16)
        except Exception as e:
            print(f"[MusicManager Warning] Mixer init exception: {e}")

    def _apply_volume(self):
        if not pygame.mixer.get_init():
            return
        effective_vol = 0.0 if self.is_muted else self.music_volume
        try:
            pygame.mixer.music.set_volume(effective_vol)
        except Exception:
            pass

    def play_track(self, track_name, loop=-1, fade_ms=600):
        """Starts playing background music loop with smooth fade-in."""
        if not pygame.mixer.get_init():
            return
        if track_name not in TRACK_FILES:
            print(f"[MusicManager] Unknown track: {track_name}")
            return
        path = TRACK_FILES[track_name]
        if not os.path.exists(path):
            print(f"[MusicManager Warning] Track file not found: {path}")
            return

        if self.current_track == track_name and self.is_playing:
            return  # Already playing this track

        try:
            self._apply_volume()
            pygame.mixer.music.load(path)
            if fade_ms > 0:
                pygame.mixer.music.play(loops=loop, fade_ms=fade_ms)
            else:
                pygame.mixer.music.play(loops=loop)
            self.current_track = track_name
            self.is_playing = True
        except Exception as e:
            print(f"[MusicManager Error] Could not play track {track_name}: {e}")

    def play_menu_theme(self, loop=-1, fade_ms=600):
        self.play_track("menu", loop=loop, fade_ms=fade_ms)

    def play_boxing_theme(self, loop=-1, fade_ms=600):
        self.play_track("boxing", loop=loop, fade_ms=fade_ms)

    def play_fruit_ninja_theme(self, loop=-1, fade_ms=600):
        self.play_track("fruit_ninja", loop=loop, fade_ms=fade_ms)

    def stop(self, fade_ms=400):
        """Stops current background music with optional fade-out."""
        if not pygame.mixer.get_init():
            return
        try:
            if fade_ms > 0:
                pygame.mixer.music.fadeout(fade_ms)
            else:
                pygame.mixer.music.stop()
            self.current_track = None
            self.is_playing = False
        except Exception:
            pass

    def pause(self):
        if pygame.mixer.get_init() and self.is_playing:
            try:
                pygame.mixer.music.pause()
            except Exception:
                pass

    def unpause(self):
        if pygame.mixer.get_init() and self.is_playing:
            try:
                pygame.mixer.music.unpause()
            except Exception:
                pass

    def set_music_volume(self, vol):
        self.music_volume = max(0.0, min(1.0, float(vol)))
        self._apply_volume()

    def set_sfx_volume(self, vol):
        self.sfx_volume = max(0.0, min(1.0, float(vol)))

    def toggle_mute(self):
        self.is_muted = not self.is_muted
        self._apply_volume()
        return self.is_muted

    def set_mute(self, muted):
        self.is_muted = bool(muted)
        self._apply_volume()

    def cleanup(self):
        """Release audio music resources completely."""
        self.stop(fade_ms=0)

_music_mgr = None
def get_music_manager(prefs=None):
    global _music_mgr
    if _music_mgr is None:
        mv = 0.70
        sv = 0.85
        mute = False
        if prefs:
            mv = prefs.get("music_volume", mv)
            sv = prefs.get("sfx_volume", sv)
            mute = prefs.get("master_mute", mute)
        _music_mgr = MusicManager(music_volume=mv, sfx_volume=sv, is_muted=mute)
    return _music_mgr
