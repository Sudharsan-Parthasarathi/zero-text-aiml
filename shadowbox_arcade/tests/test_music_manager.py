import os
import pygame
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from music_manager import MusicManager, TRACK_FILES

def test_music_manager_init_and_tracks():
    mgr = MusicManager(music_volume=0.5, sfx_volume=0.8, is_muted=False)
    assert mgr.music_volume == 0.5
    assert mgr.sfx_volume == 0.8
    assert not mgr.is_muted

    # Check track files existence
    for track_key in ["menu", "boxing", "fruit_ninja"]:
        assert track_key in TRACK_FILES
        assert os.path.exists(TRACK_FILES[track_key])

    # Volume and mute toggling
    mgr.set_music_volume(0.9)
    assert mgr.music_volume == 0.9
    mgr.toggle_mute()
    assert mgr.is_muted is True
    mgr.toggle_mute()
    assert mgr.is_muted is False

    # Safe play and stop
    mgr.play_track("menu", fade_ms=0)
    mgr.pause()
    mgr.unpause()
    mgr.stop(fade_ms=0)
    assert not mgr.is_playing
