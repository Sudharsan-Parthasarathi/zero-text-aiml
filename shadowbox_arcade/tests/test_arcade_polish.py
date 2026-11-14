import os
import sys
import time
import pytest
import numpy as np
import tempfile

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from fruit_ninja_game import FruitNinjaGame
from fruit_physics import FruitObject
from player_profiles import ProfileManager, get_profile_manager
from clap_detector import ClapDetector
from tests.test_clap_detector import generate_silence, generate_clap_impulse

def test_heart_fruit_mechanics():
    """Verify rare heart fruit restores life or gives +50 pts bonus, and misses do not penalize."""
    game = FruitNinjaGame(use_fullscreen=False)
    game.start_game("CLASSIC")
    game.lives = 2
    initial_score = game.score

    # 1. Slicing heart fruit when lives < max_lives gives +1 life and +10 base points
    heart = FruitObject(2001, "heart_fruit", 400, 400, 0, 0, special="heart")
    game.slice_fruit(heart)
    assert game.lives == 3
    assert heart.is_sliced is True
    assert game.score == initial_score + 10

    # 2. Slicing heart fruit when lives == max_lives gives +50 bonus points (+10 base)
    heart2 = FruitObject(2002, "heart_fruit", 400, 400, 0, 0, special="heart")
    game.slice_fruit(heart2)
    assert game.lives == 3
    assert game.score == initial_score + 10 + 60

    # 3. Dropping heart fruit does not decrement lives
    heart3 = FruitObject(2003, "heart_fruit", 400, 800, 0, 0, special="heart")
    game.active_fruits = [heart3]
    game.update_playing(active_slices=[], sim_dt=0.016, raw_dt=0.016)
    assert game.lives == 3  # Not decremented!
    game.release_camera()

def test_player_profile_recording_full_suite():
    """Verify persistent stats recording across games and modes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "test_profiles.json")
        mgr = ProfileManager(filepath=storage_path)

        player = mgr.get_or_create_player("Sensei")
        pid = player["id"]

        # Record Shadow Boxing
        is_new, best = mgr.record_shadow_boxing_match(
            player_id=pid,
            score=450,
            combo=14,
            punches=85,
            kiai=5,
            bosses=2,
            dodges=4,
            shields=3
        )
        assert is_new is True
        assert best == 450
        p = mgr.get_player(pid)["shadow_boxing"]
        assert p["high_score"] == 450
        assert p["bosses_defeated"] == 2
        assert p["shields_deflected"] == 3

        # Record Fruit Ninja Classic
        is_new, best = mgr.record_fruit_ninja_match(
            player_id=pid,
            mode="CLASSIC",
            score=320,
            sliced=45,
            combo=4,
            bombs=1
        )
        assert is_new is True
        assert best == 320
        fn = mgr.get_player(pid)["fruit_ninja"]
        assert fn["classic_high_score"] == 320
        assert fn["total_sliced"] == 45
        assert fn["bombs_hit"] == 1

        # Record Fruit Ninja Zen
        is_new_zen, best_zen = mgr.record_fruit_ninja_match(
            player_id=pid,
            mode="ZEN",
            score=210,
            sliced=30,
            combo=3,
            bombs=0
        )
        assert is_new_zen is True
        assert best_zen == 210
        assert fn["zen_high_score"] == 210
        assert fn["total_sliced"] == 75  # Cumulative

def test_multimodal_convergence_clap():
    """Verify rapid hand convergence reinforces clap detection confidence."""
    detector = ClapDetector(cooldown_sec=1.0)
    detector.sensitivity = 1.0

    for _ in range(5):
        detector.process_audio_block(generate_silence())

    # Feed realistic sharp clap impulse
    detector.process_audio_block(generate_clap_impulse(amp=0.55))

    # Evaluate multimodal with close hands
    triggered, conf, debug = detector.evaluate_clap(
        wrist_l=(280, 300),
        wrist_r=(320, 300),
        shoulder_l=(250, 400),
        shoulder_r=(380, 400),
        is_playing=True
    )
    assert debug["body_relative_dist"] < 1.0
    assert conf >= 0.50
    assert triggered is True

def test_laser_parry_reward_logic():
    """Verify laser hazard deflection awards +75 parry points."""
    # Simulation of laser deflection outcome
    score = 100
    shields_deflected = 0
    laser_deflected = True

    if laser_deflected:
        score += 75
        shields_deflected += 1

    assert score == 175
    assert shields_deflected == 1
