import os
import sys
import time
import pytest

# Use dummy video and audio drivers for headless test
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from fruit_ninja_game import FruitNinjaGame

def test_fruit_ninja_game_headless_lifecycle():
    # Instantiate game
    game = FruitNinjaGame(use_fullscreen=False)
    assert game.state == "MENU"
    
    # Start classic game
    game.start_game("CLASSIC")
    assert game.state == "COUNTDOWN"
    assert game.mode == "CLASSIC"
    assert game.lives == 3
    assert game.score == 0

    # Advance countdown
    game.countdown_start = time.time() - 4.0
    game.update_countdown(0.016)
    assert game.state == "PLAYING"

    # Spawn a wave manually
    game.spawner.time_to_next_wave = 0.0
    wave = game.spawner.get_wave(mode="CLASSIC", dt=0.016)
    game.active_fruits.extend(wave)
    assert len(game.active_fruits) > 0

    # Simulate slicing the first fruit
    target_fruit = game.active_fruits[0]
    initial_score = game.score
    game.slice_fruit(target_fruit)
    assert target_fruit.is_sliced is True
    assert game.total_sliced == 1
    assert game.score > initial_score
    assert len(game.sliced_halves) == 2
    assert len(game.particles) > 0

    # Simulate bomb hit in classic mode
    from fruit_physics import FruitObject
    bomb = FruitObject(999, "bomb", 500, 500, 0, 0, is_bomb=True)
    game.active_fruits.append(bomb)
    game.slice_fruit(bomb)
    assert game.bombs_hit == 1
    assert game.lives == 0
    assert game.state == "GAMEOVER"
    assert game.flash_alpha == 255.0
    assert game.screen_shake_time > 0

    # Verify that bomb flash and screen shake fade out during GAMEOVER
    for _ in range(15):
        game.update_gameover([], 0.05)
    assert game.flash_alpha == 0.0
    assert game.screen_shake_time == 0.0

    # Test rendering in GAMEOVER state after flash fades
    game.render(None)

    # Test game over and high scores
    assert game.high_scores["CLASSIC"] >= 0

    # Test restart resets screen effects
    game.start_game("CLASSIC")
    assert game.flash_alpha == 0.0
    assert game.screen_shake_time == 0.0

    # Cleanup
    game.release_camera()

def test_fruit_ninja_sword_size_settings():
    game = FruitNinjaGame(use_fullscreen=False)
    initial_preset = game.sword_size_preset
    
    # Cycle through presets
    game.cycle_sword_size()
    assert game.sword_size_preset != initial_preset
    assert game.sword_length_mult > 0
    assert game.sword_width_mult > 0

    if game.tracker:
        assert game.tracker.right_sword.length_multiplier == game.sword_length_mult
        assert game.tracker.right_sword.width_multiplier == game.sword_width_mult

    game.release_camera()

def test_fruit_ninja_hand_mode_settings():
    game = FruitNinjaGame(use_fullscreen=False)
    
    # Test set_hand_mode
    game.set_hand_mode("RIGHT_ONLY")
    assert game.hand_mode == "RIGHT_ONLY"
    if game.tracker:
        assert game.tracker.hand_mode == "RIGHT_ONLY"

    game.set_hand_mode("LEFT_ONLY")
    assert game.hand_mode == "LEFT_ONLY"
    if game.tracker:
        assert game.tracker.hand_mode == "LEFT_ONLY"

    # Test cycle_hand_mode
    game.cycle_hand_mode()
    assert game.hand_mode in ["RIGHT_ONLY", "LEFT_ONLY", "SINGLE_AUTO", "BOTH"]

    game.release_camera()

def test_fruit_ninja_menu_and_tutorial_buttons():
    game = FruitNinjaGame(use_fullscreen=False)
    
    # Validate bottom menu button hitboxes exist and are within screen bounds
    assert 0 <= game.btn_diff_rect.left < game.screen_w
    assert 0 <= game.btn_sword_size_rect.left < game.screen_w
    assert 0 <= game.btn_hand_mode_rect.left < game.screen_w
    assert 0 <= game.btn_autogrip_rect.left < game.screen_w
    assert 0 <= game.btn_tutorial_rect.left < game.screen_w
    assert 0 <= game.btn_quit_rect.right <= game.screen_w

    # Validate tutorial hand selection buttons exist
    assert game.tut_hand_right_rect.width > 0
    assert game.tut_hand_left_rect.width > 0
    assert game.tut_hand_dual_rect.width > 0

    # Test render in MENU and TUTORIAL states
    game.state = "MENU"
    game.render(None)

    game.state = "TUTORIAL"
    game.render(None)

    game.release_camera()
