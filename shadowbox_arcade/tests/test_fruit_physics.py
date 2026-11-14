import pytest
import math
from fruit_physics import (
    dist_point_to_segment,
    swept_slice_collision,
    FruitObject,
    SlicedHalf,
    JuiceParticle,
    WaveSpawner
)

def test_dist_point_to_segment():
    # Point perpendicular to midpoint of line (0, 0) to (10, 0)
    d, contact = dist_point_to_segment(5, 5, 0, 0, 10, 0)
    assert pytest.approx(d, 0.001) == 5.0
    assert pytest.approx(contact[0], 0.001) == 5.0
    assert pytest.approx(contact[1], 0.001) == 0.0

    # Point past the end of segment
    d, contact = dist_point_to_segment(15, 0, 0, 0, 10, 0)
    assert pytest.approx(d, 0.001) == 5.0
    assert pytest.approx(contact[0], 0.001) == 10.0

def test_swept_slice_tunneling_prevention():
    # Target circle at (500, 300) with radius 40
    # Swipe goes from (400, 300) to (600, 300) in a single fast frame
    # Neither end is inside the fruit, but the sweep passes directly through the center
    hit, contact = swept_slice_collision((400, 300), (600, 300), (500, 300), 40.0, blade_radius=10.0)
    assert hit is True
    assert pytest.approx(contact[0], 1.0) == 500.0

    # Swipe far away (100, 100) to (200, 100)
    hit_miss, _ = swept_slice_collision((100, 100), (200, 100), (500, 300), 40.0)
    assert hit_miss is False

def test_fruit_parabolic_physics():
    # Launch fruit upward
    fruit = FruitObject(obj_id=1, fruit_type="watermelon", x=640, y=750, vx=100.0, vy=-600.0, gravity=1000.0)
    assert fruit.y == 750.0
    assert fruit.vy == -600.0
    
    # After dt=0.1s
    # y += vy * dt = 750 + (-600 + 100) * 0.1 ...
    fruit.update(0.1)
    assert fruit.x == pytest.approx(650.0, 0.1)
    assert fruit.vy == pytest.approx(-500.0, 0.1)  # vy increased by g * dt
    assert fruit.y < 750.0  # Traveled upward!

def test_sliced_half_dynamics():
    half = SlicedHalf(fruit_type="orange", x=500, y=400, vx=-150.0, vy=-80.0, angle=0.0, angular_velocity=-120.0, side="left")
    half.update(0.1)
    assert half.x < 500.0  # Moves left
    assert half.angle != 0.0  # Spins
    assert not half.is_dead(720)

def test_wave_spawner_modes():
    spawner = WaveSpawner(screen_w=1280, screen_h=720)
    
    # Classic mode should spawn fruits, can have bombs
    spawner.time_to_next_wave = 0.0
    wave_classic = spawner.get_wave(mode="CLASSIC", dt=0.016)
    assert len(wave_classic) >= 1
    
    # Zen mode must NEVER spawn bombs
    spawner.time_to_next_wave = 0.0
    for _ in range(10):
        spawner.time_to_next_wave = 0.0
        wave_zen = spawner.get_wave(mode="ZEN", dt=0.016)
        for obj in wave_zen:
            assert obj.is_bomb is False
            assert obj.fruit_type != "bomb"
