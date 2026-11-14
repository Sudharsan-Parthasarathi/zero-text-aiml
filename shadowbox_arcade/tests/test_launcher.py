import os
import pygame
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import importlib.util

def test_arcade_launcher_init():
    spec = importlib.util.spec_from_file_location("arcade_launcher", "04_arcade_launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    launcher = mod.ArcadeLauncher()
    assert launcher.card_boxer is not None
    assert launcher.card_fruit is not None
    assert os.path.exists(mod.BOXER_SCRIPT)
    assert os.path.exists(mod.FRUIT_SCRIPT)
    pygame.quit()
