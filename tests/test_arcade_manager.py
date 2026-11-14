"""
Tests for arcade_manager.py:
Lifecycle, concurrency locking, profile I/O, and leaderboard generation.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from arcade_manager import arcade_manager, check_missing_dependencies, PROFILES_PATH, ARCADE_DIR


class TestArcadeManager(unittest.TestCase):
    def test_arcade_directory_exists(self):
        """Verify the shadowbox_arcade directory is present and populated."""
        self.assertTrue(ARCADE_DIR.exists(), "shadowbox_arcade directory must exist")
        self.assertTrue((ARCADE_DIR / "04_arcade_launcher.py").exists(), "04_arcade_launcher.py must exist")
        self.assertTrue((ARCADE_DIR / "02_shadow_boxer_game.py").exists(), "02_shadow_boxer_game.py must exist")
        self.assertTrue((ARCADE_DIR / "03_fruit_ninja_game.py").exists(), "03_fruit_ninja_game.py must exist")
        self.assertTrue((ARCADE_DIR / "hand_landmarker.task").exists(), "hand_landmarker.task must exist")
        self.assertTrue((ARCADE_DIR / "pose_landmarker_lite.task").exists(), "pose_landmarker_lite.task must exist")

    def test_status_reporting(self):
        """Verify get_status returns the required structure."""
        status = arcade_manager.get_status()
        self.assertIn("running", status)
        self.assertIn("game", status)
        self.assertIn("pid", status)
        self.assertIn("arcade_installed", status)
        self.assertIn("missing_dependencies", status)
        self.assertIn("active_player", status)
        self.assertTrue(status["arcade_installed"])
        self.assertIsInstance(status["missing_dependencies"], list)

    def test_profiles_and_leaderboard(self):
        """Verify player_profiles.json is parsed into a sorted leaderboard."""
        data = arcade_manager.get_profiles_and_leaderboard()
        self.assertIn("leaderboard", data)
        self.assertIn("players", data)
        leaderboard = data["leaderboard"]
        self.assertGreaterEqual(len(leaderboard), 1, "Leaderboard should have at least 1 player")

        # Verify sorting by best_score descending
        scores = [p["best_score"] for p in leaderboard]
        self.assertEqual(scores, sorted(scores, reverse=True), "Leaderboard must be sorted descending by best_score")

    def test_active_player_switch(self):
        """Verify changing active player updates and persists."""
        data = arcade_manager.get_profiles_and_leaderboard()
        players = list(data.get("players", {}).keys())
        if len(players) > 1:
            target = players[1]
            res = arcade_manager.set_active_player(target)
            self.assertEqual(res.get("status"), "ok")
            updated = arcade_manager.get_status()
            self.assertEqual(updated["active_player"]["id"], target)
            # Switch back to first player
            arcade_manager.set_active_player(players[0])

    def test_stop_when_not_running(self):
        """Verify stopping when no process is active returns clean response."""
        res = arcade_manager.stop()
        self.assertEqual(res.get("status"), "not_running")


if __name__ == "__main__":
    unittest.main()
