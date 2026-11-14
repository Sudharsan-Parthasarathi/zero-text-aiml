"""
Tests for server.py Arcade endpoints:
/api/arcade/status, /api/arcade/profiles, /api/arcade/stop, /api/arcade/player/active
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from server import (
    get_arcade_status,
    get_arcade_profiles,
    stop_arcade,
    set_active_player,
    ArcadePlayerRequest,
    serve_index
)


class TestArcadeEndpoints(unittest.TestCase):
    def test_frontend_has_arcade_markup(self):
        """Verify the served HTML contains the arcade tab, unlock banner, and arcade bay."""
        html = serve_index()
        self.assertIn('id="tab-arcade"', html)
        self.assertIn('id="arcade-unlock-banner"', html)
        self.assertIn('id="view-arcade"', html)
        self.assertIn('SHADOWBOX ARCADE', html)

    def test_get_status_endpoint(self):
        """Verify GET /api/arcade/status."""
        res = get_arcade_status()
        self.assertIn("running", res)
        self.assertFalse(res["running"])
        self.assertTrue(res["arcade_installed"])

    def test_get_profiles_endpoint(self):
        """Verify GET /api/arcade/profiles."""
        res = get_arcade_profiles()
        self.assertIn("leaderboard", res)
        self.assertIn("players", res)
        self.assertIsInstance(res["leaderboard"], list)

    def test_stop_arcade_endpoint(self):
        """Verify POST /api/arcade/stop when not running."""
        res = stop_arcade()
        self.assertIn("status", res)
        self.assertEqual(res["status"], "not_running")

    def test_set_active_player_endpoint(self):
        """Verify POST /api/arcade/player/active."""
        profiles = get_arcade_profiles()
        players = list(profiles.get("players", {}).keys())
        if players:
            req = ArcadePlayerRequest(player_id=players[0])
            res = set_active_player(req)
            self.assertEqual(res["status"], "ok")


if __name__ == "__main__":
    unittest.main()
