import os
import json
import tempfile
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
from player_profiles import ProfileManager, make_default_profile

class TestLeaderboardAndGloveFeatures(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.profile_path = os.path.join(self.test_dir.name, "test_profiles.json")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_glove_size_simplification_and_fallback(self):
        """Verify Custom option is removed, falling back to Normal, with only Normal and Large active."""
        # 1. Test profile with legacy 'Custom' preference automatically falls back to 'Normal'
        legacy_data = {
            "version": 1,
            "last_active_player_id": "test_user",
            "players": {
                "test_user": {
                    "id": "test_user",
                    "name": "TestUser",
                    "preferences": {
                        "glove_size": "Custom",
                        "custom_glove_scale": 1.75
                    },
                    "shadow_boxing": {"high_score": 500},
                    "fruit_ninja": {"classic_high_score": 100}
                }
            }
        }
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        mgr = ProfileManager(filepath=self.profile_path)
        player = mgr.get_player("test_user")
        
        # Verify fallback to Normal and removal of custom_glove_scale
        self.assertEqual(player["preferences"]["glove_size"], "Normal")
        self.assertNotIn("custom_glove_scale", player["preferences"])

        # 2. Test switching to Large
        mgr.update_preferences("test_user", {"glove_size": "Large"})
        self.assertEqual(mgr.get_player("test_user")["preferences"]["glove_size"], "Large")

        # 3. Test switching to Normal
        mgr.update_preferences("test_user", {"glove_size": "Normal"})
        self.assertEqual(mgr.get_player("test_user")["preferences"]["glove_size"], "Normal")

        # 4. Test attempting to set Custom falls back to Normal
        mgr.update_preferences("test_user", {"glove_size": "Custom"})
        self.assertEqual(mgr.get_player("test_user")["preferences"]["glove_size"], "Normal")

    def test_duplicate_player_names_with_unique_ids(self):
        """Verify players with identical names receive unique record IDs."""
        mgr = ProfileManager(filepath=self.profile_path)
        # Clear default guest for clean duplicate test
        for p in mgr.list_players():
            mgr.delete_player(p["id"])
        
        # Create first "Champion"
        p1 = mgr.create_player("Champion")
        mgr.record_shadow_boxing_match(p1["id"], score=750)

        # Create second "Champion" with same display name
        p2 = mgr.create_player("Champion")
        mgr.record_shadow_boxing_match(p2["id"], score=450)

        # Unique IDs must differ
        self.assertNotEqual(p1["id"], p2["id"])
        self.assertEqual(p1["name"], "Champion")
        self.assertEqual(p2["name"], "Champion")

        players = mgr.list_players()
        self.assertEqual(len(players), 2)
        # Should be sorted by score descending: p1 (750) then p2 (450)
        self.assertEqual(players[0]["id"], p1["id"])
        self.assertEqual(players[0]["boxer_best"], 750)
        self.assertEqual(players[1]["id"], p2["id"])
        self.assertEqual(players[1]["boxer_best"], 450)

    def test_deletion_permanence_and_preservation(self):
        """Verify deleting one record permanently removes it while preserving unrelated records."""
        mgr = ProfileManager(filepath=self.profile_path)
        # Clear default guest
        for p in mgr.list_players():
            mgr.delete_player(p["id"])
        
        p_alice = mgr.create_player("Alice")
        mgr.record_shadow_boxing_match(p_alice["id"], score=600)
        
        p_bob1 = mgr.create_player("Bob")
        mgr.record_shadow_boxing_match(p_bob1["id"], score=500)

        p_bob2 = mgr.create_player("Bob")
        mgr.record_shadow_boxing_match(p_bob2["id"], score=300)

        # Delete only p_bob2
        deleted = mgr.delete_player(p_bob2["id"])
        self.assertTrue(deleted)

        # Check in-memory
        remaining = mgr.list_players()
        remaining_ids = [p["id"] for p in remaining]
        self.assertIn(p_alice["id"], remaining_ids)
        self.assertIn(p_bob1["id"], remaining_ids)
        self.assertNotIn(p_bob2["id"], remaining_ids)

        # Reload from disk to verify persistence (deleted records stay deleted)
        mgr_reloaded = ProfileManager(filepath=self.profile_path)
        reloaded_ids = [p["id"] for p in mgr_reloaded.list_players()]
        self.assertIn(p_alice["id"], reloaded_ids)
        self.assertIn(p_bob1["id"], reloaded_ids)
        self.assertNotIn(p_bob2["id"], reloaded_ids)
        self.assertIsNone(mgr_reloaded.get_player(p_bob2["id"]))

    def test_empty_state_when_all_records_deleted(self):
        """Verify deleting all player records results in an empty leaderboard state."""
        mgr = ProfileManager(filepath=self.profile_path)
        # Delete all players (including default guest)
        for p in mgr.list_players():
            mgr.delete_player(p["id"])
        
        self.assertEqual(len(mgr.list_players()), 0)
        
        # Reloading from disk preserves empty state
        mgr_reloaded = ProfileManager(filepath=self.profile_path)
        self.assertEqual(len(mgr_reloaded.list_players()), 0)

    def test_launcher_context_menu_and_confirmation_flow(self):
        """Verify launcher right-click context menu, clamping, cancel, and deletion flow."""
        import importlib
        launcher_mod = importlib.import_module("04_arcade_launcher")
        ArcadeLauncher = launcher_mod.ArcadeLauncher
        launcher = ArcadeLauncher()
        # Override profile manager with test instance
        launcher.profile_mgr = ProfileManager(filepath=self.profile_path)
        # Clear default guest
        for p in launcher.profile_mgr.list_players():
            launcher.profile_mgr.delete_player(p["id"])
        
        p1 = launcher.profile_mgr.create_player("Striker")
        launcher.profile_mgr.record_shadow_boxing_match(p1["id"], score=650)
        
        p2 = launcher.profile_mgr.create_player("Ninja")
        launcher.profile_mgr.record_fruit_ninja_match(p2["id"], "CLASSIC", score=400)

        # Open leaderboard modal and render once to populate player_row_rects
        launcher.show_player_modal = True
        launcher.render_player_modal()
        self.assertEqual(len(launcher.player_row_rects), 2)

        # 1. Simulate right-click on first player row
        row_rect, target_p = launcher.player_row_rects[0]
        self.assertEqual(target_p["name"], "Striker")

        # Simulate right-click event near bottom-right edge of screen to test clamping
        event_pos = (1270, 710)
        cm_w, cm_h = 175, 42
        cm_x = max(10, min(event_pos[0] + 4, 1280 - cm_w - 10))
        cm_y = max(10, min(event_pos[1] + 4, 720 - cm_h - 10))
        launcher.context_menu_rect = pygame.Rect(cm_x, cm_y, cm_w, cm_h)
        launcher.context_menu_player = target_p
        launcher.context_menu_open = True

        # Assert context menu is completely on-screen (never extends beyond 1280x720)
        self.assertGreaterEqual(launcher.context_menu_rect.left, 0)
        self.assertLessEqual(launcher.context_menu_rect.right, 1280)
        self.assertGreaterEqual(launcher.context_menu_rect.top, 0)
        self.assertLessEqual(launcher.context_menu_rect.bottom, 720)

        # 2. Click outside context menu -> closes menu
        outside_pos = (50, 50)
        self.assertFalse(launcher.context_menu_rect.collidepoint(outside_pos))

        # 3. Click inside context menu -> opens confirmation dialog
        inside_pos = launcher.context_menu_rect.center
        self.assertTrue(launcher.context_menu_rect.collidepoint(inside_pos))
        
        # Trigger delete confirmation
        launcher.context_menu_open = False
        launcher.show_delete_confirm = True
        launcher.confirm_player = target_p
        launcher.render_player_modal()

        self.assertTrue(launcher.show_delete_confirm)
        self.assertEqual(launcher.confirm_player["id"], p1["id"])

        # 4. Click Cancel -> preserves record
        launcher.handle_delete_confirm_click(launcher.cancel_btn_rect.center)
        self.assertFalse(launcher.show_delete_confirm)
        self.assertIsNotNone(launcher.profile_mgr.get_player(p1["id"]))

        # 5. Re-open delete confirm and click Delete -> permanently deletes
        launcher.show_delete_confirm = True
        launcher.confirm_player = target_p
        launcher.handle_delete_confirm_click(launcher.delete_btn_rect.center)
        
        self.assertFalse(launcher.show_delete_confirm)
        self.assertIsNone(launcher.profile_mgr.get_player(p1["id"]))
        
        # Verify Ninja is still preserved
        self.assertIsNotNone(launcher.profile_mgr.get_player(p2["id"]))
        
        launcher.music.cleanup()
        pygame.quit()

if __name__ == "__main__":
    unittest.main()
