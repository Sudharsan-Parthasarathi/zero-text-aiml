"""
=============================================================================
PLAYER PROFILES & ATOMIC RECORD PERSISTENCE (player_profiles.py)
Lightweight Local Profile Manager, Atomic JSON Storage, Legacy Migration,
and Per-Player Statistical Tracking with Leaderboard Record Deletion.
=============================================================================
"""

import os
import re
import json
import time
import tempfile
import shutil

PROFILES_FILE = os.path.join(os.path.dirname(__file__), "player_profiles.json")
PROFILES_BACKUP = os.path.join(os.path.dirname(__file__), "player_profiles.json.bak")
LEGACY_BOXER_FILE = os.path.join(os.path.dirname(__file__), "high_score.json")
LEGACY_FRUIT_FILE = os.path.join(os.path.dirname(__file__), "fruit_ninja_high_score.json")

DEFAULT_PREFERENCES = {
    "glove_size": "Normal",       # "Normal" (1.30x), "Large" (1.45x)
    "glove_skin": "Classic Red", # "Classic Red", "Electric Blue", "Neon Purple", "Golden"
    "blade_color": "Cyan/Amber", # "Cyan/Amber", "Neon Emerald", "Violet Blade", "Molten Gold"
    "music_volume": 0.70,
    "sfx_volume": 0.85,
    "master_mute": False,
    "mic_sensitivity": 1.0,
    "camera_blend": 0.35,
    "fullscreen": False
}

def sanitize_player_name(raw_name):
    """Sanitize player name: alphanumeric + spaces/dashes, max 18 chars."""
    if not raw_name:
        return "Guest"
    # Replace non-alphanumeric (allowing spaces, hyphens, underscores)
    cleaned = re.sub(r'[^A-Za-z0-9 _-]', '', str(raw_name)).strip()
    if not cleaned:
        cleaned = "Player"
    return cleaned[:18]

def normalize_profile_id(name):
    """Normalize name for key lookup to prevent duplicates by capitalization."""
    return re.sub(r'\s+', '_', name.lower().strip())

def make_default_profile(player_id, display_name):
    return {
        "id": player_id,
        "name": display_name,
        "created_at": time.time(),
        "preferences": dict(DEFAULT_PREFERENCES),
        "shadow_boxing": {
            "high_score": 0,
            "total_matches": 0,
            "max_combo": 0,
            "total_punches": 0,
            "kiai_strikes": 0,
            "bosses_defeated": 0,
            "dodges_completed": 0,
            "shields_deflected": 0,
            "best_session_score": 0
        },
        "fruit_ninja": {
            "classic_high_score": 0,
            "zen_high_score": 0,
            "arcade_high_score": 0,
            "total_sliced": 0,
            "max_combo": 0,
            "bombs_hit": 0,
            "total_matches": 0
        }
    }

class ProfileManager:
    def __init__(self, filepath=None):
        self.filepath = filepath or PROFILES_FILE
        self.backup_path = self.filepath + ".bak"
        self.data = {
            "version": 1,
            "last_active_player_id": "guest",
            "players": {}
        }
        self.load()

    def load(self):
        """Loads profiles from JSON, falling back to backup if corrupted."""
        loaded = False
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                    loaded = True
            except Exception as e:
                print(f"[ProfileManager Warning] Primary file corrupted ({e}). Attempting backup...")
                if os.path.exists(self.backup_path):
                    try:
                        with open(self.backup_path, 'r', encoding='utf-8') as f:
                            self.data = json.load(f)
                            loaded = True
                            print("[ProfileManager] Successfully recovered profiles from backup.")
                    except Exception as be:
                        print(f"[ProfileManager Error] Backup also failed: {be}")

        if not loaded or "players" not in self.data:
            self.data = {
                "version": 1,
                "last_active_player_id": "guest",
                "players": {
                    "guest": make_default_profile("guest", "Guest")
                }
            }
            # Migrate any legacy scores into guest profile
            self.migrate_legacy_scores()
            self.save()
        else:
            # Sanitize preferences on all existing players (remove Custom fallback to Normal)
            for pid, pdata in self.data["players"].items():
                pdata["id"] = pid
                prefs = pdata.setdefault("preferences", {})
                if prefs.get("glove_size") == "Custom":
                    prefs["glove_size"] = "Normal"
                elif prefs.get("glove_size") == "Extra Large":
                    prefs["glove_size"] = "Large"
                elif prefs.get("glove_size") not in ("Normal", "Large"):
                    prefs["glove_size"] = "Normal"
                prefs.pop("custom_glove_scale", None)

            # Ensure last_active_player_id points to valid player if players exist
            if self.data["players"]:
                cur = self.data.get("last_active_player_id")
                if not cur or cur not in self.data["players"]:
                    self.data["last_active_player_id"] = list(self.data["players"].keys())[0]
            else:
                self.data["last_active_player_id"] = None

    def save(self):
        """Safe atomic write: write to temp file, validate, replace original, and update backup."""
        try:
            target_dir = os.path.dirname(os.path.abspath(self.filepath))
            os.makedirs(target_dir, exist_ok=True)
            
            # 1. Write to temporary file in the same directory
            with tempfile.NamedTemporaryFile('w', dir=target_dir, delete=False, encoding='utf-8') as tf:
                json.dump(self.data, tf, indent=2, ensure_ascii=False)
                temp_name = tf.name

            # 2. Validate by reading back
            with open(temp_name, 'r', encoding='utf-8') as tf:
                check = json.load(tf)
                if "players" not in check:
                    raise ValueError("Validation failed: missing players key")

            # 3. Create or update backup of existing file
            if os.path.exists(self.filepath):
                try:
                    shutil.copyfile(self.filepath, self.backup_path)
                except Exception:
                    pass

            # 4. Atomic replacement
            os.replace(temp_name, self.filepath)
            return True
        except Exception as e:
            print(f"[ProfileManager Error] Atomic save failed: {e}")
            return False

    def migrate_legacy_scores(self):
        """Safely import existing records from high_score.json and fruit_ninja_high_score.json."""
        guest = self.data["players"].get("guest")
        if not guest:
            return

        migrated = False
        # 1. Legacy Shadow Boxer high score
        if os.path.exists(LEGACY_BOXER_FILE):
            try:
                with open(LEGACY_BOXER_FILE, 'r', encoding='utf-8') as f:
                    b_data = json.load(f)
                    b_score = b_data.get("high_score", 0)
                    if b_score > guest["shadow_boxing"]["high_score"]:
                        guest["shadow_boxing"]["high_score"] = b_score
                        migrated = True
            except Exception:
                pass

        # 2. Legacy Fruit Ninja high scores
        if os.path.exists(LEGACY_FRUIT_FILE):
            try:
                with open(LEGACY_FRUIT_FILE, 'r', encoding='utf-8') as f:
                    f_data = json.load(f)
                    for mode, key in [("CLASSIC", "classic_high_score"),
                                      ("ZEN", "zen_high_score"),
                                      ("ARCADE", "arcade_high_score")]:
                        s = f_data.get(mode, 0)
                        if s > guest["fruit_ninja"][key]:
                            guest["fruit_ninja"][key] = s
                            migrated = True
                    tot = f_data.get("total_slices", 0)
                    if tot > guest["fruit_ninja"]["total_sliced"]:
                        guest["fruit_ninja"]["total_sliced"] = tot
                        migrated = True
            except Exception:
                pass

        if migrated:
            self.save()

    def list_players(self):
        """Returns list of player record dicts sorted by highest score descending for the leaderboard."""
        res = []
        for pid, pdata in self.data.get("players", {}).items():
            b_best = pdata.get("shadow_boxing", {}).get("high_score", 0)
            fn = pdata.get("fruit_ninja", {})
            f_best = max(fn.get("classic_high_score", 0), fn.get("arcade_high_score", 0), fn.get("zen_high_score", 0))
            best_combined = max(b_best, f_best)
            res.append({
                "id": pid,
                "name": pdata.get("name", pid),
                "boxer_best": b_best,
                "fruit_best": f_best,
                "high_score": best_combined,
                "created_at": pdata.get("created_at", 0)
            })
        return sorted(res, key=lambda x: (x["high_score"], x["boxer_best"]), reverse=True)

    def generate_unique_player_id(self, display_name):
        """Generate a guaranteed unique record ID, handling duplicate player names cleanly."""
        base_id = normalize_profile_id(display_name)
        if not base_id:
            base_id = "player"
        if base_id not in self.data.get("players", {}):
            return base_id
        suffix = 2
        while f"{base_id}_{suffix}" in self.data["players"]:
            suffix += 1
        return f"{base_id}_{suffix}"

    def get_or_create_player(self, raw_name):
        display_name = sanitize_player_name(raw_name)
        pid = normalize_profile_id(display_name)
        if pid not in self.data["players"]:
            self.data["players"][pid] = make_default_profile(pid, display_name)
            self.save()
        self.set_active_player(pid)
        return self.data["players"][pid]

    def create_player(self, raw_name):
        """Explicitly create a new player entry with a unique ID even if the name duplicates an existing one."""
        display_name = sanitize_player_name(raw_name)
        pid = self.generate_unique_player_id(display_name)
        self.data["players"][pid] = make_default_profile(pid, display_name)
        self.set_active_player(pid)
        self.save()
        return self.data["players"][pid]

    def delete_player(self, player_id):
        """Permanently remove a player record and their scores from persistent storage by unique ID."""
        if player_id in self.data.get("players", {}):
            del self.data["players"][player_id]

            # If deleted player was currently active, switch to next available player
            if self.data.get("last_active_player_id") == player_id:
                remaining = list(self.data["players"].keys())
                self.data["last_active_player_id"] = remaining[0] if remaining else None

            self.save()
            return True
        return False

    def set_active_player(self, player_id):
        if player_id in self.data["players"]:
            self.data["last_active_player_id"] = player_id
            self.save()
            return True
        return False

    def get_player(self, player_id):
        return self.data["players"].get(player_id)

    def get_active_player(self):
        pid = self.data.get("last_active_player_id")
        if not pid or pid not in self.data.get("players", {}):
            if self.data.get("players"):
                pid = list(self.data["players"].keys())[0]
                self.data["last_active_player_id"] = pid
            else:
                # Empty state: transient guest fallback profile without saving
                return make_default_profile("guest", "Guest")
        return self.data["players"][pid]

    def record_shadow_boxing_match(self, player_id, score, combo=0, punches=0, kiai=0, bosses=0, dodges=0, shields=0):
        if player_id not in self.data["players"]:
            if self.data["players"]:
                player_id = list(self.data["players"].keys())[0]
            else:
                player_id = "guest"
                self.data["players"]["guest"] = make_default_profile("guest", "Guest")
        p = self.data["players"][player_id]["shadow_boxing"]
        p["total_matches"] += 1
        p["total_punches"] += punches
        p["kiai_strikes"] += kiai
        p["bosses_defeated"] += bosses
        p["dodges_completed"] += dodges
        p["shields_deflected"] += shields
        if combo > p["max_combo"]:
            p["max_combo"] = combo
        is_new_best = False
        if score > p["high_score"]:
            p["high_score"] = score
            is_new_best = True
        p["best_session_score"] = max(p["best_session_score"], score)
        self.save()
        return is_new_best, p["high_score"]

    def record_fruit_ninja_match(self, player_id, mode, score, sliced=0, combo=0, bombs=0):
        if player_id not in self.data["players"]:
            if self.data["players"]:
                player_id = list(self.data["players"].keys())[0]
            else:
                player_id = "guest"
                self.data["players"]["guest"] = make_default_profile("guest", "Guest")
        p = self.data["players"][player_id]["fruit_ninja"]
        p["total_matches"] += 1
        p["total_sliced"] += sliced
        p["bombs_hit"] += bombs
        if combo > p["max_combo"]:
            p["max_combo"] = combo
            
        key = f"{mode.lower()}_high_score"
        is_new_best = False
        if key in p:
            if score > p[key]:
                p[key] = score
                is_new_best = True
        self.save()
        return is_new_best, p.get(key, score)

    def update_preferences(self, player_id, prefs):
        if player_id not in self.data["players"]:
            return
        # Sanitize glove size preference (remove Custom, fallback to Normal)
        if prefs.get("glove_size") == "Custom":
            prefs["glove_size"] = "Normal"
        elif prefs.get("glove_size") == "Extra Large":
            prefs["glove_size"] = "Large"
        prefs.pop("custom_glove_scale", None)
        self.data["players"][player_id]["preferences"].update(prefs)
        self.save()

_profile_manager = None
def get_profile_manager():
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager
