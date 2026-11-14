"""
=============================================================================
ARCADE MANAGER (arcade_manager.py)
Direct native subprocess management, hardware lock guard, and player profile
synchronization for SHADOWBOX ARCADE within ZERO-TEXT.
=============================================================================
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List

BASE_DIR = Path(__file__).parent.resolve()
ARCADE_DIR = BASE_DIR / "shadowbox_arcade"
PROFILES_PATH = ARCADE_DIR / "player_profiles.json"


def check_missing_dependencies() -> List[str]:
    """Check whether arcade native dependencies are installed in the active environment."""
    missing = []
    for pkg in ["pygame", "mediapipe", "sounddevice"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


class ArcadeManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.current_game: Optional[str] = None
        self.started_at: Optional[float] = None

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            if self.process is not None:
                ret = self.process.poll()
                if ret is not None:
                    # Child process exited
                    self.process = None
                    self.current_game = None
                    self.started_at = None

            is_running = self.process is not None
            active_player = self._get_active_player_summary()
            missing_deps = check_missing_dependencies()

            return {
                "running": is_running,
                "game": self.current_game,
                "pid": self.process.pid if is_running else None,
                "started_at": self.started_at,
                "arcade_installed": ARCADE_DIR.exists(),
                "missing_dependencies": missing_deps,
                "active_player": active_player,
            }

    def launch(self, game: str = "launcher", player_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            # Check if an arcade session is already active
            if self.process is not None and self.process.poll() is None:
                return {
                    "status": "already_running",
                    "message": f"Shadowbox Arcade is already running ({self.current_game})",
                    "pid": self.process.pid,
                    "game": self.current_game
                }

            if not ARCADE_DIR.exists():
                raise FileNotFoundError(f"Arcade directory not found: {ARCADE_DIR}")

            script_map = {
                "launcher": "04_arcade_launcher.py",
                "boxing": "02_shadow_boxer_game.py",
                "fruit_ninja": "03_fruit_ninja_game.py",
            }

            script_name = script_map.get(game, "04_arcade_launcher.py")
            script_path = ARCADE_DIR / script_name
            if not script_path.exists():
                raise FileNotFoundError(f"Game script not found: {script_path}")

            # Resolve player ID from profile if not explicitly passed
            if not player_id:
                active = self._get_active_player_summary()
                if active and active.get("id"):
                    player_id = active.get("id")

            cmd = [sys.executable, str(script_path)]
            if player_id and game != "launcher":
                cmd.extend(["--player-id", str(player_id)])

            # Spawn native Pygame process in arcade directory
            self.process = subprocess.Popen(cmd, cwd=str(ARCADE_DIR))
            self.current_game = game
            self.started_at = time.time()

            return {
                "status": "launched",
                "game": game,
                "pid": self.process.pid,
                "player_id": player_id
            }

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                self.process = None
                self.current_game = None
                self.started_at = None
                return {"status": "not_running", "message": "No arcade process is active"}

            prev_game = self.current_game
            prev_pid = self.process.pid
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
            except Exception as e:
                try:
                    self.process.kill()
                except Exception:
                    pass

            self.process = None
            self.current_game = None
            self.started_at = None

            return {
                "status": "stopped",
                "message": f"Terminated {prev_game} (PID {prev_pid})"
            }

    def get_profiles_and_leaderboard(self) -> Dict[str, Any]:
        if not PROFILES_PATH.exists():
            return {
                "active_player_id": None,
                "players": {},
                "leaderboard": []
            }
        try:
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            players_map = data.get("players", {})
            active_id = data.get("last_active_player_id")

            # Generate formatted leaderboard sorted by high score descending
            leaderboard = []
            for pid, pdata in players_map.items():
                boxer_stats = pdata.get("shadow_boxing", {})
                fn_stats = pdata.get("fruit_ninja", {})
                b_high = boxer_stats.get("high_score", 0)
                f_high = max(
                    fn_stats.get("classic_high_score", 0),
                    fn_stats.get("arcade_high_score", 0),
                    fn_stats.get("zen_high_score", 0),
                )
                leaderboard.append({
                    "id": pid,
                    "name": pdata.get("name", pid),
                    "is_active": pid == active_id,
                    "best_score": max(b_high, f_high),
                    "boxer_high": b_high,
                    "boxer_matches": boxer_stats.get("total_matches", 0),
                    "boxer_punches": boxer_stats.get("total_punches", 0),
                    "boxer_combo": boxer_stats.get("max_combo", 0),
                    "boxer_bosses": boxer_stats.get("bosses_defeated", 0),
                    "fruit_high": f_high,
                    "fruit_sliced": fn_stats.get("total_sliced", 0),
                    "fruit_combo": fn_stats.get("max_combo", 0),
                    "fruit_matches": fn_stats.get("total_matches", 0),
                })

            leaderboard.sort(key=lambda x: (x["best_score"], x["boxer_high"]), reverse=True)

            return {
                "active_player_id": active_id,
                "players": players_map,
                "leaderboard": leaderboard
            }
        except Exception as e:
            return {
                "error": str(e),
                "active_player_id": None,
                "players": {},
                "leaderboard": []
            }

    def set_active_player(self, player_id: str) -> Dict[str, Any]:
        if not PROFILES_PATH.exists():
            return {"status": "error", "message": "Profiles file not found"}
        try:
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if player_id in data.get("players", {}):
                data["last_active_player_id"] = player_id
                # Safe atomic write
                target_dir = PROFILES_PATH.parent
                with tempfile.NamedTemporaryFile("w", dir=str(target_dir), delete=False, encoding="utf-8") as tf:
                    json.dump(data, tf, indent=2, ensure_ascii=False)
                    temp_name = tf.name
                os.replace(temp_name, str(PROFILES_PATH))
                return {"status": "ok", "active_player_id": player_id}
            else:
                return {"status": "error", "message": f"Player '{player_id}' not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_active_player_summary(self) -> Optional[Dict[str, Any]]:
        if not PROFILES_PATH.exists():
            return None
        try:
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            active_id = data.get("last_active_player_id")
            players = data.get("players", {})
            if active_id and active_id in players:
                p = players[active_id]
                return {
                    "id": active_id,
                    "name": p.get("name", active_id),
                    "preferences": p.get("preferences", {}),
                    "boxer_high": p.get("shadow_boxing", {}).get("high_score", 0),
                    "fruit_high": max(
                        p.get("fruit_ninja", {}).get("classic_high_score", 0),
                        p.get("fruit_ninja", {}).get("arcade_high_score", 0)
                    )
                }
            elif players:
                first_id = list(players.keys())[0]
                p = players[first_id]
                return {
                    "id": first_id,
                    "name": p.get("name", first_id),
                    "preferences": p.get("preferences", {}),
                    "boxer_high": p.get("shadow_boxing", {}).get("high_score", 0),
                    "fruit_high": 0
                }
        except Exception:
            pass
        return None


arcade_manager = ArcadeManager()
