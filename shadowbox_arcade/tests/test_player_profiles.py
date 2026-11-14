import os
import json
import tempfile
import pytest
from player_profiles import (
    ProfileManager,
    sanitize_player_name,
    normalize_profile_id
)

def test_sanitize_and_normalize_name():
    assert sanitize_player_name("  CoolNinja123!@#  ") == "CoolNinja123"
    assert sanitize_player_name("A Very Long Name That Exceeds Eighteen Characters") == "A Very Long Name T"
    assert sanitize_player_name("") == "Guest"
    assert normalize_profile_id("Cool Ninja") == "cool_ninja"
    assert normalize_profile_id("cool  ninja") == "cool_ninja"

def test_profile_creation_and_atomic_save(tmp_path):
    pfile = str(tmp_path / "profiles.json")
    mgr = ProfileManager(filepath=pfile)
    
    # Create player
    p1 = mgr.get_or_create_player("Alice")
    assert p1["name"] == "Alice"
    assert p1["id"] == "alice"
    assert mgr.get_active_player()["id"] == "alice"

    # Case-insensitive duplicate prevention
    p1_dup = mgr.get_or_create_player("ALICE")
    assert p1_dup["id"] == "alice"

    # Atomic write verification
    assert os.path.exists(pfile)
    with open(pfile, "r") as f:
        data = json.load(f)
    assert "alice" in data["players"]

def test_per_player_isolation(tmp_path):
    pfile = str(tmp_path / "profiles.json")
    mgr = ProfileManager(filepath=pfile)

    p_bob = mgr.get_or_create_player("Bob")
    p_charlie = mgr.get_or_create_player("Charlie")

    # Record match for Bob
    is_new, best = mgr.record_shadow_boxing_match("bob", score=350, combo=5, punches=40)
    assert is_new is True
    assert best == 350

    # Charlie's record should remain 0
    charlie_boxer = mgr.data["players"]["charlie"]["shadow_boxing"]
    assert charlie_boxer["high_score"] == 0
    assert charlie_boxer["total_matches"] == 0

    # Record Fruit Ninja match for Charlie
    is_new_fn, best_fn = mgr.record_fruit_ninja_match("charlie", "CLASSIC", score=520, sliced=45)
    assert is_new_fn is True
    assert best_fn == 520

    # Bob's fruit ninja remains 0
    bob_fn = mgr.data["players"]["bob"]["fruit_ninja"]
    assert bob_fn["classic_high_score"] == 0

def test_legacy_migration(tmp_path):
    # Set up legacy score files
    legacy_boxer = str(tmp_path / "legacy_boxer.json")
    legacy_fruit = str(tmp_path / "legacy_fruit.json")
    with open(legacy_boxer, "w") as f:
        json.dump({"high_score": 750, "version": 2}, f)
    with open(legacy_fruit, "w") as f:
        json.dump({"CLASSIC": 880, "total_slices": 120}, f)

    # Monkeypatch legacy file paths in player_profiles module
    import player_profiles
    orig_b = player_profiles.LEGACY_BOXER_FILE
    orig_f = player_profiles.LEGACY_FRUIT_FILE
    player_profiles.LEGACY_BOXER_FILE = legacy_boxer
    player_profiles.LEGACY_FRUIT_FILE = legacy_fruit

    try:
        pfile = str(tmp_path / "profiles_migrated.json")
        mgr = ProfileManager(filepath=pfile)
        guest = mgr.data["players"]["guest"]
        assert guest["shadow_boxing"]["high_score"] == 750
        assert guest["fruit_ninja"]["classic_high_score"] == 880
        assert guest["fruit_ninja"]["total_sliced"] == 120
    finally:
        player_profiles.LEGACY_BOXER_FILE = orig_b
        player_profiles.LEGACY_FRUIT_FILE = orig_f
