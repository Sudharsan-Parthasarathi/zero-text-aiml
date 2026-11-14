"""
Audio asset procurement and verification script for SHADOWBOX ARCADE.
Downloads CC0 audio assets from Kenney's official repositories, validates playback,
and ensures proper integration into assets/audio/.
"""

import os
import sys
import hashlib
import urllib.request
import urllib.parse

os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

KENNEY_IMPACTS = {
    "impact_punch_medium_000.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_punch_medium_000.ogg",
    "impact_punch_medium_001.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_punch_medium_001.ogg",
    "impact_punch_medium_002.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_punch_medium_002.ogg",
    "impact_punch_heavy_000.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_punch_heavy_000.ogg",
    "impact_punch_heavy_001.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_punch_heavy_001.ogg",
    "impact_punch_heavy_002.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_punch_heavy_002.ogg",
    "impact_soft_heavy_000.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_soft_heavy_000.ogg",
    "impact_soft_heavy_001.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_soft_heavy_001.ogg",
    "impact_bell_heavy_000.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_bell_heavy_000.ogg",
    "impact_metal_heavy_000.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_metal_heavy_000.ogg",
    "impact_metal_heavy_001.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_metal_heavy_001.ogg",
    "impact_plate_heavy_000.ogg": "https://raw.githubusercontent.com/Boyquotes/kenney-impact-sounds-for-godot/master/addons/kenney%20impact%20sounds/impact_plate_heavy_000.ogg",
}

KENNEY_UI = {
    "click_001.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/click_001.wav",
    "select_001.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/select_001.wav",
    "select_002.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/select_002.wav",
    "toggle_001.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/toggle_001.wav",
    "toggle_002.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/toggle_002.wav",
    "open_001.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/open_001.wav",
    "close_001.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/close_001.wav",
    "confirmation_001.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/confirmation_001.wav",
    "confirmation_002.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/confirmation_002.wav",
    "back_001.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/back_001.wav",
    "error_001.wav": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/error_001.wav",
}

def download_and_verify(asset_dict, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    results = []
    pygame.init()
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    for filename, url in asset_dict.items():
        dest = os.path.join(target_dir, filename)
        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                content = resp.read()
                f.write(content)
            
            # Verify file size and md5
            size = os.path.getsize(dest)
            md5_hash = hashlib.md5(content).hexdigest()
            
            # Verify pygame mixer can load and play it
            snd = pygame.mixer.Sound(dest)
            duration = snd.get_length()
            
            print(f"  [OK] {filename} ({size} bytes, {duration:.3f}s, md5: {md5_hash[:8]})")
            results.append({
                "filename": filename,
                "path": dest,
                "size": size,
                "duration": duration,
                "md5": md5_hash,
                "url": url,
                "status": "VALID"
            })
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            results.append({
                "filename": filename,
                "path": dest,
                "status": f"FAILED: {e}",
                "url": url
            })
    return results

if __name__ == "__main__":
    boxing_dir = os.path.join(ROOT_DIR, "assets", "audio", "boxing")
    ui_dir = os.path.join(ROOT_DIR, "assets", "audio", "ui")
    
    print("=== Downloading Kenney CC0 Boxing Impact Sounds ===")
    boxing_res = download_and_verify(KENNEY_IMPACTS, boxing_dir)
    
    print("\n=== Downloading Kenney CC0 UI & Interface Sounds ===")
    ui_res = download_and_verify(KENNEY_UI, ui_dir)
    
    all_ok = all(r.get("status") == "VALID" for r in boxing_res + ui_res)
    print(f"\nCompleted procurement! All valid: {all_ok}")
    sys.exit(0 if all_ok else 1)
