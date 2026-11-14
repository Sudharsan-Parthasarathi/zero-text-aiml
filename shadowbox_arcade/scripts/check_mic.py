"""
=============================================================================
MICROPHONE INPUT & KIAI ACOUSTIC DIAGNOSTIC TOOL (scripts/check_mic.py)
Audits host audio APIs, input devices, RMS levels, and verifies Kiai DSP.
=============================================================================
"""

import sys
import os
import time
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clap_detector import ClapDetector

def check_microphone():
    print("=" * 65)
    print(" SHADOWBOX - MICROPHONE & KIAI ACOUSTIC DIAGNOSTIC SUITE")
    print("=" * 65)

    # 1. Check sounddevice availability
    try:
        import sounddevice as sd
    except ImportError:
        print("[FAIL] 'sounddevice' module not installed in Python environment.")
        return 1

    # 2. Host APIs and Devices
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    default_in = sd.default.device[0]

    print(f"\n[1] HOST APIS & INPUT DEVICES:")
    print(f"    Default Input Device ID: {default_in}")
    
    input_devices = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            input_devices.append((idx, d))
            is_def = " (DEFAULT)" if idx == default_in else ""
            api_name = hostapis[d["hostapi"]]["name"] if d["hostapi"] < len(hostapis) else "Unknown"
            print(f"    [{idx:2d}] {d['name'][:36]:36s} | API: {api_name:16s} | In: {d['max_input_channels']} ch | {int(d['default_samplerate'])} Hz{is_def}")

    if not input_devices:
        print("\n[FAIL] No microphone or audio input devices detected!")
        print("       Please plug in a microphone or headset and verify Windows permissions.")
        return 1

    # 3. Stream Opening Test
    print(f"\n[2] TESTING INPUT STREAM INITIALIZATION:")
    detector = ClapDetector(sample_rate=44100, block_size=512)
    stream = None
    stream_cfg = None

    attempts = [
        {"channels": 1, "samplerate": 44100, "blocksize": 512, "desc": "44.1 kHz Mono (Standard)"},
        {"channels": 2, "samplerate": 44100, "blocksize": 512, "desc": "44.1 kHz Stereo Array (Downmixed)"},
        {"channels": 1, "samplerate": 16000, "blocksize": 256, "desc": "16 kHz Mono (Fallback)"},
        {"channels": 2, "samplerate": 16000, "blocksize": 256, "desc": "16 kHz Stereo (Fallback)"}
    ]

    def dummy_callback(indata, frames, time_info, status):
        detector.process_audio_block(indata)

    for cfg in attempts:
        try:
            test_stream = sd.InputStream(
                channels=cfg["channels"],
                samplerate=cfg["samplerate"],
                callback=dummy_callback,
                blocksize=cfg["blocksize"]
            )
            test_stream.start()
            test_stream.stop()
            test_stream.close()
            stream_cfg = cfg
            print(f"    [PASS] Successfully opened {cfg['desc']}")
            break
        except Exception as e:
            print(f"    [SKIP] Failed {cfg['desc']}: {e}")

    if not stream_cfg:
        print("\n[FAIL] Unable to open any input stream with default device.")
        return 1

    # 4. Live Audio Recording & Level Test (1.5 seconds)
    print(f"\n[3] LISTENING TO LIVE MIC (1.5s sample)...")
    blocks_received = 0
    max_peak = 0.0
    rms_values = []

    def live_cb(indata, frames, time_info, status):
        nonlocal blocks_received, max_peak
        blocks_received += 1
        detector.process_audio_block(indata)
        p = float(np.max(np.abs(indata)))
        if p > max_peak:
            max_peak = p
        rms = float(np.sqrt(np.mean(indata**2)))
        rms_values.append(rms)

    stream = sd.InputStream(
        channels=stream_cfg["channels"],
        samplerate=stream_cfg["samplerate"],
        callback=live_cb,
        blocksize=stream_cfg["blocksize"]
    )

    with stream:
        time.sleep(1.5)

    avg_rms = float(np.mean(rms_values)) if rms_values else 0.0
    thresh = detector.vocal_threshold / max(0.2, detector.sensitivity)

    print(f"    Blocks Received:    {blocks_received}")
    print(f"    Max Peak Level:     {max_peak:.6f}")
    print(f"    Average RMS Energy: {avg_rms:.6f}")
    print(f"    Noise Floor:        {detector.noise_floor:.6f}")
    print(f"    Vocal RMS:          {detector.vocal_rms:.6f}")
    print(f"    Kiai Shout Thresh:  {thresh:.6f} (sensitivity: {detector.sensitivity}x)")
    print(f"    Recent Shout Latch: {detector.has_recent_shout(0.6)}")

    # 5. Diagnostic Evaluation
    print(f"\n[4] DIAGNOSTIC EVALUATION:")
    if blocks_received == 0:
        print("    [ERROR] Stream started but 0 audio callbacks were delivered.")
        print("            Check Windows audio service or driver permissions.")
        return 1
    elif max_peak < 0.0001:
        print("    [WARNING] Input level is near-zero (digital silence / 1 LSB).")
        print("              Possible reasons:")
        print("              1. Microphone is muted in Windows Sound Settings or hardware switch.")
        print("              2. Windows Settings > Privacy & security > Microphone is disabled.")
        print("              3. Microphone recording volume in Windows is set very low.")
        print("    [INFO] In-game vision fallback & [K] manual Kiai hotkey are active.")
    else:
        print("    [SUCCESS] Live microphone input is ACTIVE and receiving sound!")
        if detector.is_shouting():
            print("    [KIAI] Acoustic Battle Cry status: SHOUT TRIGGERED!")
        else:
            print("    [KIAI] Acoustic Battle Cry status: Ready (shout to trigger 5x critical hit).")

    print("\n" + "=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(check_microphone())
