import time
import math
import numpy as np
import pytest
from clap_detector import ClapDetector

def generate_silence(samples=512):
    return np.zeros((samples, 1), dtype=np.float32)

def generate_speech(samples=512, freq=220.0, amp=0.08):
    t = np.linspace(0, samples/44100.0, samples, endpoint=False)
    # Sine wave with harmonics - low crest factor (~1.414)
    wave = amp * (0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t))
    return wave.reshape(-1, 1).astype(np.float32)

def generate_clap_impulse(samples=512, amp=0.45):
    # Sharp impulse transient with steep onset and high crest factor
    wave = np.zeros(samples, dtype=np.float32)
    # 5-15 ms decaying exponential pulse with burst noise
    dur_samples = min(samples, int(44100 * 0.025))
    t = np.linspace(0, 0.025, dur_samples, endpoint=False)
    pulse = np.random.uniform(-amp, amp, dur_samples) * np.exp(-140 * t)
    pulse[0:5] = amp * 0.95  # Fast leading edge
    wave[:dur_samples] = pulse
    return wave.reshape(-1, 1).astype(np.float32)

def test_silence_no_detection():
    detector = ClapDetector(cooldown_sec=1.5)
    for _ in range(5):
        detector.process_audio_block(generate_silence())
    triggered, conf, debug = detector.evaluate_clap(is_playing=True)
    assert not triggered
    assert conf == 0.0
    assert debug["result_text"] == "READY"

def test_speech_no_false_positive():
    detector = ClapDetector(cooldown_sec=1.5)
    # Feed continuous speech blocks
    for _ in range(20):
        detector.process_audio_block(generate_speech(amp=0.09))
        triggered, conf, _ = detector.evaluate_clap(is_playing=True)
        assert not triggered, "Speech should not trigger clap detection"
    # Vocal RMS should reflect speaking
    assert detector.vocal_rms > 0.04

def test_single_clap_detection():
    detector = ClapDetector(cooldown_sec=1.5)
    # Baseline ambient
    for _ in range(10):
        detector.process_audio_block(generate_silence())
    
    # Sharp clap
    clap = generate_clap_impulse(amp=0.55)
    detector.process_audio_block(clap)

    # Multimodal evaluation with hands close
    triggered, conf, debug = detector.evaluate_clap(
        wrist_l=(300, 300), wrist_r=(330, 300),
        shoulder_l=(250, 400), shoulder_r=(380, 400),
        is_playing=True
    )
    assert triggered is True
    assert conf >= 0.70
    assert "CLAP" in debug["result_text"]

def test_clap_with_poor_pose_visibility():
    detector = ClapDetector(cooldown_sec=1.5)
    # Baseline
    for _ in range(10):
        detector.process_audio_block(generate_silence())
    
    # Sharp clap
    detector.process_audio_block(generate_clap_impulse(amp=0.60))

    # Pose is None (player's hands occluded or out of frame)
    triggered, conf, debug = detector.evaluate_clap(wrist_l=None, wrist_r=None, is_playing=True)
    assert triggered is True, "Audio clap should trigger even when pose tracking is temporarily absent"
    assert conf >= 0.50

def test_clap_cooldown_and_refractory():
    detector = ClapDetector(cooldown_sec=1.5)
    for _ in range(10):
        detector.process_audio_block(generate_silence())
    
    # 1st clap
    detector.process_audio_block(generate_clap_impulse(amp=0.55))
    trig1, _, _ = detector.evaluate_clap(wrist_l=(300, 300), wrist_r=(320, 300), is_playing=True)
    assert trig1 is True

    # Immediate 2nd clap during cooldown
    detector.process_audio_block(generate_clap_impulse(amp=0.55))
    trig2, _, debug2 = detector.evaluate_clap(wrist_l=(300, 300), wrist_r=(320, 300), is_playing=True)
    assert trig2 is False
    assert "COOLDOWN" in debug2["result_text"]
    assert debug2["cooldown_remaining"] > 0.0

def test_not_playing_state_ignores_claps():
    detector = ClapDetector(cooldown_sec=1.5)
    detector.process_audio_block(generate_clap_impulse(amp=0.65))
    # In WAITING or PAUSED state (is_playing=False)
    triggered, _, _ = detector.evaluate_clap(is_playing=False)
    assert triggered is False

def test_kiai_shout_detection():
    detector = ClapDetector(cooldown_sec=1.5)
    # Ambient baseline
    for _ in range(5):
        detector.process_audio_block(generate_silence())
    assert not detector.is_shouting()
    assert not detector.has_recent_shout(0.6)

    # Sustained shout (high RMS, low crest factor)
    shout = generate_speech(amp=0.15)
    for _ in range(8):
        detector.process_audio_block(shout)
    
    assert detector.is_shouting()
    assert detector.has_recent_shout(0.6)

    # Consume shout latch
    detector.consume_shout()
    # If vocal_rms is currently above threshold, is_shouting is still True but consume clears the historical latch
    assert detector.last_shout_time == 0.0

def test_kiai_shout_latch_window():
    detector = ClapDetector(cooldown_sec=1.5)
    shout = generate_speech(amp=0.18)
    for _ in range(10):
        detector.process_audio_block(shout)
    
    assert detector.has_recent_shout(0.6)
    # Now feed silence: vocal_rms will decay, but recent shout remains True within window
    for _ in range(5):
        detector.process_audio_block(generate_silence())
    
    # Should still be within 0.6s window
    assert detector.has_recent_shout(0.6)

def test_kiai_sensitivity_scaling():
    det_low = ClapDetector()
    det_low.sensitivity = 0.5
    det_high = ClapDetector()
    det_high.sensitivity = 1.5

    # Medium vocal energy (amp=0.06 -> RMS ~0.04)
    med_speech = generate_speech(amp=0.06)
    for _ in range(10):
        det_low.process_audio_block(med_speech)
        det_high.process_audio_block(med_speech)
    
    # High sensitivity should trigger easily; low sensitivity should require louder shout
    assert det_high.is_shouting()
    assert not det_low.is_shouting()

def test_stereo_input_downmixing():
    detector = ClapDetector()
    # Create stereo block shape (512, 2)
    stereo_silence = np.zeros((512, 2), dtype=np.float32)
    detector.process_audio_block(stereo_silence)
    assert detector.short_rms == 0.0

    stereo_shout = np.hstack([generate_speech(amp=0.15), generate_speech(amp=0.15)])
    for _ in range(8):
        detector.process_audio_block(stereo_shout)
    assert detector.is_shouting()
