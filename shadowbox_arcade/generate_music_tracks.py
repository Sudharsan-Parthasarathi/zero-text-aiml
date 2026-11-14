"""
Arcade Background Music Synthesizer
Generates original, seamlessly loopable arcade soundtrack themes:
1. menu_theme.wav: Atmospheric electronic ambient synth arpeggios (120 BPM)
2. boxing_theme.wav: Energetic combat theme with heavy punchy percussion (130 BPM)
3. fruit_ninja_theme.wav: Playful, bouncy pentatonic arcade groove (126 BPM)
"""

import os
import math
import wave
import numpy as np

MUSIC_DIR = os.path.join(os.path.dirname(__file__), "assets", "music")
os.makedirs(MUSIC_DIR, exist_ok=True)

SR = 44100

def save_wav(filename, stereo_audio, volume=0.85):
    """Normalize and write 16-bit stereo WAV."""
    path = os.path.join(MUSIC_DIR, filename)
    max_val = np.max(np.abs(stereo_audio))
    if max_val > 0:
        norm = stereo_audio / max_val
    else:
        norm = stereo_audio
    int16_data = (norm * 32767 * volume).astype(np.int16)
    
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(int16_data.tobytes())
    print(f"Generated soundtrack track: {path} ({os.path.getsize(path)} bytes)")

def synth_kick(dur=0.25):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    freq = np.linspace(140, 38, len(t))
    body = np.sin(2 * np.pi * freq * t)
    env = np.exp(-18 * t)
    return np.tanh(1.8 * body * env)

def synth_snare(dur=0.20):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    tone = np.sin(2 * np.pi * 180 * t) * np.exp(-32 * t)
    noise = np.random.uniform(-0.6, 0.6, len(t)) * np.exp(-22 * t)
    return np.tanh(1.5 * (0.4 * tone + 0.6 * noise))

def synth_hihat(dur=0.08):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    noise = np.random.uniform(-0.5, 0.5, len(t)) * np.exp(-65 * t)
    return noise

def gen_menu_theme():
    # 120 BPM = 2.0 sec per bar (4 beats). 8 bars = 16.0 sec loop
    bpm = 120
    spb = 60.0 / bpm
    total_beats = 32
    total_dur = total_beats * spb
    total_samples = int(SR * total_dur)
    t = np.linspace(0, total_dur, total_samples, endpoint=False)

    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)

    # Chord progression: Dm -> Bb -> F -> C (each 2 bars / 8 beats)
    roots = [146.83, 116.54, 174.61, 130.81]  # D3, Bb2, F3, C3
    
    # 1. Warm lush ambient pad
    for bar in range(4):
        start_t = bar * 8 * spb
        r = roots[bar]
        # Triad
        f1, f2, f3 = r, r * 1.2, r * 1.5
        idx1 = int(start_t * SR)
        idx2 = int((start_t + 8 * spb) * SR)
        t_sub = t[idx1:idx2] - start_t
        env = np.sin(np.pi * (t_sub / (8 * spb))) ** 1.5
        pad = (np.sin(2 * np.pi * f1 * t_sub) +
               0.7 * np.sin(2 * np.pi * f2 * t_sub) +
               0.5 * np.sin(2 * np.pi * f3 * t_sub)) * env * 0.28
        left[idx1:idx2] += pad
        right[idx1:idx2] += pad

    # 2. Hypnotic electronic arpeggiator (16th notes)
    sixteenth = spb / 4.0
    scale = [293.66, 349.23, 440.00, 523.25, 587.33]  # D minor pentatonic
    for step in range(total_beats * 4):
        note = scale[step % len(scale)]
        note_t = step * sixteenth
        dur_n = sixteenth * 0.75
        idx = int(note_t * SR)
        n_samples = int(dur_n * SR)
        if idx + n_samples <= total_samples:
            t_note = np.linspace(0, dur_n, n_samples, endpoint=False)
            tone = (np.sin(2 * np.pi * note * t_note) + 0.3 * np.sin(2 * np.pi * note * 2 * t_note)) * np.exp(-14 * t_note)
            # Stereo ping-pong
            pan = 0.5 + 0.35 * np.sin(step * 0.8)
            left[idx:idx+n_samples] += tone * (1.0 - pan) * 0.22
            right[idx:idx+n_samples] += tone * pan * 0.22

    # 3. Gentle sub-bass pulse on each measure
    for m in range(8):
        m_t = m * 4 * spb
        idx = int(m_t * SR)
        sub_dur = spb * 2.0
        n_sub = int(sub_dur * SR)
        if idx + n_sub <= total_samples:
            t_sub = np.linspace(0, sub_dur, n_sub, endpoint=False)
            sub = np.sin(2 * np.pi * 55 * t_sub) * np.exp(-2.5 * t_sub) * 0.4
            left[idx:idx+n_sub] += sub
            right[idx:idx+n_sub] += sub

    audio = np.column_stack((np.tanh(1.4 * left), np.tanh(1.4 * right)))
    save_wav("menu_theme.wav", audio, volume=0.85)

def gen_boxing_theme():
    # 130 BPM = 1.846 sec per bar. 8 bars = 14.769 sec loop
    bpm = 130
    spb = 60.0 / bpm
    total_beats = 32
    total_dur = total_beats * spb
    total_samples = int(SR * total_dur)
    t = np.linspace(0, total_dur, total_samples, endpoint=False)

    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)

    kick = synth_kick()
    snare = synth_snare()
    hihat = synth_hihat()

    # 1. Driving drum groove
    for beat in range(total_beats):
        bt = beat * spb
        b_idx = int(bt * SR)

        # Kick on 1 and 3 (beat 0 and 2 of each 4-beat bar)
        if beat % 4 in [0, 2]:
            end = min(total_samples, b_idx + len(kick))
            left[b_idx:end] += kick[:end-b_idx] * 0.55
            right[b_idx:end] += kick[:end-b_idx] * 0.55

        # Snare on 2 and 4 (beat 1 and 3)
        if beat % 4 in [1, 3]:
            end = min(total_samples, b_idx + len(snare))
            left[b_idx:end] += snare[:end-b_idx] * 0.45
            right[b_idx:end] += snare[:end-b_idx] * 0.45

        # Hi-hat 16ths
        for sub in range(4):
            hh_idx = b_idx + int(sub * (spb / 4.0) * SR)
            end = min(total_samples, hh_idx + len(hihat))
            acc = 0.25 if sub == 0 else 0.15
            left[hh_idx:end] += hihat[:end-hh_idx] * acc
            right[hh_idx:end] += hihat[:end-hh_idx] * acc

    # 2. Aggressive driving combat bassline (E minor)
    sixteenth = spb / 4.0
    bass_notes = [82.41, 82.41, 98.00, 82.41, 110.00, 82.41, 98.00, 73.42]
    for step in range(total_beats * 4):
        f = bass_notes[step % len(bass_notes)]
        st = step * sixteenth
        idx = int(st * SR)
        dur_b = sixteenth * 0.8
        n_b = int(dur_b * SR)
        if idx + n_b <= total_samples:
            tb = np.linspace(0, dur_b, n_b, endpoint=False)
            saw = (np.sin(2 * np.pi * f * tb) + 0.5 * np.sin(2 * np.pi * 2 * f * tb) + 0.3 * np.sin(2 * np.pi * 3 * f * tb))
            env = np.exp(-12 * tb)
            left[idx:idx+n_b] += saw * env * 0.35
            right[idx:idx+n_b] += saw * env * 0.35

    # 3. Motivational high-energy synth lead
    lead_scale = [329.63, 392.00, 440.00, 493.88, 587.33, 659.25]
    for bar in range(8):
        bar_t = bar * 4 * spb
        for note_i, pitch in enumerate([lead_scale[2], lead_scale[3], lead_scale[4], lead_scale[5]]):
            nt = bar_t + (note_i * 0.5 * spb) + spb * 2.0
            idx = int(nt * SR)
            dur_l = 0.35
            nl = int(dur_l * SR)
            if idx + nl <= total_samples:
                tl = np.linspace(0, dur_l, nl, endpoint=False)
                lead = (np.sin(2 * np.pi * pitch * tl) + 0.4 * np.sin(2 * np.pi * pitch * 1.5 * tl)) * np.exp(-6 * tl)
                left[idx:idx+nl] += lead * 0.28
                right[idx:idx+nl] += lead * 0.28

    audio = np.column_stack((np.tanh(1.6 * left), np.tanh(1.6 * right)))
    save_wav("boxing_theme.wav", audio, volume=0.88)

def gen_fruit_ninja_theme():
    # 126 BPM = 1.904 sec per bar. 8 bars = 15.238 sec loop
    bpm = 126
    spb = 60.0 / bpm
    total_beats = 32
    total_dur = total_beats * spb
    total_samples = int(SR * total_dur)
    t = np.linspace(0, total_dur, total_samples, endpoint=False)

    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)

    # 1. Bouncy playful marimba / kalimba melody (Pentatonic C major)
    scale = [523.25, 587.33, 659.25, 783.99, 880.00, 1046.50]
    pattern = [0, 2, 3, 2, 4, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    eighth = spb / 2.0
    for step in range(total_beats * 2):
        f = scale[pattern[step % len(pattern)]]
        st = step * eighth
        idx = int(st * SR)
        dur_m = eighth * 0.9
        nm = int(dur_m * SR)
        if idx + nm <= total_samples:
            tm = np.linspace(0, dur_m, nm, endpoint=False)
            marimba = (np.sin(2 * np.pi * f * tm) + 0.3 * np.sin(2 * np.pi * f * 2.75 * tm)) * np.exp(-16 * tm)
            pan = 0.5 + 0.4 * math.sin(step * 0.9)
            left[idx:idx+nm] += marimba * (1.0 - pan) * 0.38
            right[idx:idx+nm] += marimba * pan * 0.38

    # 2. Bouncy syncopated bass
    bass_notes = [130.81, 164.81, 196.00, 220.00]  # C, E, G, A
    for b in range(total_beats):
        bt = b * spb
        idx = int(bt * SR)
        dur_b = spb * 0.4
        nb = int(dur_b * SR)
        if idx + nb <= total_samples:
            tb = np.linspace(0, dur_b, nb, endpoint=False)
            bf = bass_notes[(b // 4) % len(bass_notes)]
            bass = np.sin(2 * np.pi * bf * tb) * np.exp(-9 * tb)
            left[idx:idx+nb] += bass * 0.32
            right[idx:idx+nb] += bass * 0.32

    # 3. Crisp percussion (Woodblock & Shaker)
    for b in range(total_beats):
        # Woodblock on 2 and 4
        if b % 2 == 1:
            idx = int(b * spb * SR)
            dur_w = 0.08
            nw = int(dur_w * SR)
            if idx + nw <= total_samples:
                tw = np.linspace(0, dur_w, nw, endpoint=False)
                wb = np.sin(2 * np.pi * 920 * tw) * np.exp(-55 * tw)
                left[idx:idx+nw] += wb * 0.28
                right[idx:idx+nw] += wb * 0.28

        # Shaker on 8th notes
        for s in range(2):
            s_idx = int((b * spb + s * eighth) * SR)
            dur_s = 0.05
            ns = int(dur_s * SR)
            if s_idx + ns <= total_samples:
                ts = np.linspace(0, dur_s, ns, endpoint=False)
                shk = np.random.uniform(-0.35, 0.35, ns) * np.exp(-80 * ts)
                left[s_idx:s_idx+ns] += shk * 0.20
                right[s_idx:s_idx+ns] += shk * 0.20

    audio = np.column_stack((np.tanh(1.5 * left), np.tanh(1.5 * right)))
    save_wav("fruit_ninja_theme.wav", audio, volume=0.86)

def generate_all_soundtracks():
    print("Synthesizing arcade background music soundtracks...")
    gen_menu_theme()
    gen_boxing_theme()
    gen_fruit_ninja_theme()
    print("All soundtracks generated successfully!")

if __name__ == "__main__":
    generate_all_soundtracks()
