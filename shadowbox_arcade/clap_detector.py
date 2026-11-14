"""
=============================================================================
CLAP DETECTOR & ACOUSTIC TRANSIENT ANALYZER (clap_detector.py)
Multimodal Hand Clap Recognition with Adaptive Noise Floor, Crest Factor,
Spectral Transient Onset, and Biomechanical Pose Context.
=============================================================================
"""

import time
import math
import numpy as np

class ClapDetector:
    def __init__(self, sample_rate=44100, block_size=512, cooldown_sec=1.5):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.cooldown_sec = cooldown_sec

        # Audio DSP State
        self.noise_floor = 0.008
        self.short_rms = 0.0
        self.long_rms = 0.008
        self.peak_amp = 0.0
        self.crest_factor = 1.0
        self.energy_onset = 0.0
        self.prev_block_rms = 0.0

        # Sensitivity & Thresholds
        self.sensitivity = 1.0  # Adjustable 0.5 to 2.0
        self.min_peak_thresh = 0.045
        self.onset_ratio_thresh = 3.2
        self.min_crest_factor = 2.8

        # Detection Events
        self.last_audio_spike_time = 0.0
        self.last_audio_confidence = 0.0
        self.last_trigger_time = 0.0
        self.last_result_text = "IDLE"

        # Separate Shout (Kiai) Tracker
        self.vocal_rms = 0.0
        self.vocal_threshold = 0.038  # Calibrated for natural shout at full-body webcam distance
        self.last_shout_time = 0.0

        # Historical wrist distance for convergence estimation
        self.prev_wrist_dist = None
        self.prev_wrist_time = None
        self.wrist_velocity = 0.0

        # Running Diagnostic Log
        self.mic_available = False
        self.last_debug_info = {}

    def process_audio_block(self, audio_data):
        """
        Executed inside the sounddevice InputStream callback.
        Must be ultra-fast and non-blocking (< 0.1 ms).
        """
        if audio_data is None or len(audio_data) == 0:
            return

        # Handle stereo or mono input robustly (downmix stereo channels if present)
        if hasattr(audio_data, "ndim") and audio_data.ndim > 1 and audio_data.shape[1] > 1:
            samples = np.mean(audio_data, axis=1)
        else:
            samples = audio_data.flatten()
        
        # 1. Peak Amplitude
        peak = float(np.max(np.abs(samples)))
        self.peak_amp = peak

        # 2. Short-time RMS
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self.short_rms = rms

        # 3. Asymmetric Vocal Envelope Tracking (fast attack ~20ms, sustained decay ~350ms)
        if rms > self.vocal_rms:
            self.vocal_rms = self.vocal_rms * 0.40 + rms * 0.60
        else:
            self.vocal_rms = self.vocal_rms * 0.94 + rms * 0.06

        # Kiai shout latch: record timestamp when vocal energy exceeds threshold
        effective_shout_thresh = self.vocal_threshold / max(0.2, self.sensitivity)
        if self.vocal_rms >= effective_shout_thresh:
            self.last_shout_time = time.time()

        # 4. Adaptive Noise Floor (slow upward, fast downward tracking)
        if rms > self.noise_floor:
            self.noise_floor = self.noise_floor * 0.992 + rms * 0.008
        else:
            self.noise_floor = self.noise_floor * 0.960 + rms * 0.040
        self.noise_floor = max(0.003, min(0.15, self.noise_floor))

        # 5. Crest Factor (Peak to RMS ratio) - impulses have high crest factor
        if rms > 1e-5:
            self.crest_factor = peak / rms
        else:
            self.crest_factor = 1.0

        # 6. Energy Onset (Rate of change of energy relative to noise floor)
        onset_ratio = (rms - self.prev_block_rms) / max(1e-4, self.noise_floor)
        self.energy_onset = onset_ratio
        self.prev_block_rms = rms

        # 7. Clap Transient Evaluation
        # A clap requires:
        # - Peak exceeding absolute minimum threshold (scaled by sensitivity)
        # - Sharp energy onset ratio
        # - High crest factor (differentiates sharp clap from loud vowel shout)
        adjusted_peak_thresh = self.min_peak_thresh / max(0.2, self.sensitivity)
        adjusted_onset_thresh = self.onset_ratio_thresh / max(0.2, self.sensitivity)

        is_transient = (
            peak >= adjusted_peak_thresh and
            onset_ratio >= adjusted_onset_thresh and
            self.crest_factor >= self.min_crest_factor
        )

        if is_transient:
            now = time.time()
            # Only record if outside micro-refractory window (80ms to avoid multi-block double-firing)
            if (now - self.last_audio_spike_time) > 0.08:
                confidence = min(1.0, (peak / (adjusted_peak_thresh * 2.0)) * 0.4 +
                                      (onset_ratio / (adjusted_onset_thresh * 2.0)) * 0.3 +
                                      (self.crest_factor / 6.0) * 0.3)
                self.last_audio_spike_time = now
                self.last_audio_confidence = confidence

    def evaluate_clap(self, wrist_l=None, wrist_r=None, shoulder_l=None, shoulder_r=None, is_playing=True):
        """
        Called in the main game loop per frame.
        Fuses acoustic clap transient with biomechanical pose context.
        Returns (triggered: bool, confidence: float, debug_info: dict)
        """
        now = time.time()
        time_since_spike = now - self.last_audio_spike_time
        time_since_trigger = now - self.last_trigger_time
        cooldown_remaining = max(0.0, self.cooldown_sec - time_since_trigger)

        # Biomechanical analysis
        body_relative_dist = 999.0
        wrist_dist_px = 999.0
        is_converging = False
        pose_visible = False

        if wrist_l is not None and wrist_r is not None:
            pose_visible = True
            wrist_dist_px = math.hypot(wrist_l[0] - wrist_r[0], wrist_l[1] - wrist_r[1])
            
            # Normalize by shoulder width if available (body-relative scaling)
            if shoulder_l is not None and shoulder_r is not None:
                shoulder_w = math.hypot(shoulder_l[0] - shoulder_r[0], shoulder_l[1] - shoulder_r[1])
                body_relative_dist = wrist_dist_px / max(20.0, shoulder_w)
            else:
                # Fallback: estimate shoulder width as 180px
                body_relative_dist = wrist_dist_px / 180.0

            # Convergence velocity estimation
            if self.prev_wrist_dist is not None and self.prev_wrist_time is not None:
                dt_w = max(1e-4, now - self.prev_wrist_time)
                # Negative velocity means hands are getting closer
                self.wrist_velocity = (wrist_dist_px - self.prev_wrist_dist) / dt_w
                if self.wrist_velocity < -120.0:
                    is_converging = True
            
            self.prev_wrist_dist = wrist_dist_px
            self.prev_wrist_time = now
        else:
            self.prev_wrist_dist = None
            self.prev_wrist_time = None
            self.wrist_velocity = 0.0

        # Check if an acoustic transient occurred recently (< 120ms)
        audio_hit = (time_since_spike <= 0.12)
        final_confidence = 0.0
        triggered = False

        if audio_hit and is_playing:
            base_conf = self.last_audio_confidence
            
            if pose_visible:
                # If hands are close (body_relative_dist < 1.1) or converging
                if body_relative_dist < 0.85:
                    # Hands close together -> strong confirmation!
                    final_confidence = min(1.0, base_conf * 1.35)
                elif body_relative_dist < 1.4 or is_converging:
                    # Medium distance or hands converging rapidly -> good confirmation
                    final_confidence = min(1.0, base_conf * 1.15)
                else:
                    # Hands are spread far apart (e.g. full punch extension)
                    # Slightly discount, but loud authentic clap can still fire
                    final_confidence = base_conf * 0.75
            else:
                # Pose temporarily not visible -> rely on audio transient with moderate confidence
                final_confidence = base_conf

            # Trigger condition
            if final_confidence >= 0.50:
                if cooldown_remaining <= 0.0:
                    triggered = True
                    self.last_trigger_time = now
                    self.last_result_text = f"CLAP! (Conf: {final_confidence:.2f})"
                    # Consume spike
                    self.last_audio_spike_time = 0.0
                else:
                    self.last_result_text = f"COOLDOWN ({cooldown_remaining:.1f}s)"
            else:
                self.last_result_text = f"LOW CONF ({final_confidence:.2f})"
        elif cooldown_remaining > 0.0:
            self.last_result_text = f"COOLDOWN ({cooldown_remaining:.1f}s)"
        else:
            self.last_result_text = "READY"

        effective_shout_thresh = self.vocal_threshold / max(0.2, self.sensitivity)
        is_shout = (self.vocal_rms >= effective_shout_thresh) or ((now - self.last_shout_time) <= 0.6)

        debug_info = {
            "mic_available": self.mic_available,
            "short_rms": self.short_rms,
            "noise_floor": self.noise_floor,
            "peak_amp": self.peak_amp,
            "crest_factor": self.crest_factor,
            "energy_onset": self.energy_onset,
            "vocal_rms": self.vocal_rms,
            "vocal_threshold": effective_shout_thresh,
            "is_shouting": is_shout,
            "wrist_dist_px": wrist_dist_px,
            "body_relative_dist": body_relative_dist,
            "is_converging": is_converging,
            "cooldown_remaining": cooldown_remaining,
            "confidence": final_confidence,
            "result_text": self.last_result_text,
            "sensitivity": self.sensitivity
        }
        self.last_debug_info = debug_info

        return triggered, final_confidence, debug_info

    def is_shouting(self):
        """Returns True if current vocal energy exceeds Kiai shouting threshold."""
        thresh = self.vocal_threshold / max(0.2, self.sensitivity)
        return self.vocal_rms >= thresh

    def has_recent_shout(self, window_sec=0.6):
        """Returns True if user shouted within recent time window (latch for punch impact)."""
        now = time.time()
        if self.is_shouting():
            self.last_shout_time = now
            return True
        return (now - self.last_shout_time) <= window_sec

    def consume_shout(self):
        """Consumes the recent shout latch so one shout triggers one punch impact."""
        self.last_shout_time = 0.0

    def reset(self):
        """Reset state when starting or restarting a match."""
        self.last_trigger_time = 0.0
        self.last_audio_spike_time = 0.0
        self.last_shout_time = 0.0
        self.last_result_text = "READY"
        self.prev_wrist_dist = None
        self.prev_wrist_time = None
        self.wrist_velocity = 0.0

    def draw_diagnostic_hud(self, frame, x=20, y=140, scale=1.0):
        """
        Renders a semi-transparent diagnostic HUD panel in OpenCV BGR frame.
        Shows live RMS, Noise Floor, Peak, Crest Factor, Vocal Kiai, Pose Distance, and Status.
        """
        import cv2

        d = self.last_debug_info
        if not d:
            return

        w = int(280 * scale)
        h = int(212 * scale)
        
        # Translucent background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 80, 110), 1)

        # Header
        cv2.putText(frame, "ACOUSTIC & KIAI DIAGNOSTIC", (x + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, (255, 215, 0), 1)

        # Status Line
        status_col = (0, 255, 0) if d.get("result_text", "").startswith("CLAP") else (
            (0, 200, 255) if d.get("result_text") == "READY" else (180, 180, 180)
        )
        status_str = d.get('result_text', 'N/A')
        if d.get("is_shouting", False):
            status_str += " | KIAI ACTIVE!"
            status_col = (0, 215, 255)
        cv2.putText(frame, f"STATUS: {status_str}", (x + 10, y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, status_col, 1)

        # Level Bars
        # 1. RMS & Noise Floor
        cur_rms = min(1.0, d.get("short_rms", 0.0) * 8.0)
        nf = min(1.0, d.get("noise_floor", 0.0) * 8.0)
        bar_w = int(140 * scale)
        bar_x = x + int(115 * scale)
        
        # Noise floor bar
        cv2.putText(frame, "RMS / Floor:", (x + 10, y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (200, 200, 200), 1)
        cv2.rectangle(frame, (bar_x, y + 54), (bar_x + bar_w, y + 66), (40, 45, 55), -1)
        cv2.rectangle(frame, (bar_x, y + 54), (bar_x + int(bar_w * cur_rms), y + 66), (0, 230, 255), -1)
        # Noise floor marker
        cv2.line(frame, (bar_x + int(bar_w * nf), y + 52), (bar_x + int(bar_w * nf), y + 68), (0, 120, 255), 2)

        # 2. Vocal / Kiai Shout Meter
        voc_rms = min(1.0, d.get("vocal_rms", 0.0) * 12.0)
        voc_th = min(1.0, d.get("vocal_threshold", 0.04) * 12.0)
        is_shout = d.get("is_shouting", False)
        shout_col = (0, 215, 255) if is_shout else (120, 180, 255)
        cv2.putText(frame, "Vocal / Kiai:", (x + 10, y + 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (200, 200, 200), 1)
        cv2.rectangle(frame, (bar_x, y + 76), (bar_x + bar_w, y + 88), (40, 45, 55), -1)
        cv2.rectangle(frame, (bar_x, y + 76), (bar_x + int(bar_w * voc_rms), y + 88), shout_col, -1)
        # Threshold marker
        cv2.line(frame, (bar_x + int(bar_w * voc_th), y + 74), (bar_x + int(bar_w * voc_th), y + 90), (0, 255, 100), 2)

        # 3. Crest Factor & Peak
        cf = d.get("crest_factor", 1.0)
        peak = d.get("peak_amp", 0.0)
        cv2.putText(frame, f"Crest/Peak: {cf:.1f} / {peak:.2f}", (x + 10, y + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (200, 200, 200), 1)

        # 4. Energy Onset
        onset = d.get("energy_onset", 0.0)
        cv2.putText(frame, f"Onset Ratio: {onset:.1f}x", (x + 10, y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (200, 200, 200), 1)

        # 5. Wrist Dist (Body Rel)
        brd = d.get("body_relative_dist", 999.0)
        dist_str = f"{brd:.2f} body-w" if brd < 100.0 else "NO POSE"
        cv2.putText(frame, f"Wrist Dist: {dist_str}", (x + 10, y + 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (200, 200, 200), 1)

        # 6. Cooldown Bar
        cd = d.get("cooldown_remaining", 0.0)
        cd_ratio = min(1.0, cd / max(0.1, self.cooldown_sec))
        cv2.putText(frame, "Cooldown:", (x + 10, y + 174),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (200, 200, 200), 1)
        cv2.rectangle(frame, (bar_x, y + 164), (bar_x + bar_w, y + 176), (40, 45, 55), -1)
        cd_col = (0, 80, 220) if cd_ratio > 0 else (0, 200, 0)
        cv2.rectangle(frame, (bar_x, y + 164), (bar_x + int(bar_w * cd_ratio), y + 176), cd_col, -1)

        # Hotkey guide
        cv2.putText(frame, "[TAB] Toggle | [S] Test Shield | [K] Test Kiai", (x + 10, y + 198),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34 * scale, (150, 160, 180), 1)
