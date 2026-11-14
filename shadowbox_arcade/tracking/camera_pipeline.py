"""
=============================================================================
SHADOWBOX TRACKING ENGINE 2.0: THREADED CAMERA PIPELINE (camera_pipeline.py)
High-Frequency Dedicated Acquisition Loop:
- Decouples hardware webcam read latency from game rendering & inference
- Bounded latest-frame buffer (zero stale frame queue accumulation)
- Hardware timestamping & capture-to-render frame age measurement
- Automatic camera drop recovery and thread-safe clean shutdown
- Headless mock support for automated CI test environments
=============================================================================
"""

import cv2
import sys
import os
import time
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Any

_CAMERA_OPEN_LOCK = threading.Lock()


class CameraPipeline:
    """
    Dedicated background camera acquisition worker.
    Runs cap.read() asynchronously so that game rendering and physics loops
    can maintain 60 FPS without being throttled by webcam driver exposure times.
    """

    def __init__(self, camera_index: int = 0, target_w: int = 640, target_h: int = 480, target_fps: int = 60):
        self.camera_index = camera_index
        self.target_w = target_w
        self.target_h = target_h
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()

        # Bounded latest-frame buffer
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_id: int = 0
        self._last_consumed_id: int = -1

        # Performance & Telemetry metrics
        self.capture_fps: float = 0.0
        self.capture_latency_ms: float = 0.0
        self._frame_times = []
        self._consecutive_failures: int = 0

        # Headless flag
        self.is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy" or os.environ.get("CI") == "true" or os.environ.get("SHADOWBOX_HEADLESS") == "1"

    def start(self) -> bool:
        """Initializes camera and launches background acquisition worker."""
        if self._running:
            return True

        self._open_camera()
        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, name="CameraWorker", daemon=True)
        self._thread.start()
        return True

    def _open_camera(self):
        if self.is_headless:
            return
        with _CAMERA_OPEN_LOCK:
            try:
                if sys.platform.startswith("win"):
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if not self.cap or not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.camera_index)
                else:
                    self.cap = cv2.VideoCapture(self.camera_index)

                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
                    self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    # Quick validation frame read
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        self.is_headless = False
                        print(f"[CameraPipeline] Hardware camera opened at index {self.camera_index} ({self.target_w}x{self.target_h}).")
                    else:
                        print(f"[CameraPipeline Warning] Camera opened but delivered empty frame. Headless fallback active.")
                        self.is_headless = True
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                else:
                    print(f"[CameraPipeline Warning] Could not open camera {self.camera_index}. Entering headless fallback.")
                    self.is_headless = True
            except Exception as e:
                print(f"[CameraPipeline Warning] Hardware capture exception: {e}. Headless fallback active.")
                self.is_headless = True
                self.cap = None

    def _capture_worker(self):
        """Dedicated background capture thread."""
        prev_t = time.time()
        while self._running:
            now = time.time()
            if self.is_headless or self.cap is None or not self.cap.isOpened():
                # Headless mode: generate dummy test frame at target FPS
                time.sleep(1.0 / max(1, self.target_fps))
                dummy = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
                with self._lock:
                    self._latest_frame = dummy
                    self._latest_timestamp = time.time()
                    self._frame_id += 1
                continue

            t_start = time.time()
            success, frame = self.cap.read()
            t_end = time.time()
            read_duration = (t_end - t_start) * 1000.0

            if success and frame is not None:
                self._consecutive_failures = 0
                dt = t_end - prev_t
                prev_t = t_end
                inst_fps = 1.0 / max(1e-4, dt)

                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = t_end
                    self._frame_id += 1
                    self.capture_latency_ms = 0.8 * self.capture_latency_ms + 0.2 * read_duration
                    self.capture_fps = 0.8 * self.capture_fps + 0.2 * inst_fps
            else:
                self._consecutive_failures += 1
                time.sleep(0.01)
                # Auto-recovery after repeated drops
                if self._consecutive_failures > 15:
                    print("[CameraPipeline] Camera drop detected. Attempting reconnect...")
                    try:
                        if self.cap:
                            self.cap.release()
                    except Exception:
                        pass
                    self._open_camera()
                    self._consecutive_failures = 0

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], float, float, int, bool]:
        """
        Retrieves the latest available frame without blocking.
        Returns:
            (success, frame, timestamp, frame_age_ms, frame_id, is_new_frame)
        """
        now = time.time()
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0, 0.0, 0, False

            frame = self._latest_frame.copy()
            timestamp = self._latest_timestamp
            fid = self._frame_id
            frame_age_ms = (now - timestamp) * 1000.0
            is_new = (fid != self._last_consumed_id)
            self._last_consumed_id = fid

            return True, frame, timestamp, frame_age_ms, fid, is_new

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns real-time pipeline performance metrics."""
        return {
            "capture_fps": round(self.capture_fps, 1),
            "read_latency_ms": round(self.capture_latency_ms, 2),
            "consecutive_failures": self._consecutive_failures,
            "is_headless": self.is_headless
        }

    def stop(self):
        """Safely terminates acquisition thread and frees camera driver."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap:
            try:
                self.cap.release()
                print("[CameraPipeline] Hardware camera cleanly released.")
            except Exception:
                pass
            self.cap = None
