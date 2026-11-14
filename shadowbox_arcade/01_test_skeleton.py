import cv2
import time
import math
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------------------------------------------------------
# SHADOW-BOXER: STEP 1 - SKELETON & CAMERA TEST
# Runs 100% locally on CPU / Integrated Graphics using Google MediaPipe
# -----------------------------------------------------------------------------

# Path to the downloaded model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure pose_landmarker_lite.task is in the same folder.")

# 1. Initialize MediaPipe PoseLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.PoseLandmarker.create_from_options(options)

# Pre-extract skeletal connections
CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

def draw_boxing_glove(frame, wrist, elbow, is_left=True):
    """Render a stylized arcade boxing glove with knuckle padding, thumb, strap & highlights."""
    wx, wy = wrist
    if elbow:
        ex, ey = elbow
        rad = math.atan2(wy - ey, wx - ex)
        deg = math.degrees(rad)
    else:
        rad = -math.pi / 2
        deg = -90

    # Colors: Electric Cyan for Left, Vivid Crimson for Right
    main_color = (245, 185, 20) if is_left else (20, 45, 245)
    shadow_color = (170, 100, 0) if is_left else (10, 20, 150)
    cuff_color = (240, 240, 240)

    # Forward knuckle projection
    fx = int(wx + 22 * math.cos(rad))
    fy = int(wy + 22 * math.sin(rad))

    px = -math.sin(rad)
    py = math.cos(rad)

    # 1. Wrist Cuff (White strap)
    cuff_p1 = (int(wx + px * 22), int(wy + py * 22))
    cuff_p2 = (int(wx - px * 22), int(wy - py * 22))
    cv2.line(frame, cuff_p1, cuff_p2, (15, 15, 15), 16)
    cv2.line(frame, cuff_p1, cuff_p2, cuff_color, 12)
    cv2.line(frame, cuff_p1, cuff_p2, (40, 40, 40), 2)

    # 2. Thumb
    thumb_dir = 1 if is_left else -1
    tx = int(wx + 14 * math.cos(rad) + thumb_dir * 25 * px)
    ty = int(wy + 14 * math.sin(rad) + thumb_dir * 25 * py)
    cv2.ellipse(frame, (tx, ty), (16, 12), deg + thumb_dir * 30, 0, 360, (15, 15, 15), -1)
    cv2.ellipse(frame, (tx, ty), (14, 10), deg + thumb_dir * 30, 0, 360, shadow_color, -1)
    cv2.ellipse(frame, (tx, ty), (11, 7), deg + thumb_dir * 30, 0, 360, main_color, -1)

    # 3. Main Padded Knuckles
    cv2.ellipse(frame, (fx, fy), (38, 30), deg, 0, 360, (15, 15, 15), -1)
    cv2.ellipse(frame, (fx, fy), (35, 27), deg, 0, 360, shadow_color, -1)
    f_inner_x = int(fx - 2 * math.cos(rad))
    f_inner_y = int(fy - 2 * math.sin(rad))
    cv2.ellipse(frame, (f_inner_x, f_inner_y), (32, 24), deg, 0, 360, main_color, -1)

    # 4. Glossy Highlight
    hx = int(fx - 12 * math.cos(rad) - 9 * px)
    hy = int(fy - 12 * math.sin(rad) - 9 * py)
    cv2.ellipse(frame, (hx, hy), (16, 7), deg - 15, 0, 360, (255, 255, 255), -1)

    # 5. Glove Brand Stamp
    lbl = 'L' if is_left else 'R'
    cv2.putText(frame, lbl, (fx - 9, fy + 7), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

# 2. Open Laptop Webcam
cap = cv2.VideoCapture(0)

# Set resolution to 640x480 for ultra-fast, smooth performance on laptops
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*50)
print("  SHADOW-BOXER SKELETON TEST RUNNING!")
print("  - Stand back so the webcam can see your upper body.")
print("  - Press 'q' on your keyboard to quit.")
print("="*50 + "\n")

prev_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Warning: Camera frame could not be read.")
        break

    # Flip horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe requires RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect body landmarks
    result = detector.detect(mp_image)

    # Draw skeleton if a person is detected
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]
        pts = {}

        # Convert normalized coordinates (0 to 1) into pixel coordinates
        for idx, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            pts[idx] = (px, py)

        # Draw bone connections (Neon Cyan lines)
        for start_idx, end_idx in CONNECTIONS:
            if start_idx in pts and end_idx in pts:
                cv2.line(frame, pts[start_idx], pts[end_idx], (255, 255, 0), 2)

        # Draw glowing joints
        for idx, (px, py) in pts.items():
            if idx not in (0, 15, 16):
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

        # Draw realistic arcade boxing gloves on wrists oriented along forearm
        if 15 in pts:
            elbow_l = pts.get(13)
            draw_boxing_glove(frame, pts[15], elbow_l, is_left=True)
        if 16 in pts:
            elbow_r = pts.get(14)
            draw_boxing_glove(frame, pts[16], elbow_r, is_left=False)

        # Display detected wrist coordinates
        if 15 in pts and 16 in pts:
            cv2.putText(frame, f"L Wrist: {pts[15]}", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"R Wrist: {pts[16]}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Calculate and display FPS (Frames Per Second)
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)} (Local CPU)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'Q' to Exit", (w - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Shadow-Boxer: Skeleton Test (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
