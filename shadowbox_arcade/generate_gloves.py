import os
import cv2
import numpy as np
import math

SKIN_PALETTES = {
    "Classic Red": {
        "c_outline": (22, 10, 115, 255),       # #730A16 Deep burgundy outline
        "c_fill": (45, 42, 235, 255),          # #EB2A2D Vibrant boxing red
        "c_shadow": (25, 20, 175, 255),        # #AF1419 Deep leather shadow
        "c_highlight": (130, 120, 255, 255),   # #FF7882 Soft specular leather highlight
        "c_highlight_subtle": (90, 85, 245, 255),
        "c_strap": (38, 35, 222, 255),         # Wrist strap base
        "c_strap_shadow": (25, 18, 160, 255),
        "c_stitch": (22, 10, 115, 255)         # Stitch line color
    },
    "Electric Blue": {
        "c_outline": (115, 45, 15, 255),       # Deep navy outline
        "c_fill": (245, 145, 20, 255),         # Vivid electric blue
        "c_shadow": (180, 85, 10, 255),        # Deep cobalt shadow
        "c_highlight": (255, 220, 140, 255),   # Luminous cyan highlight
        "c_highlight_subtle": (250, 185, 80, 255),
        "c_strap": (230, 125, 25, 255),
        "c_strap_shadow": (165, 75, 15, 255),
        "c_stitch": (115, 45, 15, 255)
    },
    "Neon Purple": {
        "c_outline": (85, 15, 95, 255),        # Deep plum outline
        "c_fill": (220, 40, 180, 255),         # Vivid cyber violet
        "c_shadow": (150, 20, 110, 255),       # Dark magenta shadow
        "c_highlight": (255, 160, 240, 255),   # Luminous lilac highlight
        "c_highlight_subtle": (240, 110, 210, 255),
        "c_strap": (205, 30, 165, 255),
        "c_strap_shadow": (135, 15, 100, 255),
        "c_stitch": (85, 15, 95, 255)
    },
    "Golden": {
        "c_outline": (15, 75, 125, 255),       # Rich bronze outline
        "c_fill": (30, 195, 255, 255),         # Brilliant gold
        "c_shadow": (20, 135, 190, 255),       # Warm amber shadow
        "c_highlight": (160, 240, 255, 255),   # Sparkling pale gold highlight
        "c_highlight_subtle": (100, 220, 255, 255),
        "c_strap": (25, 175, 235, 255),
        "c_strap_shadow": (18, 120, 170, 255),
        "c_stitch": (15, 75, 125, 255)
    }
}
SKIN_PALETTES["Neon Violet"] = SKIN_PALETTES["Neon Purple"]

def draw_reference_glove(width=360, height=480, is_left=True, skin="Classic Red"):
    """
    Generate high-resolution RGBA boxing glove sprite precisely matching
    the professional 2D boxing glove illustration in multiple cosmetic skins.
    """
    img = np.zeros((height, width, 4), dtype=np.uint8)

    palette = SKIN_PALETTES.get(skin, SKIN_PALETTES["Classic Red"])
    c_outline = palette["c_outline"]
    c_fill = palette["c_fill"]
    c_shadow = palette["c_shadow"]
    c_highlight = palette["c_highlight"]
    c_highlight_subtle = palette["c_highlight_subtle"]
    c_strap = palette["c_strap"]
    c_strap_shadow = palette["c_strap_shadow"]
    c_stitch = palette["c_stitch"]

    def P(x, y):
        if not is_left:
            x = 1.0 - x
        return (int(x * width), int(y * height))

    # 1. Main Glove Fist Body Contour Points
    body_curve = [
        P(0.18, 0.22), P(0.20, 0.16), P(0.25, 0.11), P(0.33, 0.07), P(0.44, 0.06),
        P(0.55, 0.07), P(0.64, 0.10), P(0.70, 0.15), P(0.73, 0.22),
        P(0.74, 0.32), P(0.73, 0.44), P(0.70, 0.54), P(0.66, 0.65), P(0.65, 0.72),
        P(0.28, 0.72), P(0.27, 0.64), P(0.24, 0.52), P(0.20, 0.40), P(0.18, 0.30)
    ]
    body_pts = np.array(body_curve, np.int32)

    # 2. Thumb Section Contour Points
    thumb_curve = [
        P(0.70, 0.38), P(0.75, 0.32), P(0.82, 0.30), P(0.88, 0.33), P(0.92, 0.40),
        P(0.92, 0.50), P(0.88, 0.62), P(0.82, 0.70), P(0.75, 0.74), P(0.66, 0.72),
        P(0.68, 0.60), P(0.70, 0.50), P(0.71, 0.42)
    ]
    thumb_pts = np.array(thumb_curve, np.int32)

    # 3. Wrist Cuff Trapezoid Points
    cuff_curve = [
        P(0.26, 0.72), P(0.66, 0.72),
        P(0.70, 0.94), P(0.29, 0.94)
    ]
    cuff_pts = np.array(cuff_curve, np.int32)

    # 4. Velcro Strap Points (wrapped across wrist)
    strap_curve = [
        P(0.27, 0.75), P(0.67, 0.75),
        P(0.69, 0.90), P(0.28, 0.90)
    ]
    strap_pts = np.array(strap_curve, np.int32)

    # Bottom rolled rim of cuff
    rim_curve = [
        P(0.29, 0.93), P(0.70, 0.93),
        P(0.69, 0.96), P(0.30, 0.96)
    ]
    rim_pts = np.array(rim_curve, np.int32)

    # Render base shapes
    cv2.fillPoly(img, [cuff_pts], c_strap_shadow)
    cv2.polylines(img, [cuff_pts], True, c_outline, 5, cv2.LINE_AA)

    cv2.fillPoly(img, [body_pts], c_fill)
    cv2.fillPoly(img, [thumb_pts], c_fill)

    # Shadow on outer knuckle curvature
    shadow_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(shadow_mask, [body_pts], 255)
    shadow_layer = np.zeros_like(img)
    shadow_center = P(0.18, 0.45)
    cv2.ellipse(shadow_layer, shadow_center, (int(0.22*width), int(0.35*height)),
                0 if is_left else 0, 0, 360, c_shadow, -1, cv2.LINE_AA)
    
    cv2.ellipse(shadow_layer, P(0.75, 0.65), (int(0.12*width), int(0.12*height)),
                0, 0, 360, c_shadow, -1, cv2.LINE_AA)

    for c in range(4):
        img[:, :, c] = np.where(shadow_mask == 255,
                                np.where(shadow_layer[:, :, 3] > 0, shadow_layer[:, :, c], img[:, :, c]),
                                img[:, :, c])

    # Outlines
    cv2.polylines(img, [body_pts], True, c_outline, 6, cv2.LINE_AA)
    cv2.polylines(img, [thumb_pts], True, c_outline, 6, cv2.LINE_AA)

    # Specular leather highlight
    hl_center = P(0.46, 0.16)
    hl_axes = (int(0.16 * width), int(0.06 * height))
    hl_angle = -18 if is_left else 18
    cv2.ellipse(img, hl_center, hl_axes, hl_angle, 0, 360, c_highlight, -1, cv2.LINE_AA)

    hl_center_small = P(0.34, 0.22)
    hl_axes_small = (int(0.08 * width), int(0.035 * height))
    cv2.ellipse(img, hl_center_small, hl_axes_small, hl_angle + 5, 0, 360, c_highlight_subtle, -1, cv2.LINE_AA)

    # Thumb specular highlight
    thumb_hl_center = P(0.85, 0.40)
    thumb_hl_axes = (int(0.045 * width), int(0.08 * height))
    cv2.ellipse(img, thumb_hl_center, thumb_hl_axes, -10 if is_left else 10, 0, 360, c_highlight, -1, cv2.LINE_AA)

    # Strap
    cv2.fillPoly(img, [strap_pts], c_strap)
    cv2.polylines(img, [strap_pts], True, c_outline, 5, cv2.LINE_AA)

    # Vertical double stitch detailing
    s1_top, s1_bot = P(0.40, 0.77), P(0.41, 0.88)
    s2_top, s2_bot = P(0.44, 0.77), P(0.45, 0.88)
    for top, bot in [(s1_top, s1_bot), (s2_top, s2_bot)]:
        cv2.line(img, top, bot, c_stitch, 2, cv2.LINE_AA)
        for t in [0.2, 0.4, 0.6, 0.8]:
            sx = int(top[0] + t * (bot[0] - top[0]))
            sy = int(top[1] + t * (bot[1] - top[1]))
            cv2.circle(img, (sx, sy), 1, (240, 240, 240, 255), -1)

    # Rolled bottom hem
    cv2.fillPoly(img, [rim_pts], c_strap_shadow)
    cv2.polylines(img, [rim_pts], True, c_outline, 3, cv2.LINE_AA)

    return img

def generate_all_skins(output_dir=None):
    out = output_dir or os.path.dirname(__file__)
    for skin_name in SKIN_PALETTES.keys():
        tag = skin_name.lower().replace(" ", "_")
        gl = draw_reference_glove(width=360, height=480, is_left=True, skin=skin_name)
        gr = draw_reference_glove(width=360, height=480, is_left=False, skin=skin_name)
        
        cv2.imwrite(os.path.join(out, f"boxing_glove_left_{tag}.png"), gl)
        cv2.imwrite(os.path.join(out, f"boxing_glove_right_{tag}.png"), gr)
        print(f"Generated {skin_name} glove sprites.")

    # Also ensure default files exist
    cv2.imwrite(os.path.join(out, "boxing_glove_left.png"), draw_reference_glove(is_left=True, skin="Classic Red"))
    cv2.imwrite(os.path.join(out, "boxing_glove_right.png"), draw_reference_glove(is_left=False, skin="Classic Red"))
    print("All glove skins generated successfully!")

if __name__ == "__main__":
    generate_all_skins()
