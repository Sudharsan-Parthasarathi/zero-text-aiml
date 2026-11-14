"""
Fruit Ninja Arcade Asset Generator
Procedurally generates high-resolution, anti-aliased RGBA sprites for fruits, 
sliced halves, bombs, juice splatters, particles, and UI elements.
"""

import os
import math
import numpy as np
import cv2

ASSET_ROOT = os.path.join(os.path.dirname(__file__), "assets", "fruit_ninja")

def ensure_dirs():
    for sub in ["fruits", "sliced_fruits", "bombs", "particles", "backgrounds", "ui", "audio", "licenses"]:
        os.makedirs(os.path.join(ASSET_ROOT, sub), exist_ok=True)

def save_rgba(img_rgba, path):
    cv2.imwrite(path, img_rgba)

def make_supersampled_canvas(w=512, h=512):
    return np.zeros((h, w, 4), dtype=np.uint8)

def downsample(canvas, final_w=256, final_h=256):
    return cv2.resize(canvas, (final_w, final_h), interpolation=cv2.INTER_AREA)

# -----------------------------------------------------------------------------
# 1. WHOLE FRUIT GENERATORS
# -----------------------------------------------------------------------------

def gen_watermelon():
    c = make_supersampled_canvas(512, 512)
    center = (256, 256)
    radius = 190
    
    # Outer base green
    cv2.circle(c, center, radius, (40, 145, 30, 255), -1)
    
    # Dark green wavy tiger stripes
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, center, radius - 2, 255, -1)
    
    stripes = np.zeros((512, 512, 4), dtype=np.uint8)
    for x_offset in range(70, 450, 45):
        pts = []
        for y in range(60, 450, 8):
            wave = int(14 * math.sin(y * 0.045) + 6 * math.cos(y * 0.12))
            pts.append((x_offset + wave, y))
        for i in range(len(pts) - 1):
            cv2.line(stripes, pts[i], pts[i+1], (18, 75, 12, 255), 18)
    
    # Mask stripes inside watermelon
    for i in range(4):
        c[:, :, i] = np.where(mask > 0, np.where(stripes[:, :, 3] > 0, stripes[:, :, i], c[:, :, i]), c[:, :, i])
    
    # Outer dark rind rim
    cv2.circle(c, center, radius, (15, 60, 10, 255), 6)
    
    # Specular glossy highlight curve
    cv2.ellipse(c, (200, 190), (120, 75), -35, 0, 120, (255, 255, 255, 160), 16)
    cv2.ellipse(c, (190, 180), (110, 65), -35, 10, 100, (255, 255, 255, 230), 8)
    
    return downsample(c)

def gen_apple():
    c = make_supersampled_canvas(512, 512)
    # Apple contour: two overlapping lobes
    cv2.circle(c, (220, 270), 150, (30, 30, 220, 255), -1)
    cv2.circle(c, (292, 270), 150, (30, 30, 220, 255), -1)
    cv2.circle(c, (256, 310), 130, (25, 25, 205, 255), -1)
    # Top dip
    cv2.ellipse(c, (256, 175), (45, 30), 0, 0, 360, (0, 0, 0, 0), -1)
    # Bottom dip
    cv2.ellipse(c, (256, 420), (35, 20), 0, 0, 360, (0, 0, 0, 0), -1)
    
    # Shading and golden-yellow blush
    cv2.ellipse(c, (300, 280), (100, 120), 20, 0, 360, (40, 110, 235, 130), -1)
    # Specular shine
    cv2.ellipse(c, (190, 220), (55, 75), -25, 0, 360, (255, 255, 255, 180), -1)
    cv2.ellipse(c, (185, 215), (35, 50), -25, 0, 360, (255, 255, 255, 240), -1)
    
    # Stem
    pts = np.array([[256, 185], [262, 140], [278, 100], [288, 95], [282, 105], [268, 145], [262, 185]], np.int32)
    cv2.fillPoly(c, [pts], (20, 50, 90, 255))
    
    # Leaf
    cv2.ellipse(c, (310, 110), (35, 18), 35, 0, 360, (45, 180, 50, 255), -1)
    cv2.ellipse(c, (310, 110), (33, 16), 35, 0, 360, (70, 215, 80, 255), 2)
    return downsample(c)

def gen_orange():
    c = make_supersampled_canvas(512, 512)
    center = (256, 256)
    radius = 180
    
    # Orange base
    cv2.circle(c, center, radius, (10, 130, 255, 255), -1)
    # Shading
    cv2.circle(c, (270, 275), 160, (5, 95, 225, 110), -1)
    
    # Skin texture pore dimples
    np.random.seed(42)
    for _ in range(120):
        px = int(np.random.normal(256, 85))
        py = int(np.random.normal(256, 85))
        if math.hypot(px - 256, py - 256) < radius - 15:
            cv2.circle(c, (px, py), 2, (5, 85, 200, 180), -1)
            cv2.circle(c, (px+1, py+1), 1, (100, 200, 255, 140), -1)
            
    # Specular ring highlight
    cv2.ellipse(c, (190, 180), (80, 55), -30, 0, 360, (255, 255, 255, 170), -1)
    cv2.ellipse(c, (180, 175), (55, 35), -30, 0, 360, (255, 255, 255, 235), -1)
    
    # Top calyx star
    for angle in range(0, 360, 72):
        rad = math.radians(angle)
        tx = int(256 + 18 * math.cos(rad))
        ty = int(88 + 10 * math.sin(rad))
        cv2.line(c, (256, 88), (tx, ty), (30, 140, 40, 255), 4)
    cv2.circle(c, (256, 88), 6, (40, 160, 50, 255), -1)
    return downsample(c)

def gen_banana():
    c = make_supersampled_canvas(512, 512)
    # Banana crescent
    pts_outer = []
    pts_inner = []
    for deg in range(30, 155, 3):
        rad = math.radians(deg)
        r = 280
        cx, cy = 410, 80
        pts_outer.append([int(cx - r * math.cos(rad)), int(cy + r * math.sin(rad))])
    for deg in range(150, 28, -3):
        rad = math.radians(deg)
        r = 230
        cx, cy = 370, 110
        pts_inner.append([int(cx - r * math.cos(rad)), int(cy + r * math.sin(rad))])
        
    poly = np.array(pts_outer + pts_inner, np.int32)
    cv2.fillPoly(c, [poly], (35, 215, 255, 255))
    
    # Ridge lines
    for i in range(len(pts_outer) - 1):
        mid_x = int(0.5 * pts_outer[i][0] + 0.5 * pts_inner[len(pts_inner)-1-i][0])
        mid_y = int(0.5 * pts_outer[i][1] + 0.5 * pts_inner[len(pts_inner)-1-i][1])
        cv2.circle(c, (mid_x, mid_y), 4, (10, 180, 225, 255), -1)
        
    # Tips
    cv2.circle(c, (pts_outer[0][0], pts_outer[0][1]), 12, (20, 90, 130, 255), -1)
    cv2.circle(c, (pts_outer[-1][0], pts_outer[-1][1]), 14, (30, 140, 60, 255), -1)
    # Specular shine
    cv2.polylines(c, [np.array(pts_outer[10:-10], np.int32)], False, (255, 255, 255, 180), 6)
    return downsample(c)

def gen_strawberry():
    c = make_supersampled_canvas(512, 512)
    # Heart/conical strawberry shape
    pts = [
        [256, 440], [190, 360], [140, 260], [150, 170], [210, 140],
        [256, 165], [302, 140], [362, 170], [372, 260], [322, 360]
    ]
    cv2.fillPoly(c, [np.array(pts, np.int32)], (30, 35, 235, 255))
    # Smooth with ellipse
    cv2.ellipse(c, (256, 260), (115, 120), 0, 0, 360, (30, 35, 235, 255), -1)
    # Seeds (yellow dots in pits)
    for y in range(180, 410, 24):
        for x in range(170, 350, 24):
            if cv2.pointPolygonTest(np.array(pts, np.int32), (x, y), False) > 10:
                cv2.circle(c, (x, y), 4, (15, 20, 140, 255), -1)
                cv2.circle(c, (x, y), 2, (70, 220, 245, 255), -1)
    # Highlights
    cv2.ellipse(c, (210, 220), (35, 55), -30, 0, 360, (255, 255, 255, 160), -1)
    # Green calyx leaves at top
    for angle in [-60, -30, 0, 30, 60]:
        rad = math.radians(angle)
        lx = int(256 + 65 * math.sin(rad))
        ly = int(140 - 35 * math.cos(rad))
        cv2.line(c, (256, 150), (lx, ly), (45, 185, 50, 255), 14)
    return downsample(c)

def gen_kiwi():
    c = make_supersampled_canvas(512, 512)
    # Fuzzy brown oval
    cv2.ellipse(c, (256, 256), (175, 140), 0, 0, 360, (35, 75, 115, 255), -1)
    # Fine fuzzy hair rim
    np.random.seed(99)
    for deg in range(0, 360, 2):
        rad = math.radians(deg)
        x = 256 + 175 * math.cos(rad)
        y = 256 + 140 * math.sin(rad)
        hair_len = np.random.randint(6, 16)
        hx = int(x + hair_len * math.cos(rad))
        hy = int(y + hair_len * math.sin(rad))
        cv2.line(c, (int(x), int(y)), (hx, hy), (25, 55, 85, 200), 2)
    # Specular
    cv2.ellipse(c, (210, 210), (70, 45), -25, 0, 360, (255, 255, 255, 120), -1)
    return downsample(c)

def gen_coconut():
    c = make_supersampled_canvas(512, 512)
    center = (256, 256)
    radius = 175
    # Brown husk
    cv2.circle(c, center, radius, (25, 50, 85, 255), -1)
    # Fibrous texture
    np.random.seed(123)
    for _ in range(80):
        a1 = np.random.uniform(0, 2*math.pi)
        r1 = np.random.uniform(30, radius - 15)
        p1 = (int(256 + r1 * math.cos(a1)), int(256 + r1 * math.sin(a1)))
        p2 = (int(p1[0] + np.random.randint(-20, 20)), int(p1[1] + np.random.randint(-20, 20)))
        cv2.line(c, p1, p2, (15, 35, 60, 180), 3)
    # Coconut three dark eyes
    cv2.circle(c, (220, 180), 16, (12, 25, 45, 255), -1)
    cv2.circle(c, (285, 175), 18, (12, 25, 45, 255), -1)
    cv2.circle(c, (255, 235), 15, (12, 25, 45, 255), -1)
    return downsample(c)

def gen_golden_fruit():
    c = make_supersampled_canvas(512, 512)
    center = (256, 256)
    radius = 175
    # Glowing aura
    for r in range(radius + 35, radius, -3):
        cv2.circle(c, center, r, (30, 200, 255, int(180 * (1 - (r - radius)/35))), -1)
    # Metallic gold core
    cv2.circle(c, center, radius, (25, 180, 245, 255), -1)
    cv2.circle(c, (240, 240), 140, (50, 215, 255, 255), -1)
    # Brilliant flare
    cv2.ellipse(c, (190, 180), (65, 45), -35, 0, 360, (255, 255, 255, 240), -1)
    cv2.circle(c, (180, 170), 20, (255, 255, 255, 255), -1)
    return downsample(c)

def gen_frozen_fruit():
    c = make_supersampled_canvas(512, 512)
    center = (256, 256)
    radius = 175
    # Crystalline icy blue aura
    for r in range(radius + 30, radius, -3):
        cv2.circle(c, center, r, (255, 230, 150, int(150 * (1 - (r - radius)/30))), -1)
    # Ice ball
    cv2.circle(c, center, radius, (245, 205, 90, 255), -1)
    cv2.circle(c, (240, 240), 140, (255, 240, 180, 255), -1)
    # Ice crystal spires
    for deg in range(0, 360, 45):
        rad = math.radians(deg)
        x1 = int(256 + 130 * math.cos(rad))
        y1 = int(256 + 130 * math.sin(rad))
        x2 = int(256 + 195 * math.cos(rad))
        y2 = int(256 + 195 * math.sin(rad))
        cv2.line(c, (x1, y1), (x2, y2), (255, 255, 255, 240), 6)
    return downsample(c)

# -----------------------------------------------------------------------------
# 2. SLICED FRUIT HALVES GENERATORS (Left and Right halves)
# -----------------------------------------------------------------------------

def gen_watermelon_halves():
    # Left Half
    cl = make_supersampled_canvas(512, 512)
    # Semicircle contour
    cv2.ellipse(cl, (256, 256), (190, 190), 90, 0, 180, (40, 145, 30, 255), -1)
    # Inner white rind
    cv2.ellipse(cl, (256, 256), (174, 174), 90, 0, 180, (220, 245, 220, 255), -1)
    # Juicy red flesh
    cv2.ellipse(cl, (256, 256), (155, 155), 90, 0, 180, (45, 45, 235, 255), -1)
    # Seeds
    for deg in [110, 130, 150, 170, 190, 210, 230, 250]:
        rad = math.radians(deg)
        sx = int(256 + 95 * math.cos(rad))
        sy = int(256 + 95 * math.sin(rad))
        cv2.ellipse(cl, (sx, sy), (9, 5), deg, 0, 360, (20, 20, 20, 255), -1)
    # Flat cut surface highlight
    cv2.line(cl, (256, 66), (256, 446), (255, 255, 255, 120), 4)
    
    # Right Half (horizontal flip)
    cr = cv2.flip(cl, 1)
    return downsample(cl), downsample(cr)

def gen_orange_halves():
    cl = make_supersampled_canvas(512, 512)
    # Rind
    cv2.ellipse(cl, (256, 256), (180, 180), 90, 0, 180, (10, 130, 255, 255), -1)
    # White pith
    cv2.ellipse(cl, (256, 256), (165, 165), 90, 0, 180, (230, 245, 255, 255), -1)
    # Segments
    for deg in [110, 135, 160, 185, 210, 235, 250]:
        rad = math.radians(deg)
        sx = int(256 + 90 * math.cos(rad))
        sy = int(256 + 90 * math.sin(rad))
        cv2.ellipse(cl, (sx, sy), (45, 20), deg, 0, 360, (20, 150, 255, 255), -1)
    cv2.line(cl, (256, 76), (256, 436), (255, 255, 255, 140), 4)
    cr = cv2.flip(cl, 1)
    return downsample(cl), downsample(cr)

def gen_apple_halves():
    cl = make_supersampled_canvas(512, 512)
    # Red skin border
    cv2.ellipse(cl, (256, 256), (160, 175), 90, 0, 180, (30, 30, 220, 255), -1)
    # Pale crisp flesh
    cv2.ellipse(cl, (256, 256), (145, 162), 90, 0, 180, (225, 248, 250, 255), -1)
    # Apple core and brown seed
    cv2.ellipse(cl, (256, 256), (35, 55), 90, 0, 180, (160, 205, 215, 255), -1)
    cv2.ellipse(cl, (240, 256), (10, 5), 30, 0, 360, (20, 45, 80, 255), -1)
    cv2.line(cl, (256, 81), (256, 431), (255, 255, 255, 130), 4)
    cr = cv2.flip(cl, 1)
    return downsample(cl), downsample(cr)

def gen_banana_halves():
    cl = make_supersampled_canvas(512, 512)
    # Angled yellow peel
    cv2.ellipse(cl, (256, 256), (90, 180), 90, 0, 180, (35, 215, 255, 255), -1)
    # Cream interior
    cv2.ellipse(cl, (256, 256), (75, 165), 90, 0, 180, (210, 245, 255, 255), -1)
    # Center tri-dots
    cv2.circle(cl, (246, 245), 3, (80, 150, 180, 255), -1)
    cv2.circle(cl, (246, 265), 3, (80, 150, 180, 255), -1)
    cv2.line(cl, (256, 76), (256, 436), (255, 255, 255, 140), 4)
    cr = cv2.flip(cl, 1)
    return downsample(cl), downsample(cr)

def gen_kiwi_halves():
    cl = make_supersampled_canvas(512, 512)
    # Brown skin
    cv2.ellipse(cl, (256, 256), (170, 140), 90, 0, 180, (35, 75, 115, 255), -1)
    # Lime green flesh
    cv2.ellipse(cl, (256, 256), (155, 126), 90, 0, 180, (60, 210, 120, 255), -1)
    # White cream core
    cv2.ellipse(cl, (256, 256), (40, 30), 90, 0, 180, (220, 255, 235, 255), -1)
    # Black seed ring
    for deg in [120, 140, 160, 180, 200, 220, 240]:
        rad = math.radians(deg)
        sx = int(256 + 65 * math.cos(rad))
        sy = int(256 + 50 * math.sin(rad))
        cv2.circle(cl, (sx, sy), 4, (15, 15, 15, 255), -1)
    cv2.line(cl, (256, 116), (256, 396), (255, 255, 255, 130), 4)
    cr = cv2.flip(cl, 1)
    return downsample(cl), downsample(cr)

def gen_coconut_halves():
    cl = make_supersampled_canvas(512, 512)
    # Brown shell
    cv2.ellipse(cl, (256, 256), (170, 170), 90, 0, 180, (25, 50, 85, 255), -1)
    # White coconut meat
    cv2.ellipse(cl, (256, 256), (145, 145), 90, 0, 180, (245, 245, 250, 255), -1)
    # Hollow center
    cv2.ellipse(cl, (256, 256), (90, 90), 90, 0, 180, (15, 20, 30, 255), -1)
    cv2.line(cl, (256, 86), (256, 426), (255, 255, 255, 140), 4)
    cr = cv2.flip(cl, 1)
    return downsample(cl), downsample(cr)

def gen_strawberry_halves():
    cl = make_supersampled_canvas(512, 512)
    cv2.ellipse(cl, (256, 256), (115, 120), 90, 0, 180, (30, 35, 235, 255), -1)
    # Pale center rays
    cv2.ellipse(cl, (256, 256), (40, 50), 90, 0, 180, (140, 150, 255, 255), -1)
    cv2.line(cl, (256, 136), (256, 376), (255, 255, 255, 140), 4)
    cr = cv2.flip(cl, 1)
    return downsample(cl), downsample(cr)

# -----------------------------------------------------------------------------
# 3. BOMBS & HAZARDS
# -----------------------------------------------------------------------------

def gen_bomb():
    c = make_supersampled_canvas(512, 512)
    center = (256, 280)
    radius = 150
    
    # Black iron body
    cv2.circle(c, center, radius, (20, 20, 25, 255), -1)
    # Dark shadow rim
    cv2.circle(c, (270, 295), 140, (10, 10, 12, 160), -1)
    # Specular shine
    cv2.ellipse(c, (200, 220), (60, 40), -35, 0, 360, (180, 185, 200, 190), -1)
    cv2.circle(c, (185, 210), 18, (255, 255, 255, 240), -1)
    
    # Bomb collar neck
    cv2.rectangle(c, (230, 120), (282, 145), (40, 42, 50, 255), -1)
    cv2.ellipse(c, (256, 120), (26, 12), 0, 0, 360, (55, 60, 70, 255), -1)
    
    # Curved rope fuse
    pts = [
        (256, 120), (275, 95), (265, 65), (290, 45)
    ]
    for i in range(len(pts) - 1):
        cv2.line(c, pts[i], pts[i+1], (80, 140, 180, 255), 12)
        
    # Burning fuse spark
    spark_center = (290, 45)
    for r in range(45, 10, -5):
        cv2.circle(c, spark_center, r, (20, 160, 255, int(200 * (1 - r/45))), -1)
    cv2.circle(c, spark_center, 12, (255, 255, 255, 255), -1)
    # Spark flares
    for deg in range(0, 360, 45):
        rad = math.radians(deg)
        x2 = int(spark_center[0] + 32 * math.cos(rad))
        y2 = int(spark_center[1] + 32 * math.sin(rad))
        cv2.line(c, spark_center, (x2, y2), (40, 220, 255, 255), 4)
        
    return downsample(c)

# -----------------------------------------------------------------------------
# 4. JUICE SPLATTERS & PARTICLES
# -----------------------------------------------------------------------------

def gen_juice_splash(b, g, r):
    c = make_supersampled_canvas(512, 512)
    center = (256, 256)
    
    # Central splatter pool
    cv2.circle(c, center, 65, (b, g, r, 220), -1)
    
    # Splatter rays and droplets
    np.random.seed(int(b + g + r))
    for _ in range(16):
        ang = np.random.uniform(0, 2*math.pi)
        dist = np.random.uniform(80, 210)
        thick = np.random.randint(8, 22)
        p_end = (int(256 + dist * math.cos(ang)), int(256 + dist * math.sin(ang)))
        cv2.line(c, center, p_end, (b, g, r, 200), thick)
        # Droplet at tip
        cv2.circle(c, p_end, int(thick * 0.9), (b, g, r, 240), -1)
        
    # Loose droplets
    for _ in range(25):
        ang = np.random.uniform(0, 2*math.pi)
        dist = np.random.uniform(110, 240)
        drop_pos = (int(256 + dist * math.cos(ang)), int(256 + dist * math.sin(ang)))
        drop_r = np.random.randint(6, 16)
        cv2.circle(c, drop_pos, drop_r, (b, g, r, 220), -1)
        
    return downsample(c)

# -----------------------------------------------------------------------------
# 5. WOODEN CUTTING BOARD BACKGROUND
# -----------------------------------------------------------------------------

def gen_cutting_board():
    w, h = 1280, 720
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Warm rich dark mahogany / dojo bamboo wood tones
    base_color = np.array([30, 48, 80], dtype=np.float32)  # BGR
    
    # Draw horizontal wooden planks
    plank_h = 72
    for y in range(0, h, plank_h):
        shade_var = np.random.uniform(0.85, 1.15)
        p_color = np.clip(base_color * shade_var, 0, 255).astype(np.uint8)
        img[y:min(y+plank_h, h), :] = p_color
        # Plank seam groove
        if y > 0:
            cv2.line(img, (0, y), (w, y), (12, 20, 35), 3)
            cv2.line(img, (0, y+1), (w, y+1), (50, 70, 110), 1)
            
    # Subtle wood grain lines
    np.random.seed(777)
    for _ in range(250):
        gy = np.random.randint(0, h)
        gx1 = np.random.randint(0, w - 200)
        gx2 = gx1 + np.random.randint(80, 300)
        alpha = np.random.uniform(0.1, 0.25)
        color = (18, 30, 52)
        cv2.line(img, (gx1, gy), (gx2, gy), color, 1)
        
    # Vignette darkening at screen edges
    xv, yv = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    dist = np.sqrt(xv**2 + yv**2)
    vignette = np.clip(1.0 - (dist * 0.55), 0.35, 1.0)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * vignette).astype(np.uint8)
        
    return img

def gen_all():
    ensure_dirs()
    print("Generating Fruit Ninja arcade assets...")
    
    # Whole Fruits
    fruits = {
        "watermelon": gen_watermelon(),
        "apple": gen_apple(),
        "orange": gen_orange(),
        "banana": gen_banana(),
        "strawberry": gen_strawberry(),
        "kiwi": gen_kiwi(),
        "coconut": gen_coconut(),
        "golden_fruit": gen_golden_fruit(),
        "frozen_fruit": gen_frozen_fruit()
    }
    for name, img in fruits.items():
        p = os.path.join(ASSET_ROOT, "fruits", f"{name}.png")
        save_rgba(img, p)
        print(f"Saved fruit: {p}")
        
    # Halves
    halves = {
        "watermelon": gen_watermelon_halves(),
        "apple": gen_apple_halves(),
        "orange": gen_orange_halves(),
        "banana": gen_banana_halves(),
        "strawberry": gen_strawberry_halves(),
        "kiwi": gen_kiwi_halves(),
        "coconut": gen_coconut_halves()
    }
    for name, (left, right) in halves.items():
        pl = os.path.join(ASSET_ROOT, "sliced_fruits", f"{name}_left.png")
        pr = os.path.join(ASSET_ROOT, "sliced_fruits", f"{name}_right.png")
        save_rgba(left, pl)
        save_rgba(right, pr)
        print(f"Saved halves: {name}")

    # Bombs
    bomb_img = gen_bomb()
    save_rgba(bomb_img, os.path.join(ASSET_ROOT, "bombs", "bomb.png"))
    print("Saved bomb.png")
    
    # Juice Splatters (matching fruit flesh colors)
    splatters = {
        "splash_red": (45, 45, 235),       # Watermelon / Apple / Strawberry
        "splash_orange": (15, 140, 255),    # Orange
        "splash_yellow": (35, 215, 255),    # Banana
        "splash_green": (50, 210, 80),      # Kiwi
        "splash_gold": (30, 210, 255),      # Golden Fruit
        "splash_ice": (255, 230, 120)       # Frozen Fruit
    }
    for s_name, (b, g, r) in splatters.items():
        sp = gen_juice_splash(b, g, r)
        save_rgba(sp, os.path.join(ASSET_ROOT, "particles", f"{s_name}.png"))
        print(f"Saved splash: {s_name}")
        
    # Cutting Board Background
    board = gen_cutting_board()
    cv2.imwrite(os.path.join(ASSET_ROOT, "backgrounds", "dojo_cutting_board.png"), board)
    print("Saved dojo_cutting_board.png")
    
    print("Asset generation complete!")

if __name__ == "__main__":
    gen_all()
