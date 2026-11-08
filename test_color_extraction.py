
from pathlib import Path
from PIL import Image
import json
import time

# ==========================================
# ZERO-TEXT DOMINANT COLOUR EXTRACTION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "data" / "sample_images"
OUTPUT_FILE = BASE_DIR / "data" / "color_results.json"

COLOUR_COUNT = 5

print("=" * 55)
print("       ZERO-TEXT COLOUR EXTRACTION ENGINE")
print("=" * 55)

print("\nScanning image directory...")
print("Image directory:", IMAGE_DIR)

if not IMAGE_DIR.exists():
    print("ERROR: Image directory not found!")
    raise SystemExit

extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

image_paths = []

for ext in extensions:
    image_paths.extend(IMAGE_DIR.glob(ext))

image_paths = sorted(image_paths)

if not image_paths:
    print("No images found!")
    raise SystemExit

print("Found", len(image_paths), "images.")

# ==========================================
# COLOUR EXTRACTION FUNCTION
# ==========================================

def extract_colours(image_path):

    image = Image.open(image_path).convert("RGB")

    # Reduce image size for faster processing
    image.thumbnail((200, 200))

    # Reduce image to five representative colours
    quantized = image.quantize(
        colors=COLOUR_COUNT,
        method=Image.Quantize.MEDIANCUT
    )

    # Get pixel counts for each palette entry
    colour_counts = quantized.getcolors(200 * 200)

    palette = quantized.getpalette()

    if not colour_counts:
        return []

    colour_counts.sort(reverse=True)

    total_pixels = sum(count for count, index in colour_counts)

    dominant_colours = []

    for count, index in colour_counts:

        position = index * 3

        r = palette[position]
        g = palette[position + 1]
        b = palette[position + 2]

        hex_code = "#{:02X}{:02X}{:02X}".format(r, g, b)

        percentage = (count / total_pixels) * 100

        dominant_colours.append({
            "hex": hex_code,
            "rgb": [r, g, b],
            "percentage": round(percentage, 2)
        })

    return dominant_colours


# ==========================================
# PROCESS ALL IMAGES
# ==========================================

all_results = []

start = time.time()

for image_path in image_paths:

    print("\nProcessing:", image_path.name)

    try:

        colours = extract_colours(image_path)

        print("Dominant colours:")

        for colour in colours:

            print(
                colour["hex"],
                "| RGB:", colour["rgb"],
                "|", colour["percentage"], "%"
            )

        all_results.append({
            "filename": image_path.name,
            "path": str(image_path),
            "dominant_colours": colours
        })

    except Exception as error:

        print("Colour extraction error:", error)

        all_results.append({
            "filename": image_path.name,
            "path": str(image_path),
            "dominant_colours": [],
            "error": str(error)
        })


# ==========================================
# SAVE RESULTS
# ==========================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        all_results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# FINAL REPORT
# ==========================================

successful = sum(
    bool(item["dominant_colours"])
    for item in all_results
)

print("\n" + "=" * 55)
print("COLOUR EXTRACTION COMPLETED")
print("=" * 55)

print("Images processed:", len(all_results))
print("Successful extractions:", successful)
print("Colours per image:", COLOUR_COUNT)
print("Output saved to:", OUTPUT_FILE)

print("Total time:", round(time.time() - start, 2), "seconds")

print("\nSUCCESS! ZERO-TEXT COLOUR ENGINE IS WORKING.")
