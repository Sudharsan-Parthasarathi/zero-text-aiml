"""
Generates rich, high-resolution training/sample images across diverse domains:
- Distinct dominant color themes (Red, Blue, Green, Gold, Purple, Neon, Monochrome)
- Real, dense English OCR text (Formulas, Code, Invoices, Notes, Biology)
- Clear visual categories (Animals, Vehicles, Aerospace, Architecture, Food, Art)
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "data" / "sample_images"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size: int = 24):
    try:
        # Try common Windows fonts
        for font_name in ["arial.ttf", "segoeui.ttf", "calibri.ttf", "consola.ttf"]:
            try:
                return ImageFont.truetype(font_name, size)
            except Exception:
                continue
    except Exception:
        pass
    return ImageFont.load_default()


def create_gradient_img(width, height, color1, color2, vertical=True):
    base = Image.new("RGB", (width, height), color1)
    top = Image.new("RGB", (width, height), color2)
    mask = Image.new("L", (width, height))
    mask_data = []
    for y in range(height):
        for x in range(width):
            ratio = (y / height) if vertical else (x / width)
            mask_data.append(int(255 * ratio))
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return base


def generate_all_images():
    font_lg = get_font(34)
    font_md = get_font(22)
    font_sm = get_font(16)
    font_xs = get_font(13)

    items = [
        # 1. Red Supercar
        {
            "filename": "red_supercar.jpg",
            "bg": ((217, 4, 41), (141, 2, 31)),
            "accent": (255, 255, 255),
            "accent2": (20, 20, 20),
            "title": "REDLINE GT SUPERCAR",
            "subtitle": "V12 TWIN TURBO RACING REDLINE 9000 RPM",
            "body": [
                "Color: Crimson Racing Red (#D90429)",
                "Horsepower: 850 HP | Top Speed: 340 km/h",
                "Carbon Ceramic Brakes & Aero Diffuser",
                "Italian Motorsport Engineering Excellence",
            ],
            "badge": "SUPERCAR - RED",
        },
        # 2. Deep Blue Ocean
        {
            "filename": "blue_ocean.jpg",
            "bg": ((0, 119, 182), (3, 4, 94)),
            "accent": (202, 240, 248),
            "accent2": (0, 180, 216),
            "title": "PACIFIC OCEAN TRENCH",
            "subtitle": "DEEP MARINE HYDROLOGY & OCEANOGRAPHY",
            "body": [
                "Color: Deep Oceanic Blue (#0077B6)",
                "Marianas Trench Depth: 11,034 meters",
                "Bioluminescent Marine Microorganisms",
                "Abyssal Zone Hydrothermal Vent Systems",
            ],
            "badge": "OCEAN - BLUE",
        },
        # 3. Emerald Forest
        {
            "filename": "emerald_forest.jpg",
            "bg": ((45, 106, 79), (8, 28, 21)),
            "accent": (216, 243, 220),
            "accent2": (82, 183, 136),
            "title": "AMAZON EMERALD CANOPY",
            "subtitle": "TROPICAL RAINFOREST BIODIVERSITY ECOSYSTEM",
            "body": [
                "Color: Vibrant Forest Emerald (#2D6A4F)",
                "Photosynthetic Chlorophyll Biomass Index: 98%",
                "Old Growth Woodland & Epiphytic Plant Species",
                "Global Carbon Sequestration Biosphere",
            ],
            "badge": "FOREST - GREEN",
        },
        # 4. Golden Sunflower
        {
            "filename": "golden_sunflower.jpg",
            "bg": ((255, 183, 3), (251, 133, 0)),
            "accent": (255, 255, 255),
            "accent2": (50, 30, 0),
            "title": "GOLDEN HELIANTHUS BLOOM",
            "subtitle": "SUMMER SUNFLOWER SOLAR HARVEST FIELD",
            "body": [
                "Color: Radiant Amber Gold (#FFB703)",
                "Fibonacci Spiral Seed Floret Formation",
                "Heliotropism Solar Orientation Tracking",
                "Organic Pure Sunflower Seed Honey Oil",
            ],
            "badge": "SUNFLOWER - YELLOW",
        },
        # 5. Purple Cosmic Nebula
        {
            "filename": "purple_nebula.jpg",
            "bg": ((114, 9, 183), (58, 12, 163)),
            "accent": (247, 37, 133),
            "accent2": (76, 201, 240),
            "title": "ORION STELLAR NEBULA",
            "subtitle": "DEEP SPACE ASTROPHYSICS & SUPERNOVA",
            "body": [
                "Color: Celestial Cosmic Violet (#7209B7)",
                "Interstellar Molecular Hydrogen Cloud",
                "Protostar Accretion Disk Formation",
                "Hubble Space Telescope Ultra Deep Field",
            ],
            "badge": "NEBULA - PURPLE",
        },
        # 6. Neon Cyberpunk
        {
            "filename": "neon_cyberpunk.jpg",
            "bg": ((20, 20, 35), (10, 10, 18)),
            "accent": (247, 37, 133),
            "accent2": (76, 201, 240),
            "title": "NEO TOKYO 2099",
            "subtitle": "CYBERPUNK NEON NOIR METROPOLIS",
            "body": [
                "Color: Neon Magenta & Electric Cyan",
                "Neural Interface Augmented Reality Systems",
                "Quantum Network Fiber Optic Grid",
                "Sub-orbital High Altitude Drones",
            ],
            "badge": "CYBERPUNK - NEON",
        },
        # 7. Python Deep Learning Code
        {
            "filename": "python_code.jpg",
            "bg": ((30, 30, 30), (18, 18, 18)),
            "accent": (86, 156, 214),
            "accent2": (78, 201, 176),
            "title": "PyTorch Neural Net Architecture",
            "subtitle": "MACHINE LEARNING DEEP LEARNING PYTHON SCRIPT",
            "body": [
                "import torch",
                "import torch.nn as nn",
                "class ZeroTextClassifier(nn.Module):",
                "    def __init__(self, emb_dim=512, classes=10):",
                "        self.fc = nn.Linear(emb_dim, classes)",
                "    def forward(self, x):",
                "        return torch.softmax(self.fc(x), dim=-1)",
                "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)",
            ],
            "badge": "CODE - PYTHON",
        },
        # 8. Physics Formulas
        {
            "filename": "physics_formulas.jpg",
            "bg": ((245, 247, 250), (225, 232, 240)),
            "accent": (15, 23, 42),
            "accent2": (37, 99, 235),
            "title": "QUANTUM MECHANICS FORMULAS",
            "subtitle": "THEORETICAL PHYSICS & THERMODYNAMICS",
            "body": [
                "E = mc^2  |  Planck: E = h * nu  |  Delta S >= 0",
                "Schrodinger: i * h_bar * (d/dt) Psi = H Psi",
                "Heisenberg Uncertainty: Delta x * Delta p >= h_bar / 2",
                "Maxwell Equations: Div E = rho / epsilon_0",
                "Einstein Field Equations: G_uv + Lambda g_uv = 8 pi G T_uv",
            ],
            "badge": "PHYSICS - FORMULA",
        },
        # 9. Molecular Genetics Notes
        {
            "filename": "genetics_notes.jpg",
            "bg": ((254, 250, 240), (243, 238, 225)),
            "accent": (26, 77, 46),
            "accent2": (230, 57, 70),
            "title": "MOLECULAR GENETICS STUDY NOTES",
            "subtitle": "DNA REPLICATION & CELLULAR BIOLOGY",
            "body": [
                "Base Pair Complementarity: Adenine - Thymine, Guanine - Cytosine",
                "DNA Double Helix antiparallel strands (5' to 3')",
                "Transcription by RNA Polymerase II into mRNA",
                "Ribosomal Translation synthesis of polypeptide proteins",
                "Mitochondrial oxidative phosphorylation ATP production",
            ],
            "badge": "BIOLOGY - NOTES",
        },
        # 10. Financial Tax Invoice
        {
            "filename": "tax_invoice.jpg",
            "bg": ((255, 255, 255), (240, 244, 248)),
            "accent": (30, 41, 59),
            "accent2": (14, 165, 233),
            "title": "OFFICIAL COMMERCIAL INVOICE",
            "subtitle": "BILLING ACCOUNTING STATEMENT #INV-2026-981",
            "body": [
                "Client: AI Tech Solutions Ltd | Date: 09 Nov 2026",
                "Item 1: Multi-modal Search Engine Pipeline: $3,200.00",
                "Item 2: FAISS Vector Indexing & Cloud API: $1,450.00",
                "Subtotal: $4,650.00 | VAT (10%): $465.00",
                "TOTAL BALANCE DUE: $5,115.00 [STATUS: PAID IN FULL]",
            ],
            "badge": "INVOICE - DOCUMENT",
        },
        # 11. Golden Retriever Dog
        {
            "filename": "golden_retriever.jpg",
            "bg": ((244, 162, 97), (231, 111, 81)),
            "accent": (255, 255, 255),
            "accent2": (38, 70, 83),
            "title": "GOLDEN RETRIEVER PUPPY",
            "subtitle": "DOMESTIC CANINE PET & COMPANION ANIMAL",
            "body": [
                "Breed: Purebred Scottish Golden Retriever",
                "Coat: Dense Water-Repellent Honey Gold Fur",
                "Temperament: Friendly, Intelligent, High Loyalty",
                "Active Working Guide & Retrieval Sporting Dog",
            ],
            "badge": "ANIMAL - DOG",
        },
        # 12. Supersonic Fighter Jet
        {
            "filename": "fighter_jet.jpg",
            "bg": ((100, 116, 139), (51, 65, 85)),
            "accent": (248, 250, 252),
            "accent2": (56, 189, 248),
            "title": "F-22 RAPTOR AIR SUPERIORITY",
            "subtitle": "FIFTH-GENERATION STEALTH FIGHTER AIRCRAFT",
            "body": [
                "Max Velocity: Mach 2.25 Supercruise Capability",
                "Pratt & Whitney F119 Turbofan Afterburning Engines",
                "Thrust Vectoring Nozzles for Extreme Agility",
                "Active Electronically Scanned Array Radar (AESA)",
            ],
            "badge": "AIRCRAFT - MILITARY",
        },
        # 13. Italian Pepperoni Pizza
        {
            "filename": "pepperoni_pizza.jpg",
            "bg": ((230, 57, 70), (180, 25, 40)),
            "accent": (255, 255, 255),
            "accent2": (244, 211, 94),
            "title": "WOODFIRED PEPPERONI PIZZA",
            "subtitle": "AUTHENTIC NAPOLITANA ITALIAN CUISINE",
            "body": [
                "Slow-Fermented Sourdough Crust Baked at 450 C",
                "San Marzano Crushed Tomato Puree & Basil Leaves",
                "Whole Milk Smoked Mozzarella & Spicy Pepperoni",
                "Extra Virgin Olive Oil & Aged Parmigiano Reggiano",
            ],
            "badge": "FOOD - PIZZA",
        },
        # 14. Vintage 35mm Camera
        {
            "filename": "vintage_camera.jpg",
            "bg": ((71, 85, 105), (30, 41, 59)),
            "accent": (226, 232, 240),
            "accent2": (245, 158, 11),
            "title": "LEICA 35MM RANGEFINDER",
            "subtitle": "CLASSIC MECHANICAL ANALOG FILM PHOTOGRAPHY",
            "body": [
                "Shutter Mechanism: Cloth Focal Plane 1/1000s",
                "Lens: Summicron 50mm f/2.0 Coated Glass Optics",
                "Solid Brass Chassis with Vulcanite Leather Grip",
                "High Contrast Split-Image Optical Viewfinder",
            ],
            "badge": "VINTAGE - CAMERA",
        },
        # 15. Geometric Abstract Art
        {
            "filename": "geometric_art.jpg",
            "bg": ((241, 245, 249), (203, 213, 225)),
            "accent": (15, 23, 42),
            "accent2": (239, 68, 68),
            "title": "BAUHAUS GEOMETRIC HARMONY",
            "subtitle": "ABSTRACT CONSTRUCTIVIST COMPOSITION",
            "body": [
                "Primary Palette: Cadmium Red, Cobalt Blue, Lemon Yellow",
                "Golden Ratio Intersecting Polyhedral Geometry",
                "Wassily Kandinsky Spatial Tension & Form",
                "Modernist Minimalist Exhibition Gallery 2026",
            ],
            "badge": "ART - ABSTRACT",
        },
        # 16. Handwritten Task List
        {
            "filename": "handwritten_todo.jpg",
            "bg": ((255, 253, 238), (242, 238, 215)),
            "accent": (30, 58, 138),
            "accent2": (185, 28, 28),
            "title": "PROJECT ACTION CHECKLIST",
            "subtitle": "HANDWRITTEN NOTEBOOK MEMORANDUM",
            "body": [
                "[X] 1. Fix Torch DLL loading sequence on Windows",
                "[X] 2. Build Smart Weight Decider (CLIP, OCR, Color)",
                "[X] 3. Spotlight search results in 3D & 2D galaxies",
                "[X] 4. Expand dataset to 30+ multi-modal images",
                "[ ] 5. Run full end-to-end verification tests",
            ],
            "badge": "HANDWRITTEN - TODO",
        },
        # 17. Alpine Sunset Peaks
        {
            "filename": "snowy_sunset.jpg",
            "bg": ((124, 58, 237), (249, 115, 22)),
            "accent": (255, 255, 255),
            "accent2": (254, 240, 138),
            "title": "HIMALAYAN ALPINE SUNSET",
            "subtitle": "GLACIAL MOUNTAIN PEAK ELEVATION 8,848M",
            "body": [
                "Alpenglow Sunset Gradient: Purple to Vivid Orange",
                "Sub-zero Permanent Glacial Ice Serac Formations",
                "High Altitude Extreme Mountain Climbing Expedition",
                "Clear Stratospheric Night Sky Constellations",
            ],
            "badge": "LANDSCAPE - SUNSET",
        },
        # 18. Barista Espresso Coffee
        {
            "filename": "espresso_cup.jpg",
            "bg": ((66, 32, 6), (26, 12, 2)),
            "accent": (254, 243, 199),
            "accent2": (217, 119, 6),
            "title": "SPECIALTY ROASTED ESPRESSO",
            "subtitle": "ETHIOPIAN ARABICA CAFE LATTE ART",
            "body": [
                "Roast Profile: Medium-Dark Single Origin Sidamo",
                "9 Bar High Pressure Steam Extraction Crema Layer",
                "Micro-foam Steamed Milk Rosette Latte Art",
                "Tasting Notes: Dark Cocoa, Hazelnut, Wild Blackberry",
            ],
            "badge": "CAFE - COFFEE",
        },
    ]

    width, height = 700, 500

    for item in items:
        out_path = SAMPLE_DIR / item["filename"]
        img = create_gradient_img(width, height, item["bg"][0], item["bg"][1], vertical=True)
        draw = ImageDraw.Draw(img)

        # Draw framing border
        draw.rectangle([16, 16, width - 16, height - 16], outline=item["accent2"], width=3)
        draw.rectangle([22, 22, width - 22, height - 22], outline=item["accent"], width=1)

        # Decorative graphical elements
        draw.rectangle([34, 34, 180, 68], fill=item["accent2"])
        draw.text((44, 42), item["badge"], fill=(255, 255, 255), font=font_sm)

        # Title
        draw.text((34, 85), item["title"], fill=item["accent"], font=font_lg)
        draw.text((34, 128), item["subtitle"], fill=item["accent2"], font=font_sm)

        # Separator line
        draw.line([(34, 155), (width - 34, 155)], fill=item["accent2"], width=2)

        # Body items (Rich OCR text)
        y_pos = 175
        for line in item["body"]:
            draw.text((38, y_pos), line, fill=item["accent"], font=font_md)
            y_pos += 34

        # Footer watermark
        draw.text((38, height - 42), "ZERO-TEXT MULTIMODAL BENCHMARK DATASET v2.0", fill=item["accent2"], font=font_xs)

        img.save(out_path, quality=95)
        print(f"Generated: {item['filename']} ({out_path.stat().st_size} bytes)")

    print(f"\nSuccessfully generated {len(items)} rich benchmark images in {SAMPLE_DIR}")


if __name__ == "__main__":
    generate_all_images()
