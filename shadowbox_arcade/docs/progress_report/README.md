# SHADOWBOX ARCADE: Sprint Progress Report (LaTeX Typesetting & Artifacts)

This directory contains the submission-ready engineering sprint progress report for **SHADOWBOX ARCADE: Fruit Ninja 2.0 — Fist-Controlled Laser Sword & Biomechanical Computer Vision Engine**, prepared for hackathon evaluation.

---

## 1. Directory Structure

```text
docs/progress_report/
├── SHADOWBOX_ARCADE_Progress_Report.pdf   # Master publication-ready PDF (8 pages, A4)
├── SHADOWBOX_ARCADE_Progress_Report.tex   # XeLaTeX / Tectonic master source
├── README.md                              # Build and documentation guide (this file)
├── figures/                               # 12 high-resolution UI and telemetry captures (1280x720)
│   ├── 01_arcade_launcher.png
│   ├── 02_player_modal.png
│   ├── 03_settings_modal.png
│   ├── 04_shadow_boxing_gameplay.png
│   ├── 05_shadow_boxing_gameover.png
│   ├── 06_fruit_ninja_menu.png
│   ├── 07_fruit_ninja_tutorial.png
│   ├── 08_fruit_ninja_gameplay.png
│   ├── 09_fruit_ninja_gameover.png
│   ├── 10_fruit_ninja_f3_diagnostics.png
│   ├── 11_shadow_boxing_f3_diagnostics.png
│   └── 12_shadow_boxing_calibration.png
└── page_previews/                         # Rasterized page previews (2x scale PNGs, 1191x1684)
    ├── page_1.png
    ├── page_2.png
    ├── page_3.png
    ├── page_4.png
    ├── page_5.png
    ├── page_6.png
    ├── page_7.png
    └── page_8.png
```

---

## 2. Compilation Instructions

### Option A: Using the Local Standalone Tectonic Engine (Recommended, Zero System Setup)
The repository includes a standalone Tectonic XeTeX binary (`tools/tectonic.exe`, v0.17.0) that automatically caches required packages without requiring a system-wide TeXLive or MiKTeX installation:

```powershell
# Navigate to report directory
cd docs/progress_report

# Compile directly to PDF
..\..\tools\tectonic.exe SHADOWBOX_ARCADE_Progress_Report.tex
```

### Option B: Using System XeLaTeX / LuaLaTeX
If a system-wide TeX distribution is installed:

```powershell
cd docs/progress_report
xelatex -synctex=1 -interaction=nonstopmode SHADOWBOX_ARCADE_Progress_Report.tex
# Run twice to resolve references and page numbers
xelatex -synctex=1 -interaction=nonstopmode SHADOWBOX_ARCADE_Progress_Report.tex
```

---

## 3. High-Resolution Visual Inspection & PNG Export

To re-render all 8 pages into high-resolution PNG previews for visual inspection or web upload, run the included `pypdfium2` pipeline via the project virtual environment:

```powershell
.venv\Scripts\python.exe -c "
import pypdfium2 as pdfium, os
pdf = pdfium.PdfDocument(r'docs/progress_report/SHADOWBOX_ARCADE_Progress_Report.pdf')
os.makedirs(r'docs/progress_report/page_previews', exist_ok=True)
for i, page in enumerate(pdf):
    page.render(scale=2.0).to_pil().save(f'docs/progress_report/page_previews/page_{i+1}.png')
print(f'Rendered {len(pdf)} pages successfully.')
"
```

---

## 4. Key Verified Metrics Documented in Report

| Metric / Specification | Value / Status | Verification Method |
|:---|:---|:---|
| **Hand Landmarker CPU Latency** | $17.77\text{ ms}$ ($56.3\text{ FPS}$) | `scripts/benchmark_engine.py` (Local CPU) |
| **Pose Landmarker CPU Latency** | $18.91\text{ ms}$ ($52.9\text{ FPS}$) | `scripts/benchmark_engine.py` (Local CPU) |
| **Swept Capsule Collision Overhead** | $1.73\ \mu\text{s}$ per fruit | 1,000-iteration synthetic benchmark |
| **Camera Buffer Age ($\tau_{\mathrm{age}}$)** | $8.33\text{ ms}$ average | Threaded producer-consumer telemetry |
| **Game Render Frame Rate** | **$60.0\text{ FPS}$ Locked** | Pygame display clock synchronization |
| **Automated Test Suite** | **64 / 64 Passed (100%)** | `pytest tests/ -v` (9.10s execution) |
| **Cloud / GPU Dependencies** | **None (100% Offline)** | Commodity laptop CPU, standard USB webcam |
