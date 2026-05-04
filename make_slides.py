"""Generate A4 landscape presentation slides for EyeGaze Inference."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# -- constants
W, H = 11.69, 8.27  # A4 landscape in inches
BG = "#0d1117"
SURFACE = "#161b22"
TEXT = "#e6edf3"
DIM = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
RED = "#f85149"
BORDER = "#30363d"


def _base_fig(title=None):
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG)
    ax.axis("off")
    # top bar
    ax.axhline(y=0.88, xmin=0.05, xmax=0.95, color=BORDER, linewidth=0.8)
    if title:
        ax.text(0.5, 0.93, title, ha="center", va="center",
                fontsize=26, fontweight="bold", color=TEXT, fontfamily="sans-serif")
    # footer
    ax.text(0.05, 0.03, "Emanuele Di Luzio", fontsize=8, color=DIM, fontfamily="sans-serif")
    ax.text(0.95, 0.03, "EyeGaze Inference", fontsize=8, color=DIM,
            ha="right", fontfamily="sans-serif")
    return fig, ax


def _box(ax, x, y, w, h, text, color=BLUE, fontsize=11, text_color=TEXT):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                    facecolor=SURFACE, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontfamily="sans-serif")


def _arrow(ax, x1, y1, x2, y2, color=DIM):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def _bullets(ax, x, y, items, fontsize=13, spacing=0.055, color=TEXT):
    for i, item in enumerate(items):
        ax.text(x, y - i * spacing, item, fontsize=fontsize, color=color,
                fontfamily="sans-serif", va="center")


# =========================================================================
# SLIDE 1 — Title
# =========================================================================
def slide_title():
    fig, ax = _base_fig()
    ax.text(0.5, 0.62, "EyeGaze Inference", ha="center", va="center",
            fontsize=44, fontweight="bold", color=TEXT, fontfamily="sans-serif")
    ax.text(0.5, 0.52, "Real-Time Gaze Estimation with GazeDINO",
            ha="center", va="center", fontsize=20, color=DIM, fontfamily="sans-serif")
    ax.axhline(y=0.46, xmin=0.3, xmax=0.7, color=GREEN, linewidth=2)
    ax.text(0.5, 0.38, "DINOv2 ViT-B/14  +  MediaPipe FaceMesh  +  Polynomial Calibration",
            ha="center", va="center", fontsize=14, color=BLUE, fontfamily="sans-serif")
    ax.text(0.5, 0.28, "Best model: 3.24\u00b0 mean angular error",
            ha="center", va="center", fontsize=16, color=GREEN, fontfamily="sans-serif")
    ax.text(0.5, 0.15, "Emanuele Di Luzio", ha="center", va="center",
            fontsize=16, color=DIM, fontfamily="sans-serif")
    return fig


# =========================================================================
# SLIDE 2 — Architecture
# =========================================================================
def slide_architecture():
    fig, ax = _base_fig("Model Architecture — GazeDINO")

    # Input
    _box(ax, 0.08, 0.72, 0.18, 0.08, "Input\n(B, 3, 224, 224)", GREEN)
    _arrow(ax, 0.17, 0.72, 0.17, 0.66)

    # DINOv2
    _box(ax, 0.05, 0.52, 0.24, 0.12, "DINOv2 ViT-B/14\n86M params\nSelf-supervised on 142M imgs", BLUE)
    _arrow(ax, 0.17, 0.52, 0.17, 0.46)

    # CLS token
    _box(ax, 0.08, 0.36, 0.18, 0.06, "CLS Token [768]", PURPLE)
    _arrow(ax, 0.17, 0.36, 0.17, 0.30)

    # MLP head
    _box(ax, 0.03, 0.12, 0.28, 0.16, "MLP Head\nLayerNorm(768)\n768 \u2192 512 (GELU, drop 0.30)\n512 \u2192 128 (GELU, drop 0.15)\n128 \u2192 2", ORANGE, fontsize=10)

    # Output
    _arrow(ax, 0.17, 0.12, 0.17, 0.07)
    ax.text(0.17, 0.05, "(yaw, pitch) radians", ha="center", fontsize=11,
            color=GREEN, fontweight="bold", fontfamily="sans-serif")

    # Right side — training details
    rx = 0.55
    ax.text(rx, 0.82, "Training Details", fontsize=18, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")
    _bullets(ax, rx, 0.73, [
        "\u2022  Dataset: GazeGene (56 subjects, 9 cameras, ~90K samples)",
        "\u2022  Loss: Angular error (mean angle between 3D gaze vectors)",
        "\u2022  Best val error: 3.24\u00b0",
        "\u2022  Epochs: 50, optimizer: AdamW",
        "\u2022  Backbone frozen for first epochs, then fine-tuned",
    ], fontsize=12, spacing=0.06, color=DIM)

    ax.text(rx, 0.38, "Backbone Options", fontsize=18, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")

    # table
    headers = ["Backbone", "Params", "Embed"]
    rows = [
        ["dinov2_vits14", "21M", "384"],
        ["dinov2_vitb14", "86M", "768  \u2190 default"],
        ["dinov2_vitl14", "307M", "1024"],
        ["dinov2_vitg14", "1.1B", "1536"],
    ]
    ty = 0.30
    for j, h in enumerate(headers):
        ax.text(rx + j * 0.14, ty, h, fontsize=11, fontweight="bold",
                color=BLUE, fontfamily="sans-serif")
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = GREEN if "default" in cell else DIM
            ax.text(rx + j * 0.14, ty - (i + 1) * 0.045, cell,
                    fontsize=10, color=c, fontfamily="sans-serif")

    return fig


# =========================================================================
# SLIDE 3 — Inference Pipeline
# =========================================================================
def slide_pipeline():
    fig, ax = _base_fig("Inference Pipeline")

    # Pipeline boxes
    steps = [
        (0.08, 0.68, 0.16, 0.10, "Frame\n(BGR)", GREEN),
        (0.08, 0.50, 0.16, 0.10, "MediaPipe\nFaceMesh", BLUE),
        (0.08, 0.32, 0.16, 0.10, "Face Crop\n224\u00d7224", PURPLE),
        (0.08, 0.14, 0.16, 0.10, "GazeDINO\n(yaw, pitch)", ORANGE),
    ]
    for x, y, w, h, text, color in steps:
        _box(ax, x, y, w, h, text, color, fontsize=10)
    for i in range(len(steps) - 1):
        _arrow(ax, 0.16, steps[i][1], 0.16, steps[i+1][1] + steps[i+1][3])

    # Branches from FaceMesh
    bx = 0.38
    branches = [
        (0.70, "468 Landmarks + Iris\n\u2192 Pupillometry (iris/eye ratio)\n\u2192 Blink detection (EAR < 0.21)", BLUE),
        (0.55, "Cheekbone distance\n\u2192 d = (14cm \u00d7 focal) / face_px\n\u2192 Comfort zone: 40\u201390 cm", PURPLE),
        (0.40, "6-point solvePnP\n\u2192 Head pose (yaw, pitch, roll)\n\u2192 3D rotation via Rodrigues", ORANGE),
        (0.25, "EMA Smoothing\n\u2192 Gaze: \u03b1=0.25, Distance: \u03b1=0.1\n\u2192 Iris: \u03b1=0.2", GREEN),
    ]
    for by, text, color in branches:
        _box(ax, bx, by, 0.30, 0.10, text, color, fontsize=9)
        _arrow(ax, 0.24, 0.55, bx, by + 0.05)

    # Output
    _box(ax, 0.75, 0.12, 0.20, 0.10, "Screen (x, y)\nafter calibration", RED, fontsize=10)
    _arrow(ax, 0.16, 0.14, 0.16, 0.10)
    ax.text(0.16, 0.07, "\u2192 Calibration \u2192", ha="center", fontsize=9,
            color=DIM, fontfamily="sans-serif")
    _arrow(ax, 0.26, 0.08, 0.75, 0.17)

    return fig


# =========================================================================
# SLIDE 4 — Calibration
# =========================================================================
def slide_calibration():
    fig, ax = _base_fig("16-Point Gaze Calibration")

    # Left — grid visualization
    ax.text(0.15, 0.82, "4\u00d74 Calibration Grid", ha="center", fontsize=14,
            fontweight="bold", color=TEXT, fontfamily="sans-serif")

    grid_pts = [
        (0.08, 0.08), (0.37, 0.08), (0.63, 0.08), (0.92, 0.08),
        (0.08, 0.37), (0.37, 0.37), (0.63, 0.37), (0.92, 0.37),
        (0.08, 0.63), (0.37, 0.63), (0.63, 0.63), (0.92, 0.63),
        (0.08, 0.92), (0.37, 0.92), (0.63, 0.92), (0.92, 0.92),
    ]
    # draw in a box area
    gx0, gy0 = 0.04, 0.25
    gw, gh = 0.22, 0.50
    rect = mpatches.FancyBboxPatch((gx0, gy0), gw, gh, boxstyle="round,pad=0.005",
                                    facecolor="#0a0a0a", edgecolor=BORDER, linewidth=1)
    ax.add_patch(rect)
    for rx, ry in grid_pts:
        px = gx0 + rx * gw
        py = gy0 + (1 - ry) * gh
        ax.plot(px, py, 'o', color=GREEN, markersize=6)
        ax.plot(px, py, 'o', color="white", markersize=2)

    # Right — details
    rx = 0.38
    ax.text(rx, 0.82, "How It Works", fontsize=18, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")
    _bullets(ax, rx, 0.73, [
        "1.  Fullscreen overlay shows 16 dots (4\u00d74 grid)",
        "2.  Follow each dot with your eyes (2.8s per point)",
        "3.  Samples collected after 35% of dwell time",
        "4.  Outlier filtering: remove samples > 1.5\u03c3 from median",
        "5.  Degree-3 polynomial least-squares fit (10 coeff/axis)",
        "6.  Min 10 valid points required",
        "7.  Saved to calibration.pkl for reuse",
    ], fontsize=11, spacing=0.055, color=DIM)

    ax.text(rx, 0.30, "Polynomial Features (degree 3)", fontsize=14,
            fontweight="bold", color=TEXT, fontfamily="sans-serif")
    ax.text(rx, 0.22,
            "[1,  y,  p,  y\u00b2,  yp,  p\u00b2,  y\u00b3,  y\u00b2p,  yp\u00b2,  p\u00b3]",
            fontsize=13, color=BLUE, fontfamily="monospace")
    ax.text(rx, 0.15, "Two separate fits:  features \u2192 screen_x,  features \u2192 screen_y",
            fontsize=11, color=DIM, fontfamily="sans-serif")

    return fig


# =========================================================================
# SLIDE 5 — Screen Gaze Overlay
# =========================================================================
def slide_screen_gaze():
    fig, ax = _base_fig("Screen Gaze — Real-Time Overlay")

    # Left — explanation
    ax.text(0.08, 0.82, "How It Works", fontsize=18, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")
    _bullets(ax, 0.08, 0.73, [
        "1.  Calibrate (16 points, ~45s total)",
        "2.  Enable 'Screen gaze' toggle in sidebar",
        "3.  Translucent green dot appears on screen",
        "4.  Follows your gaze in real-time",
        "5.  Position smoothed with EMA (\u03b1=0.5)",
        "6.  macOS: transparent background (systemTransparent)",
        "7.  Always-on-top overlay window",
    ], fontsize=12, spacing=0.055, color=DIM)

    # visual
    cx, cy = 0.70, 0.55
    # screen rectangle
    rect = mpatches.FancyBboxPatch((0.50, 0.25), 0.40, 0.50,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#0a0a0a", edgecolor=BORDER, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.70, 0.72, "Screen", ha="center", fontsize=10, color=DIM, fontfamily="sans-serif")

    # gaze dot
    ax.plot(cx, cy, 'o', color=GREEN, markersize=20, alpha=0.7)
    ax.plot(cx, cy, 'o', color="white", markersize=6)

    # trail (fading dots)
    trail = [(0.62, 0.48), (0.64, 0.50), (0.66, 0.52), (0.68, 0.54)]
    for i, (tx, ty) in enumerate(trail):
        alpha = 0.15 + i * 0.1
        ax.plot(tx, ty, 'o', color=GREEN, markersize=8, alpha=alpha)

    ax.text(0.70, 0.30, "gaze position", ha="center", fontsize=9,
            color=GREEN, fontfamily="sans-serif")

    # pipeline
    ax.text(0.08, 0.25, "Pipeline", fontsize=14, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")
    ax.text(0.08, 0.17,
            "GazeDINO \u2192 (yaw, pitch) \u2192 EMA \u2192 Poly3 calibrator \u2192 (screen_x, screen_y) \u2192 overlay dot",
            fontsize=11, color=BLUE, fontfamily="sans-serif")

    return fig


# =========================================================================
# SLIDE 6 — GUI & Features
# =========================================================================
def slide_gui():
    fig, ax = _base_fig("GUI & Features")

    col1x = 0.08
    col2x = 0.52

    ax.text(col1x, 0.82, "Dark-Mode Interface", fontsize=16, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")
    _bullets(ax, col1x, 0.73, [
        "\u2022  Built with CustomTkinter",
        "\u2022  Live camera feed with overlays",
        "\u2022  Sidebar: gaze, metrics, calibration, toggles",
        "\u2022  Multi-face support (up to 5 faces)",
        "\u2022  Stable face ordering via centroid tracking",
        "\u2022  Video mode: open files, seek, play/pause",
    ], fontsize=12, spacing=0.055, color=DIM)

    ax.text(col2x, 0.82, "Live Metrics", fontsize=16, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")

    metrics = [
        ("Yaw / Pitch", "Model output \u2192 degrees", GREEN),
        ("Distance", "Pinhole model, 40\u201390 cm zone", BLUE),
        ("Iris ratio", "Pupillometry proxy", PURPLE),
        ("EAR", "Blink detection (thresh 0.21)", ORANGE),
        ("Head pose", "solvePnP \u2192 yaw, pitch, roll", RED),
        ("Screen XY", "After calibration only", GREEN),
    ]
    for i, (name, desc, color) in enumerate(metrics):
        y = 0.71 - i * 0.065
        ax.text(col2x, y, name, fontsize=12, fontweight="bold",
                color=color, fontfamily="sans-serif")
        ax.text(col2x + 0.18, y, desc, fontsize=11, color=DIM, fontfamily="sans-serif")

    # Overlay toggles
    ax.text(col1x, 0.35, "Overlay Toggles", fontsize=14, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")
    toggles = ["Face box", "Gaze arrows", "Iris dots", "Screen gaze (post-calibration)"]
    for i, t in enumerate(toggles):
        y = 0.28 - i * 0.045
        ax.text(col1x + 0.02, y, "\u25a0", fontsize=10, color=GREEN, fontfamily="sans-serif")
        ax.text(col1x + 0.05, y, t, fontsize=11, color=DIM, fontfamily="sans-serif")

    # Keyboard shortcuts
    ax.text(col2x, 0.35, "Keyboard Shortcuts", fontsize=14, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")
    shortcuts = [("Q", "Quit"), ("C", "Calibrate"), ("Space", "Play/Pause"),
                 ("\u2190 \u2192", "Switch face")]
    for i, (key, desc) in enumerate(shortcuts):
        y = 0.28 - i * 0.045
        ax.text(col2x, y, key, fontsize=11, fontweight="bold",
                color=BLUE, fontfamily="monospace")
        ax.text(col2x + 0.08, y, desc, fontsize=11, color=DIM, fontfamily="sans-serif")

    return fig


# =========================================================================
# SLIDE 7 — Technical Stack & Summary
# =========================================================================
def slide_summary():
    fig, ax = _base_fig("Summary & Technical Stack")

    # Left column — stack
    ax.text(0.08, 0.82, "Tech Stack", fontsize=18, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")

    stack = [
        ("PyTorch + TorchVision", "DINOv2 backbone, inference", BLUE),
        ("MediaPipe", "FaceMesh 468 landmarks + iris", GREEN),
        ("OpenCV", "Video capture, drawing, solvePnP", PURPLE),
        ("CustomTkinter", "Dark-mode GUI", ORANGE),
        ("NumPy", "Polynomial fit, EMA, linear algebra", DIM),
    ]
    for i, (name, desc, color) in enumerate(stack):
        y = 0.72 - i * 0.065
        ax.text(0.08, y, name, fontsize=13, fontweight="bold", color=color, fontfamily="sans-serif")
        ax.text(0.32, y, desc, fontsize=11, color=DIM, fontfamily="sans-serif")

    # Right column — key numbers
    ax.text(0.55, 0.82, "Key Numbers", fontsize=18, fontweight="bold",
            color=TEXT, fontfamily="sans-serif")

    numbers = [
        ("3.24\u00b0", "Mean angular error"),
        ("86M", "DINOv2 ViT-B/14 parameters"),
        ("16", "Calibration points (4\u00d74)"),
        ("10", "Polynomial coefficients per axis"),
        ("5", "Max simultaneous faces"),
        ("~30 fps", "Real-time on Apple Silicon (MPS)"),
    ]
    for i, (num, desc) in enumerate(numbers):
        y = 0.72 - i * 0.065
        ax.text(0.55, y, num, fontsize=14, fontweight="bold", color=GREEN, fontfamily="sans-serif")
        ax.text(0.70, y, desc, fontsize=11, color=DIM, fontfamily="sans-serif")

    # Bottom — full pipeline summary
    ax.axhline(y=0.30, xmin=0.05, xmax=0.95, color=BORDER, linewidth=0.8)
    ax.text(0.50, 0.24, "End-to-End Pipeline", ha="center", fontsize=16,
            fontweight="bold", color=TEXT, fontfamily="sans-serif")

    pipeline_text = (
        "Webcam \u2192 MediaPipe FaceMesh \u2192 Face Crop \u2192 GazeDINO (yaw, pitch) "
        "\u2192 EMA \u2192 Poly3 Calibration \u2192 Screen (x, y) \u2192 Overlay Dot"
    )
    ax.text(0.50, 0.16, pipeline_text, ha="center", fontsize=12,
            color=BLUE, fontfamily="sans-serif")

    ax.text(0.50, 0.08, "github.com/emanuelediluzio/eyegaze-inference",
            ha="center", fontsize=11, color=DIM, fontfamily="sans-serif")

    return fig


# =========================================================================
# Generate PDF
# =========================================================================
def main():
    slides = [
        slide_title,
        slide_architecture,
        slide_pipeline,
        slide_calibration,
        slide_screen_gaze,
        slide_gui,
        slide_summary,
    ]

    out = "eyegaze_slides.pdf"
    with PdfPages(out) as pdf:
        for i, fn in enumerate(slides):
            print(f"  Slide {i+1}/{len(slides)}: {fn.__name__}")
            fig = fn()
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

    print(f"\nDone! {len(slides)} slides -> {out}")


if __name__ == "__main__":
    main()
