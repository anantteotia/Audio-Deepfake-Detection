from pathlib import Path

import matplotlib.pyplot as plt


def box(ax, xy, w, h, text):
    rect = plt.Rectangle(xy, w, h, fill=False, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "report" / "figures" / "framework.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_axis_off()

    # Boxes
    box(ax, (0.05, 0.35), 0.16, 0.3, "Audio (.wav)\nReal/Fake")
    box(ax, (0.27, 0.35), 0.18, 0.3, "Preprocess\n(resample 16k,\npad/trim)")
    box(ax, (0.52, 0.55), 0.18, 0.25, "Features A\nMel / MFCC")
    box(ax, (0.52, 0.20), 0.18, 0.25, "Features B\nRaw waveform")
    box(ax, (0.76, 0.55), 0.18, 0.25, "Model A\nCNN / RF / SVM")
    box(ax, (0.76, 0.20), 0.18, 0.25, "Model B\nWav2Vec2\n(frozen + head)")
    box(ax, (0.92, 0.35), 0.07, 0.3, "Output\nReal/Fake")

    # Arrows
    arrow(ax, (0.21, 0.50), (0.27, 0.50))
    arrow(ax, (0.45, 0.50), (0.52, 0.67))
    arrow(ax, (0.45, 0.50), (0.52, 0.32))
    arrow(ax, (0.70, 0.67), (0.76, 0.67))
    arrow(ax, (0.70, 0.32), (0.76, 0.32))
    arrow(ax, (0.94, 0.50), (0.92, 0.50))

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1)

    fig.suptitle("Overall Framework: Audio Deepfake Detection Pipeline", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

