import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    figures_dir = repo_root / "report" / "figures"

    for name, metrics_path in [
        ("CNN (mel)", repo_root / "results_cnn_test.json"),
        ("Wav2Vec2 (raw)", repo_root / "results_w2v2_test.json"),
    ]:
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cm = np.array(metrics["confusion_matrix"], dtype=int)
        out_path = figures_dir / f"confusion_{metrics_path.stem}.png"
        save_confusion_matrix(cm, out_path, f"{name} – Confusion Matrix (Test)")

    print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()

