from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_table(rows: list[list[str]]):
    tbl = Table(rows, hAlign="LEFT", colWidths=[2.2 * inch, 1.2 * inch, 1.2 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def try_add_image(story, img_path: Path, width_in: float):
    if not img_path.exists():
        return
    img = ImageReader(str(img_path))
    iw, ih = img.getSize()
    width = width_in * inch
    height = width * (ih / float(iw))
    story.append(Image(str(img_path), width=width, height=height))
    story.append(Spacer(1, 0.15 * inch))


def build_pdf(out_pdf: Path):
    repo_root = out_pdf.parents[1]
    figures_dir = repo_root / "report" / "figures"

    cnn_metrics = load_metrics(repo_root / "results_cnn_test.json")
    w2v2_metrics = load_metrics(repo_root / "results_w2v2_test.json")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceAfter=10))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=6))

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="Audio Deepfake Detection - Final Report",
        author="Anant Teotia",
    )

    story = []

    # Cover page
    story.append(Paragraph("PROJECT FINAL REPORT", styles["H1"]))
    story.append(Paragraph("<b>Project Title:</b> Audio Deepfake Detection using Transformer-based Models", styles["Body"]))
    story.append(
        Paragraph(
            "<b>Name:</b> Anant Teotia<br/>"
            "<b>Course:</b> ITCS 5154 - Applied Machine Learning<br/>"
            "<b>Instructor:</b> Ziyu Liu (zliu23@charlotte.edu)",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Primary paper (method backbone):</b> Baevski et al., “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations”, NeurIPS 2020.",
            styles["Body"],
        )
    )
    story.append(Paragraph(f"<b>Date:</b> {date.today().isoformat()}", styles["Body"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "<b>GitHub:</b> anantteotia/Audio-Deepfake-Detection (report + supplementary materials included in repo)",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    # 1. Introduction
    story.append(Paragraph("1. Introduction", styles["H1"]))
    story.append(Paragraph("1.1 Problem statement", styles["H2"]))
    story.append(
        Paragraph(
            "Audio deepfakes (AI-generated or manipulated speech) can bypass voice-based authentication and enable impersonation, fraud, and misinformation. "
            "The goal of this project is to detect whether an input audio clip is <i>Real</i> or <i>Fake</i> using machine learning models trained on audio features and transformer representations.",
            styles["Body"],
        )
    )
    story.append(Paragraph("1.2 Motivation, challenges, open questions", styles["H2"]))
    story.append(
        Paragraph(
            "Key challenges include generalization across speakers and synthesis methods, sensitivity to channel/noise conditions, and the computational cost of transformer inference on long audio. "
            "Open questions include which representations (spectrogram features vs. self-supervised speech embeddings) are most robust under domain shift, and how to keep inference efficient without sacrificing detection performance.",
            styles["Body"],
        )
    )
    story.append(Paragraph("1.3 Approach overview", styles["H2"]))
    story.append(
        Paragraph(
            "I implemented an end-to-end pipeline: audio loading → resampling/padding → feature extraction (mel/MFCC or raw waveform) → model training (CNN/RF/SVM or Wav2Vec2-based classifier) → evaluation with standard metrics (Accuracy, F1, AUC).",
            styles["Body"],
        )
    )

    # 2. Background / Related Work
    story.append(Paragraph("2. Backgrounds / Related Work", styles["H1"]))
    story.append(
        Paragraph(
            "<b>(A)</b> ASVspoof (2019/2021) benchmarks established standard evaluation for spoofing attacks, focusing on generalization to unseen spoofing conditions and reporting EER/min-tDCF.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>(B)</b> Self-supervised speech models like Wav2Vec2 and HuBERT learn strong representations from raw audio and can be adapted for downstream classification with small labeled datasets.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>(C)</b> Modern anti-spoofing architectures (e.g., RawNet-style models and attention-based models such as AASIST) often combine time–frequency cues with strong temporal modeling to improve robustness.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Pros / Cons and relation to this project", styles["H2"]))
    story.append(
        Paragraph(
            "Spectrogram-based CNNs are efficient and easy to train, but can overfit to dataset-specific artifacts. Transformer speech encoders provide richer representations, but are heavier and can still latch onto shortcut cues on small datasets. "
            "This project uses both: a strong baseline (CNN on mel spectrograms) and a transformer-based classifier (Wav2Vec2 + lightweight head) for comparison.",
            styles["Body"],
        )
    )

    # 3. Methods
    story.append(Paragraph("3. Methods", styles["H1"]))
    story.append(Paragraph("3.1 Data preprocessing", styles["H2"]))
    story.append(
        Paragraph(
            "Audio clips are loaded as mono, resampled to 16 kHz, and padded/truncated to a fixed duration. For feature-based models, I compute log-mel spectrograms (64 mel bins) or MFCCs. "
            "For transformer models, I feed the raw waveform directly.",
            styles["Body"],
        )
    )
    story.append(Paragraph("3.2 Models and algorithms", styles["H2"]))
    story.append(
        Paragraph(
            "<b>Baseline:</b> CNN classifier over mel spectrograms (2D conv blocks + fully-connected head), and optional classic ML baselines (RF/SVM) over aggregated features. "
            "<b>Transformer:</b> Wav2Vec2 encoder (frozen) + mean pooling over time + MLP classifier head.",
            styles["Body"],
        )
    )
    story.append(Paragraph("3.3 Overall framework figure", styles["H2"]))
    try_add_image(story, figures_dir / "framework.png", width_in=6.6)

    # 4. Experiments
    story.append(Paragraph("4. Experiments", styles["H1"]))
    story.append(Paragraph("4.1 Experimental setup", styles["H2"]))
    story.append(
        Paragraph(
            "Because large benchmark datasets require registration/download time, I validated the full pipeline using a synthetic dataset generator included in the repo. "
            "The generator creates “real” samples (clean harmonic signals + low noise) and “fake” samples (extra harmonics + noise + click artifacts). "
            "Splits: Train 84, Val 18, Test 18 (balanced real/fake).",
            styles["Body"],
        )
    )
    story.append(Paragraph("4.2 Results", styles["H2"]))

    rows = [["Model", "Test Acc", "Test F1"]]
    if cnn_metrics:
        rows.append(
            [
                "CNN (mel spectrogram)",
                f"{cnn_metrics['accuracy']:.3f}",
                f"{cnn_metrics['f1']:.3f}",
            ]
        )
    if w2v2_metrics:
        rows.append(
            [
                "Wav2Vec2 (raw, frozen encoder)",
                f"{w2v2_metrics['accuracy']:.3f}",
                f"{w2v2_metrics['f1']:.3f}",
            ]
        )
    story.append(metric_table(rows))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Confusion matrices (test)", styles["H2"]))
    try_add_image(story, figures_dir / "confusion_results_cnn_test.png", width_in=4.8)
    try_add_image(story, figures_dir / "confusion_results_w2v2_test.png", width_in=4.8)

    story.append(Paragraph("4.3 Discussion: reproduction and limitations", styles["H2"]))
    story.append(
        Paragraph(
            "I was not able to reproduce ASVspoof-reported numbers in this submission because I did not run training on the official ASVspoof dataset within this environment. "
            "However, I did reproduce an end-to-end training + evaluation workflow and generated quantitative results on a controlled dataset. "
            "The CNN achieves perfect performance on the synthetic test set, indicating the synthetic task is likely too easy and highlights the need for real benchmarks. "
            "The Wav2Vec2 model shows weaker fake-class recall on the synthetic test split, suggesting that a frozen encoder and small dataset may be insufficient without careful tuning or more realistic data.",
            styles["Body"],
        )
    )

    # 5. Conclusions
    story.append(Paragraph("5. Conclusions and Future Work", styles["H1"]))
    story.append(
        Paragraph(
            "This project delivered a working, modular audio deepfake detection pipeline with both feature-based and transformer-based models, producing measurable results and reusable tooling. "
            "Future work: train/evaluate on ASVspoof 2019/2021 (report EER/min-tDCF), add stronger augmentations (codec, noise, reverb), perform cross-dataset evaluation, and explore efficiency methods (token pruning/merging) for long audio inference.",
            styles["Body"],
        )
    )

    # 6. Contributions
    story.append(Paragraph("6. My Contributions", styles["H1"]))
    story.append(
        Paragraph(
            "<b>Existing work used:</b> pretrained speech encoders from Hugging Face Transformers (Wav2Vec2/HuBERT) and standard ML libraries (PyTorch, scikit-learn, librosa). "
            "<b>My work:</b> implemented the dataset loader, feature extraction, baseline/transformer classifier modules, training/evaluation scripts, synthetic dataset generator, and produced experiment artifacts and this final report.",
            styles["Body"],
        )
    )

    # 7. Sharing agreement (set conservative default)
    story.append(Paragraph("7. (Anonymous) Sharing agreement", styles["H1"]))
    story.append(
        Paragraph(
            "I do <b>not</b> agree to share this submission as an example for next semester. (This field can be changed by the student before final Canvas upload if desired.)",
            styles["Body"],
        )
    )

    # References
    story.append(Paragraph("8. References", styles["H1"]))
    refs = [
        "Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS.",
        "Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM TASLP.",
        "Gong, Y., Chung, Y.-A., & Glass, J. (2021). AST: Audio Spectrogram Transformer. Interspeech.",
        "ASVspoof Challenge (2019/2021) papers and dataset documentation.",
    ]
    for r in refs:
        story.append(Paragraph(f"- {r}", styles["Body"]))

    doc.build(story)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_pdf = repo_root / "report" / "Final_Report_Anant_Teotia.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(out_pdf)
    print(f"Wrote PDF: {out_pdf}")


if __name__ == "__main__":
    main()

