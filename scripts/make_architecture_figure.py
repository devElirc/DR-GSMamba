from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, text, width=1.9, height=0.55, color="#e8f1fb"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#1f4e79",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8)
    return patch


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=11, linewidth=1.0, color="#333333"))


def main():
    out = Path("paper/figures/dr_gsmamba_architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    box(ax, (0.25, 1.75), "HSI cube\nX", color="#f7fbff")
    box(ax, (2.05, 2.75), "Center\nspectrum")
    box(ax, (2.05, 1.75), "Local\npatch")
    box(ax, (2.05, 0.75), "Patch nodes")

    box(ax, (4.1, 2.75), "Spectral\nSSM encoder", color="#e6f4ea")
    box(ax, (4.1, 1.75), "Spatial\nCNN stem", color="#e6f4ea")
    box(ax, (4.1, 0.75), "Similarity graph\nmessage passing", color="#e6f4ea")

    box(ax, (6.25, 1.75), "Cross-branch\nfusion", color="#fff3e0")
    box(ax, (8.05, 2.25), "Linear\nclassifier", color="#fce8e6")
    box(ax, (8.05, 1.15), "Prototype and\nuncertainty head", color="#fce8e6")

    ax.text(8.98, 2.0, "Class map\n+ uncertainty", ha="center", va="center", fontsize=8)

    arrow(ax, (2.15, 2.03), (2.05, 3.02))
    arrow(ax, (2.15, 2.03), (2.05, 1.98))
    arrow(ax, (2.15, 2.03), (2.05, 1.02))
    arrow(ax, (3.95, 3.02), (4.1, 3.02))
    arrow(ax, (3.95, 2.02), (4.1, 2.02))
    arrow(ax, (3.95, 1.02), (4.1, 1.02))
    arrow(ax, (6.0, 3.02), (6.25, 2.2))
    arrow(ax, (6.0, 2.02), (6.25, 2.02))
    arrow(ax, (6.0, 1.02), (6.25, 1.85))
    arrow(ax, (8.15, 2.02), (8.05, 2.52))
    arrow(ax, (8.15, 2.02), (8.05, 1.42))
    arrow(ax, (9.95, 2.52), (9.65, 2.1))
    arrow(ax, (9.95, 1.42), (9.65, 1.9))

    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    main()

