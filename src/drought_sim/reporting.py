from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def save_report_figure(climate: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    recent = climate.tail(730)
    axes[0].fill_between(recent["date"], recent["drought_index"], color="#D9863D", alpha=0.75)
    axes[0].set_title("Relative meteorological drought stress (last two simulated years)")
    axes[0].set_ylabel("Stress index (0–1)")
    axes[0].set_ylim(0, 1)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].errorbar(
        summary["richness"],
        summary["biomass_retention_mean"],
        yerr=summary["retention_ci95"],
        marker="o",
        capsize=4,
        color="#2AA887",
    )
    axes[1].set_title("Final biomass retention by initial species richness")
    axes[1].set_xlabel("Species richness")
    axes[1].set_ylabel("Drought / control biomass")
    axes[1].set_xticks(summary["richness"])
    axes[1].grid(alpha=0.2)
    figure.suptitle("Climate-driven plant-community simulation with 95% confidence intervals")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
