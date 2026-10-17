"""数据准备的可复核科研图：只读 CSV/NetCDF，不修改模型与训练结果。"""

from pathlib import Path

import numpy as np
import pandas as pd


def plot_basin_forcing_checks(report_dir, figure_dir):
    """共同覆盖区的年度 P/PET/Q 量级和月气候态，不将 PET 误认作 AET。"""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    report_dir, figure_dir = Path(report_dir), Path(figure_dir)
    table = pd.read_csv(report_dir / "basin_daily_climate_discharge.csv", index_col="date", parse_dates=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"font.family": "Times New Roman", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "axes.linewidth": 0.7}):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), layout="constrained")
        # 仅完整年画年度水深，缺测流量年份不填0或推算全年量。
        for name, label, color in (("prec", "P", "#406d91"), ("pet", "PET", "#ac674b"), ("qobs_mm_day", "Observed runoff", "#222222")):
            grouped = table[name].groupby(table.index.year)
            total, count = grouped.sum(min_count=1), grouped.count()
            expected = np.asarray([366 if pd.Timestamp(int(year), 12, 31).is_leap_year else 365 for year in total.index])
            total[count.to_numpy() != expected] = np.nan
            axes[0].plot(total.index, total.values, "o-", color=color, lw=1.1, ms=3, label=label)
        axes[0].set(title="a  Annual basin-average forcing and observed runoff", ylabel="Water depth (mm/year)", xlabel="Year")
        axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
        monthly = table.groupby(table.index.month).mean(numeric_only=True)
        for name, label, color in (("prec", "P", "#406d91"), ("pet", "PET", "#ac674b")):
            axes[1].plot(monthly.index, monthly[name], "o-", color=color, ms=3, label=label)
        axes[1].set(title="b  Monthly climatology over the available forcing period", xlabel="Month", ylabel="Mean daily depth (mm/day)", xticks=range(1, 13))
        temperature_axis = axes[1].twinx()
        temperature_axis.plot(monthly.index, monthly.t, "s--", color="#6b7863", ms=3, label="Temperature")
        temperature_axis.set_ylabel("Temperature (°C)")
        axes[1].legend(loc="upper left", ncol=2)
        temperature_axis.legend(loc="upper right")
        fig.savefig(figure_dir / "daily_forcing_checks.jpg", dpi=450)
        plt.close(fig)
    return figure_dir / "daily_forcing_checks.jpg"

