def generate_report(results: dict[str, list[dict]], model_names: list[str]):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    sns.set_theme(style="white", font_scale=1.05)
    colors = LinearSegmentedColormap.from_list(
        "performance", ["#c75050", "#e5e7eb", "#36865a"]
    )
    metrics = [
        ("score", "Score ↑", ".2f", colors),
        ("seconds", "Agent time (seconds) ↓", ".1f", colors.reversed()),
        (
            "estimated_cost_usd",
            "Estimated token spend (USD) ↓",
            ".5f",
            colors.reversed(),
        ),
    ]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(6, 2 * len(model_names)), 3 * (0.5 * len(results) + 2.5)),
        layout="constrained",
    )
    for ax, (metric, title, fmt, cmap) in zip(axes, metrics):
        rows = []
        for task_results in results.values():
            rows.append([result[metric] for result in task_results])
        data = pd.DataFrame(rows, index=results, columns=model_names, dtype=float)
        data = pd.concat([pd.DataFrame([data.mean()], index=["Average"]), data])
        ax.set_facecolor("#e5e7eb")
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            vmin=data.min().min(),
            vmax=data.max().max(),
            linewidths=2,
            linecolor="white",
            cbar=False,
            xticklabels=[name.rsplit("/", 1)[-1] for name in model_names],
            ax=ax,
        )
        ax.set_title(title, loc="left", weight="bold", pad=20)
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0, labelrotation=0)
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.axhline(1, color="white", linewidth=5)
    fig.savefig("report.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("Saved report.png")
