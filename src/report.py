def generate_report(results: dict[str, list[dict]], model_names: list[str]):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="white", font_scale=1.05)
    metrics = [
        ("score", "Score ↑", ".2f", "crest", 1),
        ("seconds", "Agent time (seconds) ↓", ".1f", "crest_r", None),
        ("estimated_cost_usd", "Estimated token spend (USD) ↓", ".5f", "crest_r", None),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(max(18, 5 * len(model_names)), 0.5 * len(results) + 3.5),
        layout="constrained",
    )
    for ax, (metric, title, fmt, cmap, maximum) in zip(axes, metrics):
        rows = []
        for task_results in results.values():
            rows.append([result[metric] for result in task_results])
        data = pd.DataFrame(rows, index=results, columns=model_names, dtype=float)
        data = pd.concat([pd.DataFrame([data.mean()], index=["Average"]), data])
        ax.set_facecolor("#e5e7eb")
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap=cmap, vmin=0, vmax=maximum,
            linewidths=2, linecolor="white", cbar=False,
            xticklabels=[name.rsplit("/", 1)[-1] for name in model_names], ax=ax,
        )
        ax.set_title(title, loc="left", weight="bold", pad=20)
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0, labelrotation=0)
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.axhline(1, color="white", linewidth=5)
    fig.supxlabel(
        "Base token rates; Kimi uses Modal's OpenRouter rate. Excludes caching, long-context premiums and Modal compute. Gray: unpriced.",
        fontsize=10, color="#64748b",
    )
    fig.savefig("report.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("Saved report.png")
