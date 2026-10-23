def generate_report(scores: dict[str, list[float]], model_names: list[str]):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", font_scale=1.05)
    averages = [sum(column) / len(column) for column in zip(*scores.values())]
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(model_names)), 0.5 * len(scores) + 2.5))
    sns.heatmap(
        [averages, *scores.values()],
        annot=True, fmt=".2f", cmap="crest", vmin=0, vmax=1,
        linewidths=2, linecolor="white",
        xticklabels=[name.rsplit("/", 1)[-1] for name in model_names],
        yticklabels=["Average", *scores],
        cbar_kws={"label": "Score", "ticks": [0, 0.5, 1]}, ax=ax,
    )
    ax.set_title("Agent evaluation scores", loc="left", weight="bold", pad=20)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, labelrotation=0)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.axhline(1, color="white", linewidth=5)
    fig.savefig("report.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("Saved report.png")
