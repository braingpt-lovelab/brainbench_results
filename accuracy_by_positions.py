import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import argparse_helper
from utils import human_meta


def overall_accuracy_human_vs_machine_created_cases_by_type_of_researcher(
        use_human_abstract
    ):
    # Read data
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")

    researcher_type_proportions = defaultdict(int)
    for researcher_type in human_meta.researcher_types:
        for _, row in df.iterrows():
            if row["current position"] == researcher_type:
                researcher_type_proportions[researcher_type] += 1

    if use_human_abstract:
        who = "human"
    else:
        who = "machine"

    acc_by_type = {}
    for researcher_type in human_meta.researcher_types:
        correct = 0
        total = 0
        for _, row in df.iterrows():
            if row["journal_section"].startswith(who) \
                    and row["current position"] == researcher_type:
                correct += row["correct"]
                total += 1
        acc_by_type[researcher_type] = (correct / total) if total > 0 else 0
    print(f"acc_by_type: {acc_by_type}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.rcParams.update({"font.size": 16, "font.weight": "bold"})
    colors = [
        "#FF7B89", "#8A5082", "#6F5F90", "#758EB7", "#A5CAD2", "#F7EDE2"
    ]

    # Plot accuracy by type of researcher
    researcher_types_legend = human_meta.researcher_types.copy()
    researcher_types_legend[-1] = "Other"

    wedges, texts, autotexts = axes[0].pie(
        list(researcher_type_proportions.values()),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 16},
        wedgeprops={"edgecolor": "black", "linewidth": 0.5}
    )

    # Adjust the position of a specific label (e.g., the last label)
    for text, autotext in zip(texts, autotexts):
        if autotext.get_text() == '2.4%':  # Check if this is the small slice
            x, y = autotext.get_position()
            autotext.set_position((x * 1.2, y * 1.2)) 

    axes[1].bar(
        np.arange(len(human_meta.researcher_types)),
        list(acc_by_type.values()),
        color=colors,
        edgecolor="black",
        label=researcher_types_legend
    )

    # Text accuracy on top of each bar
    for i, acc in enumerate(list(acc_by_type.values())):
        axes[1].text(
            i, acc + 0.02, f"{round(acc, 2)}", 
            ha="center", va="center", fontsize=12
        )
    
    axes[1].set_ylabel(f"Accuracy")
    axes[1].set_ylim(0., 1)
    axes[1].set_xticks([])
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)

    # Save each figure with a unique name
    # adjust bottom space
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.legend(loc='lower right', ncol=3, bbox_to_anchor=(1, -0.))
    plt.savefig(f"figs/accuracy_{who}_by_type_of_researcher.pdf")
    plt.close(fig)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=argparse_helper.str2bool, default=True)

    human_results_dir = "human_results"
    overall_accuracy_human_vs_machine_created_cases_by_type_of_researcher(
        parser.parse_args().use_human_abstract
    )
