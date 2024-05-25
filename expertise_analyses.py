import os
import copy
import argparse
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from utils import subfields
from utils import argparse_helper

plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})


def plot_human_expertise_distributions_by_subfields(use_human_abstract):
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"

    human_expertise_all_subfields = collections.defaultdict(list)
    for subfield in subfields.subfield_names:
        for _, row in df.iterrows():
            if row["journal_section"].startswith(who) and row["journal_section"].endswith(subfield):
                human_expertise_all_subfields[subfield].append(row["expertise"])

    # Plot five subplots, each showing the distribution of expertise for a subfield
    fig, axs = plt.subplots(5, 1, figsize=(5, 10))
    for i, subfield in enumerate(subfields.subfield_names):
        sns.kdeplot(human_expertise_all_subfields[subfield], ax=axs[i], color="skyblue")
        axs[i].set_xlabel("Expertise")
        axs[i].set_ylabel("Density")
        axs[i].set_title(f"{subfield}")

        # Plot vertical lines indicating top 20% expertise
        top_20_percentile = np.percentile(human_expertise_all_subfields[subfield], 80)
        axs[i].axvline(top_20_percentile, color="purple", linestyle="--", label="Top 20%")
        axs[i].legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(f"{figs_dir}/expertise_by_subfields_distribution_{who}_abstracts.pdf")


def plot_human_expertise_distributions_by_question(use_human_abstract):
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"
    
    # Group expertise by `abstract_id`
    human_expertise_by_question = collections.defaultdict(list)
    for _, row in df.iterrows():
        if row["journal_section"].startswith(who):
            human_expertise_by_question[row["abstract_id"]].append(row["expertise"])
    
    # sort by abstract_id
    human_expertise_by_question = dict(sorted(human_expertise_by_question.items(), key=lambda x: x[0]))

    # Plot len(human_expertise_by_question.keys()) subplots, each row has 5 subplots
    # in each subplot, plot the distribution of expertise for a question
    num_cols = 10
    num_rows = len(human_expertise_by_question.keys()) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(40, 40))
    top_20_expertise = []
    for i, (abstract_id, expertises) in enumerate(human_expertise_by_question.items()):
        row, col = i // num_cols, i % num_cols
        sns.kdeplot(expertises, ax=axes[row, col], color="skyblue")
        axes[row, col].set_title(f"Q{int(abstract_id)}")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].set_ylabel("")
        axes[row, col].set_xlim(-10, 110)
        
        # xtick only expertise of 50
        axes[row, col].set_xticks([1, 50, 100])

        # Plot vertical lines indicating top 20% expertise
        top_20_percentile = np.percentile(expertises, 80)
        axes[row, col].axvline(top_20_percentile, color="purple", linestyle="--", label="Top 20%")

        top_20_expertise.append(top_20_percentile)

    plt.tight_layout()
    plt.savefig(f"{figs_dir}/expertise_by_questions_distribution_{who}_abstracts.pdf")
    plt.close()

    # Plot the distribution of top 20% expertise
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.kdeplot(top_20_expertise, ax=ax, color="skyblue")
    ax.set_xlabel("Expertise")
    ax.set_ylabel("Density")
    ax.set_title(f"Top 20% Expertise\nAcross Test Cases")
    plt.tight_layout()
    plt.savefig(f"{figs_dir}/top_20_expertise_distribution_across_testcases_{who}_abstracts.pdf")


def plot_human_expertise_correlation_with_accuracy(use_human_abstract):
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"
    
    expertise = []
    correct = []
    for _, row in df.iterrows():
        if row["journal_section"].startswith(who):
            expertise.append(row["expertise"])
            correct.append(row["correct"])
    
    # Plot scatter plot of expertise vs. accuracy
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.scatterplot(x=expertise, y=correct, ax=ax)
    ax.set_xlabel("Expertise")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Expertise vs. Accuracy")

    # Spearman correlation
    rho, pval = stats.spearmanr(expertise, correct)
    print(f"Spearman correlation: {rho}, p-value: {pval}")
    plt.tight_layout()
    plt.savefig(f"{figs_dir}/expertise_vs_accuracy_{who}_abstracts.pdf")


def main():
    plot_human_expertise_distributions_by_subfields(args.use_human_abstract)
    # plot_human_expertise_correlation_with_accuracy(args.use_human_abstract)
    # plot_human_expertise_distributions_by_question(args.use_human_abstract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_human_abstract",
        type=argparse_helper.str2bool,
        default=True
    )

    model_results_dir = "model_results"
    human_results_dir = "human_results"
    testcases_dir = "testcases"
    figs_dir = "figs"
    args = parser.parse_args()
    main()
