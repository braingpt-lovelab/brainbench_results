import os
import copy
import argparse
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import subfields
from utils import model_list
from utils import argparse_helper
from utils import scorer


def get_llm_acc_subfields(use_human_abstract=True):
    if use_human_abstract:
        type_of_abstract = "human_abstracts"
        human_abstracts_fpath = f"{testcases_dir}/BrainBench_Human_v0.1.csv"
        df = pd.read_csv(human_abstracts_fpath)
        journal_column_name = "journal_section"
    else:
        type_of_abstract = "llm_abstracts"
        llm_abstracts_fpath = f"{testcases_dir}/BrainBench_GPT-4_v0.1.csv"
        df = pd.read_csv(llm_abstracts_fpath)
        journal_column_name = "journal_section"

    llms = copy.deepcopy(model_list.llms)

    # e.g.,
    # {
    #     "subfield1": {
    #         "llm1": {"acc": 0.5, "color": "red", "alpha": 0.5, "hatch": "//"},
    #         "llm2": {"acc": 0.5, "color": "red", "alpha": 0.5, "hatch": "//"},
    #         ...
    #     },
    all_subfields_llms = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict()
        )
    )
    for subfield in subfields.subfield_names:
        for llm_family in llms.keys():
            for llm in llms[llm_family]:
                results_dir = os.path.join(
                    f"{model_results_dir}/{llm.replace('/', '--')}/{type_of_abstract}"
                )
                PPL_fname = "PPL_A_and_B"
                label_fname = "labels"
                PPL_A_and_B = np.load(f"{results_dir}/{PPL_fname}.npy")
                labels = np.load(f"{results_dir}/{label_fname}.npy")

                results_by_subfield = collections.defaultdict(list)
                labels_by_subfield = collections.defaultdict(list)
                # For each sample, need to iterate through df to find their subfield
                # and record the corresponding PPL for A and B, and label.
                for j, (ppl_A, ppl_B) in enumerate(PPL_A_and_B):
                    row = df.iloc[j]
                    if row[journal_column_name] == subfield:
                        results_by_subfield[subfield].append([ppl_A, ppl_B])
                        labels_by_subfield[subfield].append(labels[j])

                # Compute acc by category
                all_subfields_llms[subfield][llm]['acc'] = \
                    scorer.acc(
                        np.array(results_by_subfield[subfield]), 
                        np.array(labels_by_subfield[subfield])
                )

                # Get color, alpha, hatch
                all_subfields_llms[subfield][llm]['color'] = \
                    llms[llm_family][llm]['color']
                all_subfields_llms[subfield][llm]['alpha'] = \
                    llms[llm_family][llm]['alpha']
                all_subfields_llms[subfield][llm]['hatch'] = \
                    llms[llm_family][llm]['hatch']
    
    return all_subfields_llms


def get_human_acc_subfields(use_human_abstract):
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"

    all_subfields_human = collections.defaultdict(
        lambda: collections.defaultdict()
    )
    for subfield in subfields.subfield_names:
        correct = 0
        total = 0
        for _, row in df.iterrows():
            if row["journal_section"].startswith(who) and row["journal_section"].endswith(subfield):
                correct += row["correct"]
                total += 1
        all_subfields_human[subfield] = (correct / total)
    return all_subfields_human


def get_subfield_proportions(use_human_abstract):
    if use_human_abstract:
        human_abstracts_fpath = f"{testcases_dir}/BrainBench_Human_v0.1.csv"
        df = pd.read_csv(human_abstracts_fpath)
        journal_column_name = "journal_section"
    else:
        llm_abstracts_fpath = f"{testcases_dir}/BrainBench_GPT-4_v0.1.csv"
        df = pd.read_csv(llm_abstracts_fpath)
        journal_column_name = "journal_section"
    
    subfield_proportions = collections.defaultdict(
        lambda: collections.defaultdict()
    )
    for subfield in subfields.subfield_names:
        total = 0
        for _, row in df.iterrows():
            if row[journal_column_name] == subfield:
                total += 1
        subfield_proportions[subfield] = total / len(df)
    return subfield_proportions


def radar_plot(use_human_abstract):
    from math import pi

    subfield_abbreviations = {
        "Development/Plasticity/Repair": "Dev/Plast/Rep",
        "Behavioral/Cognitive": "Behav/Cog",
        "Cellular/Molecular": "Cell/Mol",
        "Neurobiology of Disease": "Neuro Dis",
        "Systems/Circuits": "Sys/Circ"
    }
    
    all_subfields_human = get_human_acc_subfields(use_human_abstract)
    all_subfields_llms = get_llm_acc_subfields(use_human_abstract)

    # Initialize figure for comparing human and average LLM accuracy
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    
    # Radar chart setup
    num_variables = len(subfield_abbreviations)
    angles = [n / float(num_variables) * 2 * pi for n in range(num_variables)]
    angles += angles[:1]  # Complete the loop
    
    # Data for radar chart
    llm_accuracies = []
    human_accuracies = []
    
    # for subfield in subfield_abbreviations.keys():
    for subfield in subfields.subfield_names:
        llm_accuracies.append(np.mean([all_subfields_llms[subfield][llm]['acc'] for llm_family in model_list.llms for llm in model_list.llms[llm_family]]))
        human_accuracies.append(all_subfields_human[subfield])
    
    # Complete the loop for plotting
    llm_accuracies += llm_accuracies[:1]
    human_accuracies += human_accuracies[:1]

    # Plotting the subfield proportion figure as pie chart
    subfield_proportions = get_subfield_proportions(use_human_abstract)

    ax1.pie(
        [subfield_proportions[sf] for sf in subfields.subfield_names][::-1], 
        labels=[subfield_abbreviations[sf] for sf in subfields.subfield_names][::-1],
        autopct='%1.1f%%',
        startangle=90,
        colors=["#F5CEC7", "#E79796", "#FFC988", "#FFB284", "#C6C09C"],
        textprops={'fontsize': 16},
        wedgeprops={"edgecolor": "black", "linewidth": 0.5}
    )

    # Second subplot: Radar chart
    ax2 = fig.add_subplot(122, polar=True)
    ax2.set_ylim(0.5, 1)
    ax2.set_theta_offset(pi / 2)
    ax2.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], [subfield_abbreviations[sf] for sf in subfields.subfield_names], color='k', size=16)
    
    ax2.plot(angles, llm_accuracies, linewidth=1, linestyle='solid')
    ax2.fill(angles, llm_accuracies, 'o', alpha=0.3, label='Average LLM')
    
    ax2.plot(angles, human_accuracies, linewidth=1, linestyle='solid')
    ax2.fill(angles, human_accuracies, 'b', alpha=0.3, label='Human')
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.8, -0.05))
    
    plt.tight_layout()

    # Save the comparison figure
    if use_human_abstract:
        plt.savefig("figs/comparison_human_llm_accuracy_subfields_human_abstract.pdf")
    else:
        plt.savefig("figs/comparison_human_llm_accuracy_subfields_llm_abstract.pdf")
    plt.close(fig)


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
    radar_plot(use_human_abstract=parser.parse_args().use_human_abstract)
