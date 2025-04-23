import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("output/wandb_export_2025-04-09T23_28_18.840-04_00.csv")
# df = pd.read_csv("output/wandb_export_2025-04-10T00_40_25.064-04_00.csv")


def plot_acc_over_distance(df, metric_name, model_name):
    df = df[df["Name"] == model_name]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",  # Change to 'sans-serif', 'monospace', etc. as needed
            "font.size": 14,  # Adjust overall font size
            "axes.titlesize": 14,  # Title font size
            "axes.labelsize": 14,  # X and Y label font size
            "legend.fontsize": 14,  # Legend font size
            "xtick.labelsize": 14,  # X-tick labels font size
            "ytick.labelsize": 14,  # Y-tick labels font size
        }
    )

    distance_threshold_list = np.array(df.distance_threshold)

    if metric_name == "pw_cost":
        df[f"{metric_name}_all"] = df[f"{metric_name}_all"] / 23282
        df[f"{metric_name}_old"] = df[f"{metric_name}_old"] / 18163
        df[f"{metric_name}_new"] = df[f"{metric_name}_new"] / 5119

    acc_list = np.array(
        [
            list(df[f"{metric_name}_all"]),
            list(df[f"{metric_name}_old"]),
            list(df[f"{metric_name}_new"]),
        ]
    )

    # Sort the distance_threshold_list and reorder acc_list accordingly
    sort_idx = np.argsort(distance_threshold_list)
    distance_threshold_list = distance_threshold_list[sort_idx]
    acc_list = acc_list[:, sort_idx]

    evenly_spaced_x = np.linspace(
        0, len(distance_threshold_list) - 1, len(distance_threshold_list)
    )

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(["All", "Old", "New"]):
        plt.plot(distance_threshold_list, acc_list[i, :], marker="o", label=label)
        # Optional text annotations
        # for x, y in zip(evenly_spaced_x, acc_list[i, :]):
        #     plt.text(x, y + 0.1, f"{y:.2f}", ha="center", va="bottom", fontsize=14)

    plt.xticks(distance_threshold_list, distance_threshold_list)
    # plt.ylim(0, 1)

    plt.xlabel("Distance threshold")
    plt.ylabel(metric_name)
    plt.legend(loc="best", framealpha=0.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(f"figures/5_{metric_name}_{model_name}.png", bbox_inches="tight")
    plt.show()


def format_csv(df):
    print(df.columns)

    columns_drop = [
        "Runtime",
        "Created",
        "State",
        "k",
        "merge_cost",
        "split_cost",
        "standardize",
        "dataset.max_classes",
        "metric",
        "search_mode.name",
        "pca.n_components",
    ]

    df = df.drop(columns=columns_drop)

    df = df.applymap(lambda x: round(x, 4) if isinstance(x, (float, int)) else x)

    print(df)


# format_csv(df[df["Name"] == "ami_trap_resnet50_AMI-C"])
# for metric_name in ["ACC", "NMI", "ARI", "AMI", "pw_cost"]:
#     plot_acc_over_distance(df, metric_name, "ami_trap_resnet50_OI-C")
#     plot_acc_over_distance(df, metric_name, "ami_trap_resnet50_AMI-C")

metric_name = "pw_cost"
plot_acc_over_distance(df, metric_name, "ami_trap_resnet50_OI-C")
plot_acc_over_distance(df, metric_name, "ami_trap_resnet50_AMI-C")
