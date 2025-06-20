# Functions to visualize results of grid search on UC-pairs

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter

def plot_reduced_heatmaps(pivot_ews, pivot_tws, patient_title="???"):
    """Plot two heatmaps with reduced UC range: one for the epoch-wise and one for the trial-wise score"""

    # Reduced range
    UC_mean_exponents = np.arange(-7, -2)  
    UC_mean_exponents = UC_mean_exponents[:4]
    UC_mean_range = 0.5 * (2.0 ** UC_mean_exponents)
    UC_cov_exponents = np.arange(-17, 2)  
    UC_cov_range = 0.5 * (2.0 ** UC_cov_exponents)

    # Create subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(11, 6), sharex=True, constrained_layout=True)

    # Plot first heatmap (Epoch-Wise Score)
    sns.heatmap(pivot_ews, annot=True, fmt=".2f", cmap="viridis", ax=ax1, cbar_kws={"label": "Mean AUC-ROC score"})
    ax1.set_yticks(np.arange(len(UC_mean_range)) + 0.5)
    ax1.set_yticklabels([fr"$0.5 \times 2^{{{y}}}$" for y in UC_mean_exponents], rotation=0, fontsize=10)
    cbar = ax1.collections[0].colorbar
    cbar.set_label("Mean AUC-ROC score", fontsize=12, rotation=90, labelpad=15)
    ax1.set_xlabel(" ")
    ax1.set_xticklabels([])
    ax1.tick_params(axis='x', bottom=False, top=False)
    ax1.set_ylabel("UC_mean", fontsize=12)
    ax1.set_title(f"Average Epoch-Wise Accuracy of {patient_title}", fontsize=14)

    # Plot second heatmap (Trial-Wise Score)
    sns.heatmap(pivot_tws, annot=True, fmt=".2f", cmap="magma", ax=ax2, cbar_kws={"label": "Predicted trials [%]"})
    ax2.set_yticks(np.arange(len(UC_mean_range)) + 0.5)
    ax2.set_yticklabels([fr"$0.5 \times 2^{{{y}}}$" for y in UC_mean_exponents], rotation=0, fontsize=10)
    ax2.set_xticks(np.arange(len(UC_cov_range)) + 0.5)
    ax2.set_xticklabels([fr"$0.5 \times 2^{{{x}}}$" for x in UC_cov_exponents], rotation=90, fontsize=10)
    cbar = ax2.collections[0].colorbar
    cbar.set_label("Predicted trials [%]", fontsize=12, rotation=90, labelpad=15)
    ax2.set_xlabel("UC_cov", fontsize=12)
    ax2.set_ylabel("UC_mean", fontsize=12)
    ax2.set_title(f"Average Trial-Wise Accuracy of {patient_title}", fontsize=14)

    plt.show()


def plot_versions_map_reduced():
    """Grid of versions and their corresponding UC-pair (only for the reduced UC ranges). 
    
    This is not part of the thesis results, but rather relevant if you want to work with the results and understand their structure
    Every UC-pair has a version number
    """

    # arrange ranges for UC-values
    UC_mean_exponents = np.arange(-7, -2)  
    UC_mean_exponents = UC_mean_exponents[:4]
    UC_mean_range = 0.5 * (2.0 ** UC_mean_exponents)
    UC_cov_exponents = np.arange(-17, 2)  
    UC_cov_range = 0.5 * (2.0 ** UC_cov_exponents)

    # arrange versions from 0 to 75
    myarray = np.arange(0,19)
    myarray = np.array([myarray, myarray+19, myarray+38, myarray+57])

    # plot versions against UC-pair
    plt.figure(figsize=(12,3))
    ax = sns.heatmap(myarray, annot=True, cbar=False,cmap="Blues", vmin=100, vmax=100)
    ax.set_yticks(np.arange(len(UC_mean_range)) + 0.5)
    ax.set_yticklabels([fr"$0.5 \times 2^{{{y}}}$" for y in UC_mean_exponents], rotation=0, fontsize=10)
    ax.set_xticks(np.arange(len(UC_cov_range)) + 0.5)
    ax.set_xticklabels([fr"$0.5 \times 2^{{{x}}}$" for x in UC_cov_exponents], rotation=90, fontsize=10)
    ax.set_title("Versions and corresponding UC-pair")
    ax.set_xlabel("UC_cov")
    ax.set_ylabel("UC_mean")
    plt.show()