from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_histogram(dreamsim_list, l2_list, save_path, y_bins_max=250,
                   y_bins_slot=20, x_bins_max=1.8, x_bins_slot=0.2,
                   label_size=20):
    sns.set(style="darkgrid")
    bins = np.arange(0, x_bins_max, x_bins_slot)
    ybins = np.arange(0, y_bins_max, y_bins_slot)
    # Creating histogram
    plt.rcParams['font.size'] = 2

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.histplot(data=dreamsim_list, color="#008080", label="LPIPS", kde=True, bins=100)
    sns.histplot(data=l2_list, color="red",
                 label="R-LPIPS", kde=True, bins=100)
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=label_size)
    ax.set_xticks(bins)
    ax.set_yticks(ybins)
    ax.set(xlim=(0, x_bins_max), ylim=(0, y_bins_max))
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.show()


if __name__ == '__main__':
    dreamsim_list = torch.load('dists.pt')
    l2_list = torch.load('l2_lists.pt')
    plot_histogram(dreamsim_list, l2_list, 'fig.pdf')
