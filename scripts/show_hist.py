from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_histogram(data_list, label_list, save_path, color_list=["#008080", "#FCA592", "yellow", "lightblue"],
                   y_bins_max=18, y_bins_slot=3, x_bins_max=1.4, x_bins_slot=0.2, label_size=20):
    sns.set(style="darkgrid")
    bins = np.arange(0, x_bins_max, x_bins_slot)
    ybins = np.arange(0, y_bins_max, y_bins_slot)
    plt.rcParams['font.size'] = 2

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, data in enumerate(data_list):
        sns.histplot(data=data, color=color_list[idx], label=label_list[idx], kde=True, bins=100)

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
    data_list = list()
    data_list.append(torch.load('dist_dir/dino_list.pt', map_location=torch.device('cpu')))
    data_list.append(torch.load('dist_dir/clip_list.pt', map_location=torch.device('cpu')))
    plot_histogram(data_list=data_list, label_list=['DINO', 'CLIP'], save_path='fig.pdf')
