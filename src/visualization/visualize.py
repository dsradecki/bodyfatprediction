import matplotlib.pyplot as plt
import numpy as np

def show_correlogram(data):

    corr = data.corr()
    #corr = (corr > 0.5) * 1
    fig = plt.figure(figsize = (15, 15))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=14)

    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)

    for y in range(corr.shape[0]):
        for x in range(corr.shape[1]):
            plt.text(x, y, '%.2f' % corr.iloc[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
    plt.show()

def boxplots(data):

    fig, axs = plt.subplots(5, 3, figsize = (10, 15))

    for idx, ax in enumerate(axs.reshape(-1)):

        ax.boxplot(data.iloc[:, idx])
        ax.set_xticklabels([])
        ax.set_title(data.columns[idx])
        #ax.axis('auto')
        #ax.set_autoscale_on(True)
        ax.margins(0.2)

        ax.grid(linestyle = '-')