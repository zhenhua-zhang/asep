import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot


def load_feature_importance(**pkl_file_pool):
    fi_dtfm = None
    for pkl_name, pkl_file in pkl_file_pool.items():
        with open(pkl_file, "rb") as ipf:
            fi = pickle.load(ipf)
            fi_mean = {x: np.mean(fi[x]) for x in fi}

        if fi_dtfm is None:
            fi_dtfm = pd.DataFrame(fi_mean, index=pkl_name)
        else:
            fi_dtfm.loc[pkl_name] = fi_mean
        
        fi_dtfm = fi_dtfm.sort_values(by=pkl_name)
        fi_dtfm.loc["{}_rank".format(pkl_name)] = range(len(fi_mean))

    return fi_dtfm


def draw_slope_graph(fi_dtfm):
    _, n_fi = fi_dtfm.shape
    ax = host_subplot(111)
    ax.scatter(fi_dtfm.loc["gtex_rank"], [0]*79)
    ax.scatter(fi_dtfm.loc["bios_rank"], [1]*79)

    for x1, x2 in zip(fi_dtfm.loc["gtex_rank"], fi_dtfm.loc["bios_rank"]):
        ax.plot([x1, x2], [0, 1])

    ax_ = ax.twin()
    ax_.set_xticks(range(n_fi))
    ax_.set_xticklabels(fi_dtfm.columns, rotation=-45, ha="right", rotation_mode="anchor")

    labels = fi_dtfm.loc["gtex_rank"].sort_values().index
    ax.set_xticks(range(n_fi))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")

    ax.axis["right"].set_visible(False)
    ax.axis["left"].set_visible(False)
    ax_.axis["right"].set_visible(False)
    ax_.axis["left"].set_visible(False)
    ax_.axis["bottom"].set_visible(False)

    fig = ax.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(5)
    plt.tight_layout()
    fig.savefig("feature_importanc_shift.pdf")
    plt.close()
