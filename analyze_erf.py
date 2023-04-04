import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns
#   Set figure parameters
large = 24; med = 24; small = 24
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
plt.rc('font', **{'family': 'Times New Roman'})
plt.rcParams['axes.unicode_minus'] = False


parser = argparse.ArgumentParser('Script for analyzing the ERF', add_help=False)
# parser.add_argument('--source', default='cprr_temp_1558.npy', type=str, help='path to the contribution score matrix (.npy file)')
# parser.add_argument('--heatmap_save', default='cprr_heatmap_1558.png', type=str, help='where to save the heatmap')
parser.add_argument('--source', default='basel_temp_1558.npy', type=str, help='path to the contribution score matrix (.npy file)')
parser.add_argument('--heatmap_save', default='basel_heatmap_1558.png', type=str, help='where to save the heatmap')
# parser.add_argument('--source', default='corr_lpn.npy', type=str, help='path to the contribution score matrix (.npy file)')
# parser.add_argument('--heatmap_save', default='corr_lpn_heatmap_1.png', type=str, help='where to save the heatmap')
args = parser.parse_args()

import numpy as np
#RdYlGn   10, 10.75
def heatmap(data, camp='viridis', figsize=(10.75, 10), ax=None, save_path=None):
    print('yes')
    plt.figure(figsize=figsize, dpi=40)
    print('yes --------')
    ax = sns.heatmap(data,
                xticklabels=False,
                yticklabels=False, cmap=camp,
                center=0, annot=False, ax=ax, cbar=False, annot_kws={"size": 24}, fmt='.2f')
    #   =========================== Add a **nicer** colorbar on top of the figure. Works for matplotlib 3.3. For later versions, use matplotlib.colorbar
    #   =========================== or you may simply ignore these and set cbar=True in the heatmap function above.
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.colorbar import colorbar
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='5%', pad='2%')
    colorbar(ax.get_children()[0], cax=cax, orientation='vertical')  #vertical
    cax.xaxis.set_ticks_position('top')
    #   ================================================================
    #   ================================================================
    plt.savefig(save_path, bbox_inches='tight')


def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, ((i * 2 + 1) / 1024) ** 2
    return None

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def analyze_erf(args):
    data = np.load(args.source)
    print(np.max(data))
    print(np.min(data))
    data = np.log(data + 1)       #   the scores differ in magnitude. take the logarithm for better readability
    print(np.max(data))
    print(np.min(data))
    # data = data / np.max(data)     #   rescale to [0,1] for the comparability among models
    data = tanh(data)
    print('======================= the high-contribution area ratio =====================')
    # for thresh in [0.2, 0.3, 0.5, 0.99]:   #[0.2, 0.3, 0.5, 0.99]
    #     side_length, area_ratio = get_rectangle(data, thresh)
    #     print('thresh, rectangle side length, area ratio: ', thresh, side_length, area_ratio)
    heatmap(data, save_path=args.heatmap_save)
    print('heatmap saved at ', args.heatmap_save)


if __name__ == '__main__':
    analyze_erf(args)