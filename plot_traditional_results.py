"""Plot test errors from the completed traditional-method runs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent/'data/result/traditional_methods'
    data = pd.read_csv(root/'comparison_metrics.csv')
    test = data.loc[data['set'] == 'test']
    regions = test.region.unique()
    fig, axes = plt.subplots(2, len(regions), figsize=(16, 8), layout='constrained', squeeze=False)
    for column, region in enumerate(regions):
        frame = test.loc[test.region == region]
        for row, metric in enumerate(('RMSE', 'MAE')):
            ax = axes[row, column]
            x = np.arange(3)
            for offset, source, color in ((-.19, 'sio', '#2878b5'), (.19, 'swot', '#ec8c45')):
                values = frame.loc[frame.source == source].set_index('method').loc[['GGM', 'SAS', 'Parker'], metric]
                bars = ax.bar(x+offset, values, width=.38, color=color, label=source.upper())
                ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
            ax.set_xticks(x, ['GGM', 'SAS-style', 'Parker-Oldenburg'])
            ax.set_ylabel(f'Test {metric} (m)')
            ax.set_title(region)
            ax.set_ylim(0, frame[metric].max()*1.18)
            ax.grid(axis='y', linestyle=':', alpha=.4)
            ax.set_axisbelow(True)
            ax.legend()
    for extension in ('png', 'pdf'):
        fig.savefig(root/f'test_error_comparison.{extension}', dpi=250)
    plt.close(fig)
    test.to_csv(root/'test_metrics.csv', index=False, encoding='utf-8-sig')
    print(test.to_string(index=False))


if __name__ == '__main__':
    main()
