import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def plot_normal(target_band=5.0, storing_folder='plots'):
    # Check if preprocessed_dataset data is created
    if not os.path.exists(storing_folder):
        os.mkdir(storing_folder)

    file_names = ['samples_office_None_1', 'samples_lab_None_1', 'samples_chamber_None_8']
    labels = ['Office', 'Laboratory', 'Noise Floor', 'Jamming 5200MHz (60cm, 0dBm)', 'Jamming 5200MHz (20cm, 10dBm)']

    final_data = {}
    for file_name in file_names:
        final_data[file_name] = {}

    dfs = []
    for file_name in file_names:
        df = pd.read_csv(file_name + '.csv')
        # Ignore sometimes corrupted first row and filter freq
        df = df.iloc[1:, :]
        df[df['snr'] < -100] = np.NaN
        df[df['snr'] > 100] = np.NaN
        df.fillna(df['snr'].mean(), inplace=True)

        if target_band == 2.4:
            df = df.loc[df['freq1'] <= 3000]
        else:
            df = df.loc[df['freq1'] > 5000]
            df = df.loc[df['freq1'] < 5400]

        dfs.append(df)

    frequencies = set(dfs[0]['freq1'])
    frequencies = sorted(list(frequencies))
    for i, df in enumerate(dfs):
        # plt.scatter(df['freq1'], df['snr'], alpha=0.5, label=file_names[i])
        temp = {}
        for freq in frequencies:
            temp[freq] = []
        for freq, value in zip(df['freq1'], df['snr']):
            temp[freq].append(value)
        final_data[file_names[i]] = temp

    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(3.5)

    boxes = []
    # space = [-5, -2.5, 0, 2.5, 5]
    space = [-5, 0, 5]

    for i, file_name in enumerate(file_names):
        file_data = list(final_data[file_name].values())
        width = 4
        bp = ax.boxplot(file_data, 0, '', positions=np.array(frequencies) + space[i], widths=width,
                        patch_artist=True)

        plt.setp(bp['whiskers'], color=px.colors.qualitative.G10[i], linestyle='-')
        plt.setp(bp['boxes'], color=px.colors.qualitative.G10[i])
        plt.setp(bp['fliers'], color=px.colors.qualitative.G10[i])
        plt.setp(bp['means'], color=px.colors.qualitative.G10[i])
        plt.setp(bp['medians'], color=px.colors.qualitative.G10[i])
        plt.setp(bp['caps'], color=px.colors.qualitative.G10[i])
        boxes.append(bp)

    bxs = [boxes[i]['boxes'][0] for i in range(len(file_names))]
    fln = [labels[i] for i in range(len(file_names))]
    ax.legend(bxs, fln, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.225))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Received Power (dBm)')
    plt.xticks(np.floor(np.array(frequencies)))
    plt.xlim(min(frequencies) - 10, max(frequencies) + 10)
    plt.ylim(-13.5, 9.75)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/plot1.pdf', dpi=300)


def plot_jamming(target_band=5.0, storing_folder='plots'):
    # Check if preprocessed_dataset data is created
    if not os.path.exists(storing_folder):
        os.mkdir(storing_folder)

    file_names = ['samples_chamber_5200MHz_20cm_10dBm_5', 'samples_chamber_5280MHz_60cm_0dBm_2']
    labels = ['Jam. (20cm, 10dBm)', 'Jam. (60cm, 0dBm)']

    final_data = {}
    for file_name in file_names:
        final_data[file_name] = {}

    dfs = []
    for file_name in file_names:
        df = pd.read_csv(file_name + '.csv')
        # Ignore sometimes corrupted first row and filter freq
        df = df.iloc[1:, :]
        df[df['snr'] < -100] = np.NaN
        df[df['snr'] > 100] = np.NaN
        df.fillna(df['snr'].mean(), inplace=True)

        if target_band == 2.4:
            df = df.loc[df['freq1'] <= 3000]
        else:
            df = df.loc[df['freq1'] > 5000]
            df = df.loc[df['freq1'] < 5400]

        dfs.append(df)

    frequencies = set(dfs[0]['freq1'])
    frequencies = sorted(list(frequencies))
    for i, df in enumerate(dfs):
        # plt.scatter(df['freq1'], df['snr'], alpha=0.5, label=file_names[i])
        temp = {}
        for freq in frequencies:
            temp[freq] = []
        for freq, value in zip(df['freq1'], df['snr']):
            temp[freq].append(value)
        final_data[file_names[i]] = temp

    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(3.5)

    boxes = []
    # space = [-5, -2.5, 0, 2.5, 5]
    space = [-2.5, 2.5]

    for i, file_name in enumerate(file_names):
        file_data = list(final_data[file_name].values())
        width = 4
        bp = ax.boxplot(file_data, 0, '', positions=np.array(frequencies) + space[i], widths=width,
                        patch_artist=True)

        plt.setp(bp['whiskers'], color=px.colors.qualitative.G10[i], linestyle='-')
        plt.setp(bp['boxes'], color=px.colors.qualitative.G10[i])
        plt.setp(bp['fliers'], color=px.colors.qualitative.G10[i])
        plt.setp(bp['means'], color=px.colors.qualitative.G10[i])
        plt.setp(bp['medians'], color=px.colors.qualitative.G10[i])
        plt.setp(bp['caps'], color=px.colors.qualitative.G10[i])
        boxes.append(bp)

    bxs = [boxes[i]['boxes'][0] for i in range(len(file_names))]
    fln = [labels[i] for i in range(len(file_names))]
    ax.legend(bxs, fln, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.225))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Received Power (dBm)')
    plt.xticks(np.floor(np.array(frequencies)))
    plt.xlim(min(frequencies) - 10, max(frequencies) + 10)
    plt.ylim(-12, 31)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/plot2.pdf', dpi=300)


def main():
    np.random.seed(0)
    plt.style.use('classic')
    plot_normal()
    plot_jamming()


if __name__ == '__main__':
    main()
