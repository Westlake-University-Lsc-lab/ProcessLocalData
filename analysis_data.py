
from landaupy import landau
import numpy as np
import matplotlib.pyplot as plt
    
def landau_fit(xdata, location, scale, A):
    mu = location
    sigma = scale
    return A*landau.pdf(xdata, mu, sigma)

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def load_runlist_to_numpy(runlist_path):
    with open(runlist_path, 'r') as file:
        runlist = file.readlines()
    runlist = [run.strip() for run in runlist]
    return np.array(runlist)
    
def load_runlist_to_numpy(runlist_path):
    with open(runlist_path, 'r') as file:
        runlist = file.readlines()
    runlist = [run.strip() for run in runlist]
    return np.array(runlist)

def plot_waveform(wave, baseline, st, ed, lp, xmin=0, xmax=150):
    plt.figure()
    plt.step(np.arange(len(wave)), wave, where='mid')
    plt.axhline(y=baseline, color='b', linestyle='--', label='Baseline')  # 基线
    plt.axhline(y=baseline -20, color='r', linestyle='--', label='Threshold')  # 阈值
    plt.scatter(st, wave[st], color='r', marker='o', label='start')  # 起始点
    plt.scatter(ed, wave[ed], color='g', marker='o', label='end')  # 结束点
    plt.scatter(lp, wave[lp], color='b', marker='o', label='minpoint')  # 最低点
    plt.xlabel('Time [4 ns]')
    plt.ylabel('ADC Count')
    plt.legend()
    plt.xlim(xmin, xmax)
    plt.show()