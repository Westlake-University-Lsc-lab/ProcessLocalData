
from landaupy import landau
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
    
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

def plot_waveform(wave, baseline, st, ed, lp, xmin=0, xmax=150, ttt=888, area=100, pmt='LV2414',  ch='Anode'):
    plt.figure()
    plt.step(np.arange(len(wave)), wave, where='mid')
    plt.axhline(y=baseline, color='b', linestyle='--', label='Baseline')  # 基线
    plt.axhline(y=baseline -20, color='r', linestyle='--', label='Threshold')  # 阈值
    plt.scatter(st, wave[st], color='r', marker='o', label='start')  # 起始点
    plt.scatter(ed, wave[ed], color='g', marker='o', label='end')  # 结束点
    plt.scatter(lp, wave[lp], color='b', marker='o', label='minpoint')  # 最低点
    plt.title(r'Waveform of PMT {} {}, TTT={}, {:.2f}PE'.format(pmt,ch, ttt, area))
    plt.xlabel('Time [4 ns]')
    plt.ylabel('ADC Count')
    plt.legend()
    plt.xlim(xmin, xmax)
    plt.show()
    
def plot_fit_histgram_vs_Gaussion(array, nbins, left_edge, right_edge, p0=[1.e4, 100, 10],file_tag='20240830_LED_run',xlabel='Ch0 Area', title='LED Ch0 Area' ):
    hist, bins_edges = np.histogram(array, bins= nbins, range=(left_edge, right_edge))
    bins = (bins_edges[:-1] + bins_edges[1:])/2
    popt, _ = curve_fit(gaussian, bins, hist, p0=p0)
    x_fit = np.linspace(np.min(bins), np.max(bins), 1000)
    y_fit = gaussian(x_fit, *popt)
    plt.plot(x_fit, y_fit, label=r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(popt[1], popt[2]))
    plt.hist(array, bins=nbins, range=(left_edge, right_edge),  color='black', density=False, alpha=0.5, label=xlabel)
    plt.xlabel(r'{} [PE]'.format(xlabel))
    plt.ylabel('Entries')
    plt.title(r'{} {}'.format(file_tag, title))
    plt.legend()
    print(r'Fit: mu= {:.2f}, sigma ={:.2f}'.format(popt[1], popt[2]))
    return popt[1], popt[2]