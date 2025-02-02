
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fit_package 

def plot_waveform(wave, baseline,  xmin=0, xmax=150, ttt=888, area=100, pmt='LV2414',  ch='Anode', Save=False, file_tag='20240830_LED_run', title='LED Ch0 Waveform'):
    plt.figure()
    plt.step(np.arange(len(wave)), wave, where='mid')
    plt.axhline(y=baseline, color='b', linestyle='--', label='Baseline')  # 基线
    plt.axvline(x=100, color='r', linestyle='--', label='intg_st')  # 阈值
    plt.axvline(x=370, color='g', linestyle='--', label='intg_ed')  # 阈值
    # plt.axvline(x=400, color='r', linestyle='--', label='')  # 阈值
    # plt.axhline(y=baseline -20, color='r', linestyle='--', label='Threshold')  # 阈值
    # plt.scatter(st, wave[st], color='r', marker='o', label='start')  # 起始点
    # plt.scatter(ed, wave[ed], color='g', marker='o', label='end')  # 结束点
    # plt.scatter(lp, wave[lp], color='b', marker='o', label='minpoint')  # 最低点
    plt.title(r'Waveform of PMT {} {}, TTT={}, {:.2f}PE'.format(pmt,ch, ttt, area))
    plt.xlabel('Time [4 ns]')
    plt.ylabel('ADC Count')
    plt.legend()
    plt.xlim(xmin, xmax)
    if Save==True:
        plt.savefig(r'./figs/{}_{}.png'.format(file_tag, title,dpi=300))
    elif Save==False:
        plt.show()
    # plt.show()
    
def plot_fit_histgram_vs_Gaussion(array, nbins, left_edge, right_edge, p0=[1.e4, 100, 10],file_tag='20240830_LED_run',xlabel='Ch0 Area', title='LED Ch0 Area', Save=False):
    hist, bins_edges = np.histogram(array, bins= nbins, range=(left_edge, right_edge))
    bins = (bins_edges[:-1] + bins_edges[1:])/2
    popt, _ = curve_fit(fit_package.gaussian, bins, hist, p0=p0)
    x_fit = np.linspace(np.min(bins), np.max(bins), 1000)
    y_fit = fit_package.gaussian(x_fit, *popt)
    plt.plot(x_fit, y_fit, label=r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(popt[1], popt[2]))
    plt.hist(array, bins=nbins, range=(left_edge, right_edge),  color='black', density=False, alpha=0.5, label=xlabel)
    plt.xlabel(r'{} '.format(xlabel))
    plt.ylabel('Entries')
    plt.title(r'{} {}'.format(file_tag, title))
    plt.legend()
    if Save==True:
        plt.savefig(r'./figs/{}_{}.png'.format(file_tag, title,dpi=300))
    elif Save==False:
        plt.show()
    print(r'Fit: mu= {:.2f}, sigma ={:.2f}'.format(popt[1], popt[2]))
    return popt[1], popt[2]


def landau_distribution(xdata, mu,sigma, A):
    landau = lambda t, mu, sigma, xdata, A : A*np.exp(-t)*np.cos(t*(xdata -mu)/sigma + 2*t/np.pi *np.log(t/sigma) ) / (sigma *  np.pi)
    integral, error = integrate.quad(landau, 0, np.inf, args=(mu,sigma,xdata, A))
    return integral

def landau_distribution_array(xdata,mu, sigma, A):
    return np.array([landau_distribution(x, mu,sigma, A) for x in xdata])


def exponential_decay(t, N0, lambda_):
    return  N0* np.exp(-lambda_ * t)
def fit_decay_data(time_data, count_data):
    # Initial guess for the parameters N0 and lambda
    initial_guess = [7600, 0.2]
    
    # Curve fitting
    popt, pcov = curve_fit(exponential_decay, time_data, count_data, p0=initial_guess)    
    N0, lambda_ = popt
    return N0, lambda_, pcov


def find_threshold_points(waveform, baseline, threshold, negative_pulse=True, start_index=0, end_index=0):
    threshold_points = []
    for i in range(start_index, len(waveform[:end_index])):
        if negative_pulse:
            if waveform[i] < baseline -threshold:
                threshold_points.append(i)
        else:
            if waveform[i] > baseline +threshold:
                threshold_points.append(i)    
    st = threshold_points[0]
    ed = threshold_points[-1]
    return st, ed


def calculate_wf_mean_std(file, threshold=100, start_index=1000, Channel='Anode'):  
    import pandas as pd
    import glob
    #####################################################################
    #### load all data by DataFrame format          #####################
    #####################################################################
    h5_files_pattern = r'{}*.h5py'.format(file.split('raw_')[0])
    print(h5_files_pattern)
    h5_files = glob.glob(h5_files_pattern)
    df = pd.DataFrame()  #### comine all data
    for files in h5_files:
        _df = pd.read_hdf(files, key='winfo')
        df = pd.concat([df, _df], ignore_index=True)
    ######################################################################
    #### select Ch==0, first Anode waveform to calculate st,ed index #####
    ######################################################################
    index = None 
    for i in range(3):
        if df.Ch[i] != 0:
            continue
        else:
            index = i
        index = i  
    if 'Wave' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Wave' column.")        
    st, ed = find_threshold_points(df.Wave[:][index], df.Baseline[:][index], negative_pulse=True, threshold=threshold, start_index=start_index, end_index=len(df.Wave[:][index]) )
    #########################################################################
    #### Calculate wf mean and std for selected Channel, anode or dynode ####
    #########################################################################
    if Channel == 'Anode':
        ch_selec = df.Ch == 0
    elif Channel == 'Dynode':
        ch_selec = df.Ch == 2
    df = df[ch_selec]
    wave_array = df['Wave'].values 
    data_array = np.zeros((len(wave_array), len(wave_array[0])))
    for i in range(len(wave_array)):
        for j in range(len(wave_array[0])):
            data_array[i][j] = wave_array[i][j]        
    mean_array = np.mean(data_array, axis=0)
    std_array = np.std(data_array, axis=0)
    data_array = None
    return mean_array[st-50:ed+50], std_array[st-50:ed+50]



def calculate_wf_mean_std(file, threshold=100, start_index=1000, Channel='Anode'):  
    import pandas as pd
    import glob
    #####################################################################
    #### load all data by DataFrame format          #####################
    #####################################################################
    h5_files_pattern = r'{}*.h5py'.format(file.split('raw_')[0])
    print(h5_files_pattern)
    h5_files = glob.glob(h5_files_pattern)
    df = pd.DataFrame()  #### comine all data
    for files in h5_files:
        _df = pd.read_hdf(files, key='winfo')
        df = pd.concat([df, _df], ignore_index=True)
    ######################################################################
    #### select Ch==0, first Anode waveform to calculate st,ed index #####
    ######################################################################
    index = None 
    for i in range(3):
        if df.Ch[i] != 0:
            continue
        else:
            index = i
        index = i  
    if 'Wave' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Wave' column.")        
    st, ed = find_threshold_points(df.Wave[:][index], df.Baseline[:][index], negative_pulse=True, threshold=threshold, start_index=start_index, end_index=len(df.Wave[:][index]) )
    #########################################################################
    #### Calculate wf mean and std for selected Channel, anode or dynode ####
    #########################################################################
    if Channel == 'Anode':
        ch_selec = df.Ch == 0
    elif Channel == 'Dynode':
        ch_selec = df.Ch == 2
    df = df[ch_selec]
    wave_array = df['Wave'].values 
    data_array = np.zeros((len(wave_array), len(wave_array[0])))
    for i in range(len(wave_array)):
        for j in range(len(wave_array[0])):
            data_array[i][j] = wave_array[i][j]        
    mean_array = np.mean(data_array, axis=0)
    std_array = np.std(data_array, axis=0)
    data_array = None
    return mean_array[st-50:ed+50], std_array[st-50:ed+50]



def calculate_wf_mean_std_s2(file, threshold=100,  Channel='Anode'):  
    import pandas as pd
    import glob
    #####################################################################
    #### load all data by DataFrame format          #####################
    #####################################################################
    h5_files_pattern = r'{}*.h5py'.format(file.split('raw_')[0])
    print(h5_files_pattern)
    h5_files = glob.glob(h5_files_pattern)
    df = pd.DataFrame()  #### comine all data
    for files in h5_files:
        _df = pd.read_hdf(files, key='winfo')
        df = pd.concat([df, _df], ignore_index=True)
    ######################################################################
    #### select Ch==0, first Anode waveform to calculate st,ed index #####
    ######################################################################
    index = None 
    for i in range(3):
        if df.Ch[i] != 0:
            continue
        else:
            index = i
        index = i  
    if 'Wave' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Wave' column.")        
    # st, ed = find_threshold_points(df.Wave[:][index], df.Baseline[:][index], negative_pulse=True, threshold=threshold, start_index=start_index, end_index=len(df.Wave[:][index]) )
    #########################################################################
    #### Calculate wf mean and std for selected Channel, anode or dynode ####
    #########################################################################
    if Channel == 'Anode':
        ch_selec = df.Ch == 0
    elif Channel == 'Dynode':
        ch_selec = df.Ch == 2
    df = df[ch_selec]
    wave_array = df['Wave'].values 
    data_array = np.zeros((len(wave_array), len(wave_array[0])))
    for i in range(len(wave_array)):
        for j in range(len(wave_array[0])):
            data_array[i][j] = wave_array[i][j]        
    mean_array = np.mean(data_array, axis=0)
    std_array = np.std(data_array, axis=0)
    data_array = None
    return mean_array[50:400], std_array[50:400]


def calculate_fullwf_mean_std(file, threshold=100,  Channel='Anode'):  
    import pandas as pd
    import glob
    #####################################################################
    #### load all data by DataFrame format          #####################
    #####################################################################
    h5_files_pattern = r'{}*.h5py'.format(file.split('raw_')[0])
    print(h5_files_pattern)
    
    h5_files = glob.glob(h5_files_pattern)
    df = pd.DataFrame()  #### comine all data
    for files in h5_files:
        _df = pd.read_hdf(files, key='winfo')
        df = pd.concat([df, _df], ignore_index=True)
    ######################################################################
    #### select Ch==0, first Anode waveform to calculate st,ed index #####
    ######################################################################
    index = None 
    for i in range(3):
        if df.Ch[i] != 0:
            continue
        else:
            index = i
        index = i  
    if 'Wave' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Wave' column.")        
    # st, ed = find_threshold_points(df.Wave[:][index], df.Baseline[:][index], negative_pulse=True, threshold=threshold, start_index=start_index, end_index=len(df.Wave[:][index]) )
    #########################################################################
    #### Calculate wf mean and std for selected Channel, anode or dynode ####
    #########################################################################
    if Channel == 'Anode':
        ch_selec = df.Ch == 0
    elif Channel == 'Filter':
        ch_selec = df.Ch == 1
    elif Channel == 'Dynode':
        ch_selec = df.Ch == 2
    df = df[ch_selec]
    wave_array = df['Wave'].values 
    data_array = np.zeros((len(wave_array), len(wave_array[0])))
    for i in range(len(wave_array)-1):
        for j in range(len(wave_array[0])):
            data_array[i][j] = wave_array[i][j]        
    mean_array = np.mean(data_array, axis=0)
    std_array = np.std(data_array, axis=0)
    data_array = None
    return mean_array, std_array
