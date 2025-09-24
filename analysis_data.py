import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fit_package 

   
def read_file_names(file_list_path):
    flist = []
    with open(file_list_path, 'r') as file:
        for line in file:
            rawfilename = line.rstrip('\n')  
            flist.append(rawfilename)  
            print(rawfilename)
    return flist

def merge_files(flist):
    all_data = pd.DataFrame()
    for file_path in flist:
        df_ = pd.read_hdf(file_path, key='winfo')
        if all_data.empty:
            all_data = df_.copy()  # 首次直接赋值
        else:
            try:
                all_data = pd.concat([all_data, df_], ignore_index=True)
            except ValueError as e:
                print(f"合并失败：文件 {file_path} 结构不一致")
                raise e
    return all_data


def find_pulse_width(waveform: np.ndarray, baseline: float, frac: float = 0.5, polarity: str = "positive"):
    """
    计算波形起始点、峰值点、下降沿 frac*高度对应点，并返回时间宽度
    不依赖 dt，结果以 sample index 为单位

    参数:
        waveform: np.ndarray, 波形
        baseline: float, 基线值
        frac: float, 阈值比例 (0.5=50%, 0.9=90%)
        polarity: "positive" 或 "negative"

    返回:
        start_idx: 起始点索引 (含插值)
        peak_idx: 峰值索引
        frac_idx: 下降沿 frac crossing 索引 (含插值)
        width: 宽度 (frac_idx - start_idx)
    """

    n = len(waveform)
    wf = waveform.copy()

    # 峰值
    if polarity == "positive":
        peak_idx = np.argmax(wf)
    else:
        peak_idx = np.argmin(wf)
    peak_val = wf[peak_idx]

    # 起始点：从 baseline 到 10%高度
    target_start = baseline + 0.1 * (peak_val - baseline)
    if polarity == "positive":
        candidates = np.where(wf[:peak_idx] >= target_start)[0]
    else:
        candidates = np.where(wf[:peak_idx] <= target_start)[0]

    if len(candidates) == 0:
        start_idx = 0
    else:
        i1 = candidates[0]
        i0 = i1 - 1 if i1 > 0 else i1
        # 线性插值
        x0, y0 = i0, wf[i0]
        x1, y1 = i1, wf[i1]
        start_idx = x0 + (target_start - y0) / (y1 - y0) * (x1 - x0)

    # 下降沿 frac
    target_frac = baseline + frac * (peak_val - baseline)
    if polarity == "positive":
        desc = np.where(wf[peak_idx:] <= target_frac)[0]
    else:
        desc = np.where(wf[peak_idx:] >= target_frac)[0]

    if len(desc) == 0:
        frac_idx = n - 1
    else:
        j1 = peak_idx + desc[0]
        j0 = j1 - 1 if j1 > peak_idx else j1
        # 插值
        x0, y0 = j0, wf[j0]
        x1, y1 = j1, wf[j1]
        frac_idx = x0 + (target_frac - y0) / (y1 - y0) * (x1 - x0)

    width = frac_idx - start_idx
    return start_idx, peak_idx, frac_idx, width


def Area_ratio(df):
    area_ratio = []
    area_ratio_err = []

    # 遍历每两个相邻的行
    for i in range(0, len(df) - 1, 2):
        area_0 = df.loc[i, 'area_mu']
        area_1 = df.loc[i+1, 'area_mu']
        err_0 = df.loc[i, 'area_err']
        err_1 = df.loc[i+1, 'area_err']

        # 取较小值除以较大值
        if area_0 <= area_1:
            ratio = area_0 / area_1
            # 误差传递
            ratio_err = ratio * np.sqrt((err_0 / area_0) ** 2 + (err_1 / area_1) ** 2)
        else:
            ratio = area_1 / area_0
            ratio_err = ratio * np.sqrt((err_1 / area_1) ** 2 + (err_0 / area_0) ** 2)

        area_ratio.append(ratio*100)
        area_ratio_err.append(ratio_err*100)
        print(ratio*100, ratio_err*100)
    return area_ratio, area_ratio_err


    
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
