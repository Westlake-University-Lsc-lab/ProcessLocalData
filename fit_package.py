import pandas as pd
import numpy as np
import analysis_data
import sys
def check_type(input_numb):
    try:
        int_value = int(input_numb)
        print(r'input {} is int type'.format(input_numb))
        return True
    except ValueError:
        if isinstance(input_numb, str):
            print("intput is str")
            return False
        elif isinstance(input_numb, int):
            print(r'input {} is int typ'.format(input_numb))
            return True
        else:
            print("Please input a valid parameter type.")

def ftag(file):
    file_tag = file.split('outnpy/')[1].split('.h5py')[0]
    # fdate = file.split('LED_')[0].split('lv2414_')[1]
    # led_config = file.split('LED_')[1].split('.h5py')[0]
    # file_tag = fdate + led_config
    return file_tag

def read_hdfile(file):
    import glob
    h5_files_pattern = r'{}*.h5py'.format(file.split('raw_')[0])
    print(h5_files_pattern)
    h5_files = glob.glob(h5_files_pattern)
    df = pd.DataFrame()  
    for files in h5_files:
        _df = pd.read_hdf(files, key='winfo')
        df = pd.concat([df, _df], ignore_index=True)
    return df

def plot_example_waveform(df,st=0,ed=500):    
    index = None 
    for i in range(3):
        if df.Ch[i] != 0:
            continue
        else:
            index = i
        index = i  
    if 'Wave' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Wave' column.")
    channel = df.Ch[:][index]
    wave = df.Wave[:][index]
    baseline = df.Baseline[:][index]    
    ttt = df.TTT[:][index]  
    area = df.Area_S2[:][index]   
    if channel == 0:
        pmt = 'LV1414'
    elif channel == 1:
        pmt = 'LV2415'
    elif channel == 2:
        pmt = 'LV2414 Dynode'
    analysis_data.plot_waveform(wave,baseline,st,ed,pmt=pmt,ch=r'Ch={}'.format(channel),ttt=ttt,area=area)
    
        
def fit_single_channel(df, channel, ftag):
    if check_type(channel) ==True:
        pass
    else:
        print("Please input a valid parameter type with interger value.")
        sys.exit()   
    area = df.Area[df.Ch == channel].astype(np.float64).to_numpy()
    if channel == 2:
        area = -area
        mean = np.mean(area)
        std = np.std(area)
        redge = np.max(area)
        ledge = np.min(area)       
        amp = len(area)
    elif channel != 2:       
        mean = np.mean(area)
        std = np.std(area)
        ledge = np.min(area)
        redge = np.max(area)    
        amp = len(area)
    nbins = 100        
    if ledge < 0 or ledge < mean/4:
        ledge = mean/4.      
    if channel == 0:
        pmt = 'LV1414'
    elif channel == 1:
        pmt = 'LV2415'
    elif channel == 2:
        pmt = 'LV2414 Dynode'
    tile = r'{}'.format(pmt)
    s2_mu, s2_sigma =  analysis_data.plot_fit_histgram_vs_Gaussion(
        area,nbins,ledge,redge,p0=[amp,mean,std],file_tag=ftag, 
        xlabel=(r'Ch{}Area (PE)'.format(channel)),title=tile,Save=False)
    return s2_mu, s2_sigma
    


def fit_baseline(df, channel, ftag):
    if check_type(channel) ==True:
        pass
    else:
        print("Please input a valid parameter type with interger value.")
        sys.exit()   
    base = df.Baseline[df.Ch == channel].astype(np.float64).to_numpy()
    mean = np.mean(base)
    std = np.std(base)
    redge = np.max(base)
    ledge = np.min(base)
    amp = len(base)
    nbins = 5           
    if channel == 0:
        pmt = 'LV1414'
    elif channel == 1:
        pmt = 'LV2415'
    elif channel == 2:
        pmt = 'LV2414 Dynode'
    tile = r'{}'.format(pmt)
    base_mu, base_sigma =  analysis_data.plot_fit_histgram_vs_Gaussion(
        base,nbins,ledge,redge,p0=[amp,mean,std],file_tag=ftag, 
        xlabel=(r'Ch{}Area (PE)'.format(channel)),title=tile,Save=False)
    return base_mu, base_sigma
   

import analysis_data
import runinfo
def wf_array(file_list, Channel='Anode', tag=''):
    config = []
    waveform_dictionary={}
    ftag = ''
    with open(file_list, 'r') as list:
        for line in list: 
            file = line.rstrip('\n')
            index = file.find('reference') 
            if tag == 'Rd7':
                if index != -1:
                    ftag = 'Ref'
                    config_tag=file.split('base_')[1].split('_run')[0]                    
                elif index == -1:             
                    ftag = file.split('Rd7_')[1].split('_run')[0]
                    config_tag=file.split('base_')[1].split('_run')[0]                    
            if tag == 'Rd':
                if index != -1:
                    ftag = 'Ref'
                    config_tag=file.split('base_')[1].split('_run')[0]                    
                elif index == -1:             
                    ftag = file.split('base_Rd')[1].split('_Riz')[0]
                    config_tag=file.split('base_')[1].split('_run')[0]
            if tag == 'Riz':
                if index != -1:
                    ftag = 'Ref'
                    config_tag=file.split('base_')[1].split('_run')[0]                    
                elif index == -1:             
                    ftag = file.split('_Riz')[1].split('_Cc10nf')[0]
                    config_tag=file.split('base_')[1].split('_run')[0]
            if tag == 'Dt':
                ftag=file.split('680mv_')[1].split('_50hz')[0]
                config_tag=file.split('combine_')[0]+'combine_'+file.split('combine_')[1].split('680mv_')[0]+'680mv_'
                config_tag = config_tag.split('lv2414_')[1]
            if tag == '':
                print('No tag is given, this is just check wf feature')
                # ftag=file.split('base_')[1].split('_run')[0]
                ftag=file.split('outnpy/')[1].split('.h5py')[0] 
                # config_tag=file.split('outnpy/')[1].split('.h5py')[0]  
                config_tag = runinfo.determine_runtype(file)               
            mean_, std_ = analysis_data.calculate_fullwf_mean_std(file, threshold=100, Channel=Channel)
            waveform_dictionary[ftag] = {'mean_wf':mean_, 'std_wf':std_}
            config.append(config_tag) 
    return waveform_dictionary, config 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = plt.cm.rainbow(np.linspace(0, 1, 9))
fig, ax = plt.subplots() 

def plot_waveform(mean_wf, std_wf, cmap_index, delta_t):
    """plot waveform 
    parameter:
        mean_wf (np.array): mean value of waveform.
        std_wf (np.array): standard deviation of waveform, same length with mean_wf.
        Channel (str): 'Anode or Dynode'
        LED_config (str): '1p8v_900mv'
    """ 
    x = np.arange(len(mean_wf))  
    ax.fill_between(x, mean_wf - std_wf, mean_wf + std_wf, color=cmap[cmap_index], alpha=0.3)  
    ax.plot(x, mean_wf, color=cmap[cmap_index], label=delta_t)  
    ax.set_xlabel('Sample Index[4ns]')
    ax.set_ylabel('Amplitude[ADC]')  

from scipy.optimize import curve_fit

def linear_fit(x_data, y_data):
    import numpy as np
    coefficients = np.polyfit(x_data, y_data, 1) 
    return coefficients

def log_model(x, a, b):
    return a * np.log(x) + b
def linear_model(x, m, c):
    return m * x + c

def linear_model_fit(x, y):
    popt, pcov = curve_fit(linear_model, x, y)
    return popt, pcov

def log_model_fit(x, y):
    popt, pcov = curve_fit(log_model, x, y)
    return popt, pcov