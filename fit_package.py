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
    area = df.Area_S2[df.Ch == channel].astype(np.float64).to_numpy()
    if channel == 2:
        area = -area
        mean = np.mean(area)
        std = np.std(area)
        redge = np.max(area)
        ledge = np.min(area)
        if ledge < 0:
            ledge = redge /2.      
        amp = len(area)
    elif channel != 2:       
        mean = np.mean(area)
        std = np.std(area)
        ledge = np.min(area)
        redge = np.max(area)    
        amp = len(area)
    nbins = 100              
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
    
def fit_S1_channel(df, channel, ftag):
    if check_type(channel) ==True:
        pass
    else:
        print("Please input a valid parameter type with interger value.")
        sys.exit()   
    area = df.Area_S1[df.Ch == channel].astype(np.float64).to_numpy()
    if channel == 2:
        area = -area
        mean = np.mean(area)
        std = np.std(area)
        redge = np.max(area)
        ledge = np.min(area)
        if ledge < 0:
            ledge = redge /2.      
        amp = len(area)
    elif channel != 2:       
        mean = np.mean(area)
        std = np.std(area)
        ledge = np.min(area)
        redge = np.max(area)    
        amp = len(area)
    nbins = 100              
    if channel == 0:
        pmt = 'LV1414'
    elif channel == 1:
        pmt = 'LV2415'
    elif channel == 2:
        pmt = 'LV2414 Dynode'
    tile = r'{}'.format(pmt)
    s1_mu, s1_sigma =  analysis_data.plot_fit_histgram_vs_Gaussion(
        area,nbins,ledge,redge,p0=[amp,mean,std],file_tag=ftag, 
        xlabel=(r'Ch{}Area (PE)'.format(channel)),title=tile,Save=False)
    return s1_mu, s1_sigma


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
def calculate_wf_data_array(file_list, Channel='Anode'):
    # time_delay_map = {'5us':5, '200us':200, '1ms':1000, '10ms':10000}  ## time delay unit on 'us'
    led_config = ''
    waveform_dictionary={}
    with open(file_list, 'r') as list:
        for line in list: 
            file = line.rstrip('\n')
            time_space=file.split('680mv_')[1].split('_50hz')[0]
            led_config=file.split('combine_')[0]+'combine_'+file.split('combine_')[1].split('680mv_')[0]+'680mv_'
            mean_, std_ = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
            waveform_dictionary[time_space] = {'mean_wf':mean_, 'std_wf':std_}
    led_config = led_config.split('lv2414_')[1]
    print(led_config)  
    
    return waveform_dictionary, led_config   

def wf_array_S1(file_list, Channel='Anode'):
    # time_delay_map = {'5us':5, '200us':200, '1ms':1000, '10ms':10000}  ## time delay unit on 'us'
    base_config = []
    waveform_dictionary={}
    Rd7 = ''
    with open(file_list, 'r') as list:
        for line in list: 
            file = line.rstrip('\n')
            config_tag=file.split('base_')[1].split('_run')[0]
            index = file.find('reference') 
            if index != -1:
                Rd7 = 'Ref'
            elif index == -1:             
                Rd7 = file.split('Rd7_')[1].split('_run')[0]
            mean_, std_ = analysis_data.calculate_wf_mean_std_s2(file, threshold=100, Channel=Channel)
            waveform_dictionary[Rd7] = {'mean_wf':mean_, 'std_wf':std_}
            # print(Rd7)           
            base_config.append(config_tag) 
    return waveform_dictionary, base_config  

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap('tab10')  
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
    ax.fill_between(x, mean_wf - std_wf, mean_wf + std_wf, color=cmap(cmap_index), alpha=0.3)  
    ax.plot(x, mean_wf, color=cmap(cmap_index), label=delta_t)  
    ax.set_xlabel('Sample Index[4ns]')
    ax.set_ylabel('Amplitude[ADC]')  
    ax.legend()  