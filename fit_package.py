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
    fdate = file.split('LED_')[0].split('lv2414_')[1]
    led_config = file.split('LED_')[1].split('.h5py')[0]
    file_tag = fdate + led_config
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
        ledge = 0
        redge = np.max(area)
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
            if time_space == '1us':
                mean_1us, std_1us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['1us'] = {'mean_wf':mean_1us, 'std_wf':std_1us}
            elif time_space == '2us':
                mean_2us, std_2us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['2us'] = {'mean_wf':mean_2us, 'std_wf':std_2us}
            elif time_space == '5us':
                mean_5us, std_5us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['5us'] = {'mean_wf':mean_5us, 'std_wf':std_5us}
            elif time_space == '10us':
                mean_1us, std_1us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100, Channel=Channel)
                waveform_dictionary['1us'] = {'mean_wf':mean_1us, 'std_wf':std_1us}
            elif time_space == '20us':
                mean_20us, std_20us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,Channel=Channel)
                waveform_dictionary['20us'] = {'mean_wf':mean_20us, 'std_wf':std_20us}
            elif time_space == '50us':
                mean_50us, std_50us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['50us'] = {'mean_wf':mean_50us, 'std_wf':std_50us}
            elif time_space == '100us':
                mean_100us, std_100us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['100us'] = {'mean_wf':mean_100us, 'std_wf':std_100us}
            elif time_space == '200us' :
                mean_200us, std_200us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['200us'] = {'mean_wf':mean_200us, 'std_wf':std_200us}
            elif time_space == '500us':
                mean_500us, std_500us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['500us'] = {'mean_wf':mean_500us, 'std_wf':std_500us}
            elif time_space == '1ms' :
                mean_1ms, std_1ms = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['1ms'] = {'mean_wf':mean_1ms, 'std_wf':std_1ms}
            elif time_space == '10ms' :
                mean_10ms, std_10ms = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)    
                waveform_dictionary['10ms'] = {'mean_wf':mean_10ms, 'std_wf':std_10ms}
            elif time_space == '1000us':
                mean_1000us, std_1000us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100,  Channel=Channel)
                waveform_dictionary['1000us'] = {'mean_wf':mean_1000us, 'std_wf':std_1000us}
            elif time_space == '10000us':
                mean_10000us, std_10000us = analysis_data.calculate_wf_mean_std_s2(file, threshold=100, Channel=Channel)
                waveform_dictionary['10000us'] = {'mean_wf':mean_10000us, 'std_wf':std_10000us}
    led_config = led_config.split('lv2414_')[1]
    print(led_config)  
    
    return waveform_dictionary, led_config   

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