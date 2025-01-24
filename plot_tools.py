import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = plt.cm.rainbow(np.linspace(0, 1, 9))

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

def plot_wf_array(flist, Channel='Anode'):
    wf_dic = {}
    with open(flist, 'r') as list:
        for line in list: 
            file = line.rstrip('\n')  
            runtag = runinfo.find_run_tag(file)           
            mean, std = analysis_data.calculate_fullwf_mean_std(file, threshold=100, Channel=Channel)
            wf_dic[runtag] = {'mean':mean,'std':std}            
    i = 0
    for tag, data in wf_dic.items():
        plot_waveform(data['mean'], data['std'], i , tag)
        i +=1
    plt.xlabel('Sample Index[4ns]')
    plt.ylabel('Amplitude[ADC]')
    plt.legend(loc='upper right')   
    plt.show()

def plot_waveform(mean_wf, std_wf, index, delta_t):
    """plot waveform 
    parameter:
        mean_wf (np.array): mean value of waveform.
        std_wf (np.array): standard deviation of waveform, same length with mean_wf.
        Channel (str): 'Anode or Dynode'
        LED_config (str): '1p8v_900mv'
    """ 
    x = np.arange(len(mean_wf))  
    plt.fill_between(x, mean_wf - std_wf, mean_wf + std_wf, color=cmap[index], alpha=0.3)  
    plt.plot(x, mean_wf, color=cmap[index], label=delta_t)  
    plt.set_xlabel('Sample Index[4ns]')
    plt.set_ylabel('Amplitude[ADC]')  
    
    

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
    
    
def plot_PEns(f, index, DyOption=False ):
    df = pd.read_hdf(f, key='winfo')
    plt.errorbar(df.PEns_filter, df.PEns_anode,
                 xerr=df.PEns_filter_err, yerr= df.PEns_anode_err,
                 fmt='.', markersize=10, color=cmap[index], alpha=0.5, 
                 capsize=3, elinewidth=2, capthick=2, label=df.RunTag.values[0])
    if DyOption is True:
        plt.errorbar(df.PEns_filter, df.PEns_dynode,
                     xerr=df.PEns_filter_err, yerr= df.PEns_dynode_err,
                     fmt='.', markersize=10, color=cmap[index], alpha=0.9, 
                     capsize=3, elinewidth=2, capthick=2, label=df.RunTag.values[0])
        
    plt.ylabel('Saturation-mitigated[PE/ns]')
    plt.xlabel('Monitor[PE/ns]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(-1.E1, 3.e3)
    plt.ylim(-1.E2, 3.e3)
    plt.legend(loc='upper left')
    plt.show()
    

