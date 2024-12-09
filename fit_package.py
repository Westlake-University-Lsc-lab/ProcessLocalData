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
    
   
