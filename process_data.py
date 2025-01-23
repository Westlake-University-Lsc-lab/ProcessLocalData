import pandas as pd
import numpy as np
import daw_readout
import time
import os

### read data and clculate start, min, end index ###
def pusle_index(waveform_data):
    ### find min index ###
    min_index = 0
    min_value = waveform_data[0]
    for i in range(1, len(waveform_data)-1):
        if waveform_data[i] < min_value:
            min_value = waveform_data[i]
            min_index = i
    
    ### searching from min index position to left till back to baseline 
    start_index = min_index
    while start_index > 0 and (waveform_data[start_index] - waveform_data[start_index-1]) < 0  :
        start_index -=1         
     
    ### searching form min index position to right till back to baseline
    end_index = min_index
    if (waveform_data[min_index] == waveform_data[min_index+1]):
        end_index = min_index+1    
    while end_index < len(waveform_data)-1 and (waveform_data[end_index] - waveform_data[end_index+1]) < 0  :
        end_index +=1       
        
    return start_index, min_index, end_index

def find_waveform_intersections(waveform, baseline, negative_pulse=True, percent=0.1):
    if negative_pulse:
        waveform_below_baseline = waveform[waveform < baseline]
        if len(waveform_below_baseline) == 0:
            return None, None, None
        peak_value = np.min(waveform_below_baseline)
        peak_index = np.argmin(waveform)
        height = peak_value - baseline
        threshold = baseline + percent * height
    crossings = np.where(np.diff(np.sign(waveform - threshold)) != 0)[0]
    if len(crossings) == 0:
        return peak_index, None, None
    min_index = crossings[0]
    max_index = crossings[-1]
    return peak_index, min_index, max_index


### calculate area of puse with dynamic range ###
pe_fact  = (2./16384)*4.e-9/(50*1.6e-19)/1.e6  ## to PE
def pulse_area(
    waveform_data,
    st: 'int',
    ed: 'int',
    baseline: 'int'
):
    sum = np.sum( waveform_data[st: ed])
    area = baseline  * (ed - st) - sum  ### adc
    return area
    #return area*pe_fact

### calculate area of pulse with fixed width ###
def pulse_area_fix_len(
    waveform_data,
    minp: 'int',
    fix_len: 'int',
    baseline : 'int',        
):
    sum = np.sum( waveform_data[minp-fix_len : minp+fix_len])
    area = baseline * fix_len*2 - sum
    #pe_fact  = (2./16384)*4.e-9/(50*1.6e-19)/1.e6  ## to PE
    return area

def pulse_area_fix_range(
    waveform_data,
    left: 'int',
    right: 'int',
    baseline : 'int',        
):
    sum = np.sum( waveform_data[left: right])
    area = baseline * (right - left) - sum
    return area

def write_to_hdf5(df, filename):
    start_time = time.time()
    df.to_hdf(filename, key='winfo', mode='w', complib='blosc:blosclz', complevel=9)  
    write_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    print(r"h5 Write Time: {:.2f} s ".format(write_time))
    print(r"h5 File Size: {:.2f} MB".format( file_size/(1024*1024)) )
    print("Save to {}".format(filename))
    return write_time,  file_size

def cal_rms(data):
    import math
    squared_sum = sum(element**2 for element in data)
    average_squared = squared_sum / len(data)
    rms = math.sqrt(average_squared)
    return rms